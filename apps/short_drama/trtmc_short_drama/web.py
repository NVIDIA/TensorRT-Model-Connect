# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The web app: projects, steps, flow versions and settings over a small JSON API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import logging
from pathlib import Path
import re
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import editing, media, planning, prompts
from .engine import Engine, Served
from .flow import Context, Flows, Steps
from .jobs import Job, Jobs
from .llm import ChatClient
from .store import Project, Store, read_json, write_json, write_secret

log = logging.getLogger(__name__)

STATIC = Path(__file__).with_name("static")
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ah_q_zheng_zhuan.txt"
CORE_STEPS = [planning.STEP, media.STILLS, media.SHOTS, editing.SUBTITLES, editing.ASSEMBLE]
WRITE_HEADER = "x-short-drama"
TOKEN_COOKIE, TOKEN_HEADER = "short_drama_token", "x-short-drama-token"
PLAN_ID = re.compile(r"[a-z0-9_]{1,40}")


@dataclass
class Services:
    """What steps use: the engine, the served models and the optional external chat API."""

    engine: Engine
    models: dict[str, tuple[str, Served]]  # name -> (role, served bundle)
    data: Path

    def load(self, name: str | None) -> None:
        if name is None or name not in self.models:
            raise ValueError(f"model {name!r} is not served; see the Models list in Settings")
        role, served = self.models[name]
        wanted = [served]
        if role == "video":  # speech recognition is small; keep it with the video model
            wanted += [s for r, s in self.models.values() if r == "asr"]
        self.engine.ensure(wanted)

    def external(self) -> dict[str, str]:
        return read_json(self.data / "settings.json", {}).get("external_chat", {})

    def api_key(self) -> str:
        """The saved key, only for the endpoint it was entered with."""

        secrets = read_json(self.data / "secrets.json", {})
        same = secrets.get("external_chat_base_url") == self.external().get("base_url")
        return secrets.get("external_chat_api_key", "") if same else ""

    def chat(self, name: str | None) -> ChatClient:
        external = self.external()
        if external.get("base_url") and external.get("model"):
            return ChatClient(external["base_url"], external["model"], self.api_key(), local=False)
        self.load(name)
        return ChatClient(self.engine.url, name)


def check_plan(plan: dict[str, Any]) -> None:
    """Ids become file names, so an edited plan may only use short lowercase ids."""

    if not isinstance(plan.get("episodes"), list) or not isinstance(plan.get("characters"), list):
        raise HTTPException(400, "a plan has characters and episodes")
    for group in ("characters", "locations", "props"):
        for item in plan.get(group) or []:
            if not isinstance(item, dict) or not PLAN_ID.fullmatch(str(item.get("id", ""))):
                raise HTTPException(400, f"{group} ids use a-z, 0-9 and _ (at most 40)")
    for episode in plan["episodes"]:
        if not isinstance(episode, dict) or type(episode.get("number")) is not int:
            raise HTTPException(400, "each episode needs an integer number")
        for shot in episode.get("shots") or []:
            if not isinstance(shot, dict) or not PLAN_ID.fullmatch(str(shot.get("id", ""))):
                raise HTTPException(400, "shot ids use a-z, 0-9 and _ (at most 40)")


def create_app(services: Services, *, plugins: Path | None, info: dict[str, Any],
               allowed_hosts: list[str] | None = None, token: str | None = None) -> FastAPI:
    data = services.data
    store = Store(data)
    roles = {}
    for name, (role, _) in services.models.items():
        roles.setdefault(role, name)
    flows = Flows(data / "flows", roles)
    steps = Steps(CORE_STEPS, plugins)

    def execute(job: Job) -> None:
        project = store.get(job.project)
        _, flow = flows.load(project.meta.get("flow", "default"))
        entry = next((e for e in flow["steps"] if e["step"] == job.step), {"params": {}, "model": None})
        step = steps.get(job.step)
        if step is None:
            raise ValueError(f"step {job.step!r} is not installed; reload plugins or edit the flow")

        def progress(message: str) -> None:
            job.message = message

        context = Context(project, {**entry.get("params", {}), **job.params}, entry.get("model"),
                          job.episode, services, progress, job.cancelled)
        project.log(f"run {job.step}" + (f" episode {job.episode}" if job.episode else ""))
        step.run(context)

    jobs = Jobs(execute)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.close()
        services.engine.stop()

    app = FastAPI(title="TRTMC Short Drama", lifespan=lifespan)

    @app.middleware("http")
    async def same_app_only(request: Request, call_next):
        # A page from another site cannot add this header without a CORS preflight, which the
        # app never grants, so it cannot change settings or start jobs.
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get(WRITE_HEADER) != "1":
            return JSONResponse({"detail": f"missing {WRITE_HEADER} header"}, status_code=403)
        return await call_next(request)

    if token:  # listening beyond loopback: every request but static assets needs the token
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if request.url.path == "/" and hmac.compare_digest(request.query_params.get("token", ""), token):
                response = await call_next(request)
                response.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="strict")
                return response
            given = request.cookies.get(TOKEN_COOKIE) or request.headers.get(TOKEN_HEADER, "")
            if not request.url.path.startswith("/static/") and not hmac.compare_digest(given, token):
                return JSONResponse({"detail": "open the URL with ?token= printed at startup"}, status_code=401)
            return await call_next(request)

    if allowed_hosts:  # refuse other Host names, which a DNS-rebinding page would send
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.state.jobs = jobs
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def project_or_404(project_id: str) -> Project:
        try:
            return store.get(project_id)
        except KeyError:
            raise HTTPException(404, "project not found") from None

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        external = services.external()
        return {**info, "models": [{"name": n, "role": r, "bundle": s.bundle.name}
                                   for n, (r, s) in services.models.items()],
                "loaded": services.engine.loaded, "projects": store.projects(), "jobs": jobs.views(),
                "steps": steps.describe(), "plugin_errors": steps.errors,
                "flows": {name: flows.versions(name) for name in flows.names()},
                "external_chat": {"base_url": external.get("base_url", ""), "model": external.get("model", ""),
                                  "has_key": bool(services.api_key())},
                "sample": SAMPLE.is_file()}

    @app.post("/api/projects")
    def create(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        novel = SAMPLE.read_text(encoding="utf-8") if body.get("sample") else str(body.get("novel", ""))
        try:
            reference, _ = flows.load(str(body.get("flow") or "default"))
            project = store.create(str(body.get("title") or "drama")[:80], novel, flow=reference,
                                   episodes=max(1, min(20, int(body.get("episodes", 4)))),
                                   max_seconds=max(10, min(600, int(body.get("max_seconds", 60)))))
        except (KeyError, ValueError) as error:
            raise HTTPException(400, str(error)) from None
        return {"id": project.id}

    @app.get("/api/projects/{project_id}")
    def project_view(project_id: str) -> dict[str, Any]:
        project = project_or_404(project_id)
        plan = project.plan
        episodes = {str(e["number"]): {"continues": media.continues(plan, e["number"]),
                                       **project.shot_state(e["number"])} for e in plan.get("episodes", [])}
        _, flow = flows.load(project.meta.get("flow", "default"))
        flow_steps = [{"step": entry["step"], "installed": steps.get(entry["step"]) is not None,
                       "scope": getattr(steps.get(entry["step"]), "scope", None),
                       "core": entry["step"] in steps.core} for entry in flow["steps"]]
        previews = {}  # the video prompt each shot would get now, before it is rendered
        for episode in plan.get("episodes", []):
            for shot in episode.get("shots", []):
                references, _ = media.shot_references(project, plan, shot, previous=None, require_approval=False)
                previews[shot["id"]] = prompts.video_prompts(plan, shot, references)[0]
        return {"id": project.id, "meta": project.meta, "plan": plan, "stills": project.stills,
                "episodes": episodes, "novel_chars": len(project.novel), "flow_steps": flow_steps,
                "previews": previews}

    @app.put("/api/projects/{project_id}/plan")
    def save_plan(project_id: str, plan: dict[str, Any] = Body(...)) -> dict[str, str]:
        project = project_or_404(project_id)
        check_plan(plan)
        project.save_plan(plan)
        return {"status": "saved"}

    @app.put("/api/projects/{project_id}/stills/{key}")
    def update_still(project_id: str, key: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        project = project_or_404(project_id)
        stills = project.stills
        if key not in stills:
            raise HTTPException(404, "still not found")
        if body.get("status") in {"approved", "draft"}:
            stills[key]["status"] = body["status"]
        if "prompt_override" in body:
            stills[key]["prompt_override"] = str(body["prompt_override"] or "")
        project.save_stills(stills)
        return stills[key]

    @app.put("/api/projects/{project_id}/episodes/{number}")
    def update_episode(project_id: str, number: int, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        project = project_or_404(project_id)
        state = project.shot_state(number)
        if "use_previous_frame" in body:
            state["use_previous_frame"] = bool(body["use_previous_frame"])
        for shot_id, prompt in (body.get("prompt_overrides") or {}).items():
            state.setdefault(shot_id, {})["prompt_override"] = str(prompt or "")
        for shot_id, take in (body.get("use_take") or {}).items():
            chosen = next((t for t in state.get(shot_id, {}).get("takes", []) if t["take"] == take), None)
            if chosen is None:
                raise HTTPException(404, f"{shot_id} has no take {take}")
            state[shot_id].pop("speech", None)
            state[shot_id].update(chosen)
            state.pop("assembled", None)
        project.save_shot_state(number, state)
        return state

    @app.put("/api/projects/{project_id}/flow")
    def select_flow(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        project = project_or_404(project_id)
        try:
            reference, _ = flows.load(str(body.get("flow", "")))
        except KeyError as error:
            raise HTTPException(400, str(error)) from None
        return project.update_meta(flow=reference)

    @app.post("/api/projects/{project_id}/run")
    def run(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        project = project_or_404(project_id)
        _, flow = flows.load(project.meta.get("flow", "default"))
        names = [entry["step"] for entry in flow["steps"]]
        episode = body.get("episode")
        if body.get("step") == "episode":  # every episode step of the flow, in order
            queued = [jobs.submit(project.id, name, int(episode), {}) for name in names
                      if (step := steps.get(name)) and step.scope == "episode"]
            return {"jobs": [job.id for job in queued]}
        if body.get("step") not in names:
            raise HTTPException(400, "that step is not part of this project's flow")
        step = steps.get(str(body.get("step")))
        if step is None:
            raise HTTPException(400, f"step {body.get('step')} is not installed; reload plugins or edit the flow")
        if step.scope == "episode" and not isinstance(episode, int):
            raise HTTPException(400, "choose an episode")
        params = {key: value for key, value in (body.get("params") or {}).items() if key == "only"}
        return {"jobs": [jobs.submit(project.id, step.name, episode, params).id]}

    @app.post("/api/jobs/cancel")
    def cancel() -> dict[str, str]:
        jobs.cancel()
        return {"status": "cancelling"}

    @app.get("/api/projects/{project_id}/files/{relative:path}")
    def file(project_id: str, relative: str) -> FileResponse:
        try:
            return FileResponse(project_or_404(project_id).file(relative))
        except FileNotFoundError:
            raise HTTPException(404, "file not found") from None

    @app.get("/api/flows/{reference}")
    def get_flow(reference: str) -> dict[str, Any]:
        try:
            canonical, flow = flows.load(reference)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None
        return {"reference": canonical, "flow": flow}

    @app.post("/api/flows/{name}")
    def save_flow(name: str, body: dict[str, Any] = Body(...)) -> dict[str, str]:
        try:
            return {"reference": flows.save(name, body.get("flow"), note=str(body.get("note", ""))[:200])}
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from None

    @app.post("/api/plugins/reload")
    def reload_plugins() -> dict[str, Any]:
        steps.reload()
        return {"steps": steps.describe(), "errors": steps.errors}

    @app.put("/api/settings/external-chat")
    def external_chat(body: dict[str, Any] = Body(...)) -> dict[str, str]:
        base_url = str(body.get("base_url", "")).strip()
        if base_url and not re.fullmatch(r"https?://[^\s]+", base_url):
            raise HTTPException(400, "the base URL must start with http:// or https://")
        settings = read_json(data / "settings.json", {})
        settings["external_chat"] = {"base_url": base_url, "model": str(body.get("model", "")).strip()}
        write_json(data / "settings.json", settings)
        secrets = read_json(data / "secrets.json", {})
        if body.get("api_key"):
            secrets = {"external_chat_api_key": str(body["api_key"]), "external_chat_base_url": base_url}
        elif secrets.get("external_chat_base_url") != base_url:
            secrets = {}  # a saved key never follows a new endpoint
        write_secret(data / "secrets.json", secrets)
        return {"status": "saved"}

    return app
