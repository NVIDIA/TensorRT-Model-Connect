# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import stat
import time

from fastapi.testclient import TestClient

from trtmc_short_drama.engine import Served
from trtmc_short_drama.web import Services, create_app

PLAN_STEPS = [{"location": "temple", "characters": ["aq"], "speaker": "aq", "line": "滚出去！", "seconds": 6}]
BIBLE = {"title": "阿Q正传", "characters": [{"id": "aq", "name": "阿Q", "look": "thin"}],
         "locations": [{"id": "temple", "name": "土谷祠", "look": "a shrine"}],
         "episodes": [{"number": 1, "title": "一", "summary": "s", "chapters": [1]}]}


class FakeEngine:
    url = "http://127.0.0.1:1"
    loaded: list[str] = []

    def __init__(self):
        self.ensured = []

    def ensure(self, models):
        self.ensured.append(sorted(m.name for m in models))

    def status(self):
        return {"loaded": [], "loading": [], "since": None, "failure": ""}

    def log_tail(self, lines=200):
        return ["loading v", "ready"][-lines:]

    def stop(self):
        pass


STORY = "第一章 手机\n" + "他捡起一部手机，屏幕亮了。" * 20


class FakeChat:
    def complete(self, system, user, **_):
        return STORY

    def complete_json(self, system, user, **_):
        if user.startswith("Summarize"):
            return {"summary": "x"}
        if user.startswith("Rewrite one shot"):
            return {**PLAN_STEPS[0], "line": "你也配姓赵！"}
        if user.startswith("Review"):
            return [{"shot": "e1s1", "problem": "台词太凶", "fix": "语气缓和"}]
        return BIBLE if user.startswith("Plan") else PLAN_STEPS


def make_client(tmp_path: Path, monkeypatch, allowed_hosts=None):
    services = Services(FakeEngine(), {"q": ("llm", Served("q", tmp_path / "q.bundle")),
                                       "v": ("video", Served("v", tmp_path / "v.bundle")),
                                       "w": ("asr", Served("w", tmp_path / "w.bundle"))}, tmp_path)
    monkeypatch.setattr(Services, "chat", lambda self, name: FakeChat())
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    app = create_app(services, plugins=plugins, info={"gpu": "test", "tier": "a"}, allowed_hosts=allowed_hosts)
    return TestClient(app, headers={"X-Short-Drama": "1"}), services


def wait(client):
    for _ in range(300):
        jobs = client.get("/api/state").json()["jobs"]
        if jobs and all(j["status"] not in {"queued", "running"} for j in jobs):
            return jobs
        time.sleep(0.02)
    raise AssertionError("jobs did not finish")


def test_create_plan_edit_and_select_flow(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        state = client.get("/api/state").json()
        assert state["flows"] == {"default": [1]} and state["sample"] is True
        created = client.post("/api/projects", json={"title": "AQ", "sample": True, "episodes": 1,
                                                     "max_seconds": 30}).json()
        project = client.get(f"/api/projects/{created['id']}").json()
        assert project["novel_chars"] > 8000 and project["meta"]["flow"] == "default@1"

        assert client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"}).status_code == 200
        jobs = wait(client)
        assert jobs[0]["status"] == "done", jobs
        plan = client.get(f"/api/projects/{created['id']}").json()["plan"]
        assert plan["episodes"][0]["shots"][0]["line"] == "滚出去！"

        assert "<d>[Chinese] 滚出去！</d>" in client.get(f"/api/projects/{created['id']}").json()["previews"]["e1s1"]
        plan["episodes"][0]["shots"][0]["line"] = "你还不配！"
        assert client.put(f"/api/projects/{created['id']}/plan", json=plan).status_code == 200
        assert client.get(f"/api/projects/{created['id']}").json()["plan"]["episodes"][0]["shots"][0]["line"] == "你还不配！"

        flow = client.get("/api/flows/default").json()["flow"]
        saved = client.post("/api/flows/default", json={"flow": flow, "note": "same"}).json()
        assert saved == {"reference": "default@2"}
        assert client.put(f"/api/projects/{created['id']}/flow", json={"flow": "default@2"}).json()["flow"] == "default@2"


def test_episode_steps_need_an_episode_and_shots_need_approved_stills(tmp_path, monkeypatch):
    client, services = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "第一章\n文本", "episodes": 1}).json()
        assert client.post(f"/api/projects/{created['id']}/run", json={"step": "shots"}).status_code == 400
        client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"})
        wait(client)
        client.post(f"/api/projects/{created['id']}/run", json={"step": "shots", "episode": 1})
        jobs = wait(client)
        assert jobs[0]["status"] == "failed" and "approve these reference stills" in jobs[0]["error"]
        assert services.engine.ensured[-1] == ["v", "w"]  # speech recognition stays with the video model


def test_files_stay_inside_the_project_and_keys_stay_private(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文本"}).json()
        assert client.get(f"/api/projects/{created['id']}/files/novel.txt").status_code == 200
        assert client.get(f"/api/projects/{created['id']}/files/../../settings.json").status_code == 404
        assert client.get("/api/projects/..%2f..%2fetc/files/passwd").status_code == 404
        client.put("/api/settings/external-chat", json={"base_url": "https://example.invalid/v1",
                                                        "model": "m", "api_key": "secret-key"})
        state = client.get("/api/state").json()
        assert state["external_chat"] == {"base_url": "https://example.invalid/v1", "model": "m", "vision_model": "",
                                          "has_key": True}
        assert "secret-key" not in json.dumps(state)
        assert stat.S_IMODE((tmp_path / "secrets.json").stat().st_mode) == 0o600


def test_speech_recognition_loads_with_a_speaking_video_model_only(tmp_path):
    models = {"w": ("asr", Served("w", tmp_path / "w.bundle")), "v": ("video", Served("v", tmp_path / "v.bundle"))}
    speaking = Services(FakeEngine(), models, tmp_path)
    speaking.load("v")
    silent = Services(FakeEngine(), models, tmp_path, first_frames=frozenset({"v"}))
    silent.load("v")  # on a 24 GB card Wan2.2 and Whisper would not fit together
    assert speaking.engine.ensured == [["v", "w"]] and silent.engine.ensured == [["v"]]


def test_a_vision_model_named_in_settings_comes_before_the_one_served_here(tmp_path):
    services = Services(FakeEngine(), {"w": ("asr", Served("w", tmp_path / "w.bundle")),
                                       "see": ("vision", Served("see", tmp_path / "see.bundle")),
                                       "v": ("video", Served("v", tmp_path / "v.bundle"))}, tmp_path)
    local = services.vision()
    assert (local.model, local.local, services.vision_name()) == ("see", True, "see")
    assert services.engine.ensured == [["see", "w"]]  # with speech recognition, which the analysis uses too
    (tmp_path / "settings.json").write_text(json.dumps({"external_chat": {"base_url": "https://example.invalid/v1",
                                                                           "model": "", "vision_model": "api-vl"}}))
    remote = services.vision()
    assert (remote.model, remote.local, services.vision_name()) == ("api-vl", False, "api-vl")
    assert Services(FakeEngine(), {}, tmp_path / "none").vision() is None


def test_choosing_a_take_switches_the_clip_and_drops_its_timing(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文本"}).json()
        state_file = tmp_path / "projects" / created["id"] / "episodes" / "01" / "shots.json"
        state_file.parent.mkdir(parents=True)
        takes = [{"take": 1, "file": "c1.mp4", "sheet": "s1.jpg"},
                 {"take": 2, "file": "c2.mp4", "sheet": "s2.jpg", "silent": True, "first_frame": "f.png", "keyframe": None}]
        state_file.write_text(json.dumps({"assembled": {}, "e1s1": {**takes[1], "takes": takes, "speech": {}}}))
        state = client.put(f"/api/projects/{created['id']}/episodes/1", json={"use_take": {"e1s1": 1}}).json()
        assert state["e1s1"]["file"] == "c1.mp4" and "speech" not in state["e1s1"] and "assembled" not in state
        assert not {"silent", "first_frame", "keyframe"} & set(state["e1s1"])  # take 1 was filmed with sound
        assert client.put(f"/api/projects/{created['id']}/episodes/1", json={"use_take": {"e1s1": 3}}).status_code == 404


def test_a_video_model_that_animates_a_first_frame_previews_its_motion_prompt(tmp_path, monkeypatch):
    client, services = make_client(tmp_path, monkeypatch)
    services.first_frames = frozenset({"v"})
    with client:
        created = client.post("/api/projects", json={"title": "AQ", "sample": True, "episodes": 1,
                                                     "max_seconds": 30}).json()
        assert client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"}).status_code == 200
        assert wait(client)[0]["status"] == "done"
        preview = client.get(f"/api/projects/{created['id']}").json()["previews"]["e1s1"]
        assert preview.startswith("A vertical short-drama shot") and "<d>" not in preview


def test_previous_frame_is_used_only_when_the_place_continues():
    from trtmc_short_drama.media import continues

    plan = {"episodes": [{"number": 1, "shots": [{"location": "a"}, {"location": "b"}]},
                         {"number": 2, "shots": [{"location": "b"}]}, {"number": 3, "shots": [{"location": "c"}]}]}
    assert continues(plan, 2) and not continues(plan, 3) and not continues(plan, 1)


def test_writes_need_the_app_header_and_a_local_host(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch, allowed_hosts=["testserver", "127.0.0.1"])
    with client:
        assert client.post("/api/projects", json={"title": "x", "novel": "文"},
                           headers={"X-Short-Drama": ""}).status_code == 403
        assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
        assert client.post("/api/projects", json={"title": "x", "novel": "文"}).status_code == 200


def test_edited_plans_cannot_use_ids_that_escape_the_project(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文"}).json()
        bad = {"characters": [{"id": "../../evil"}], "episodes": []}
        assert client.put(f"/api/projects/{created['id']}/plan", json=bad).status_code == 400
        shot = {"characters": [{"id": "aq"}], "episodes": [{"number": 1, "shots": [{"id": "../x"}]}]}
        assert client.put(f"/api/projects/{created['id']}/plan", json=shot).status_code == 400


def test_a_saved_key_never_follows_a_new_endpoint(tmp_path, monkeypatch):
    client, services = make_client(tmp_path, monkeypatch)
    with client:
        client.put("/api/settings/external-chat", json={"base_url": "https://a.example/v1", "model": "m",
                                                        "api_key": "secret-key"})
        assert services.api_key() == "secret-key"
        client.put("/api/settings/external-chat", json={"base_url": "https://b.example/v1", "model": "m"})
        assert services.api_key() == "" and client.get("/api/state").json()["external_chat"]["has_key"] is False
        assert client.put("/api/settings/external-chat", json={"base_url": "file:///etc", "model": "m"}).status_code == 400



def test_without_a_language_model_the_plan_points_to_settings(tmp_path):
    services = Services(FakeEngine(), {"v": ("video", Served("v", tmp_path / "v.bundle"))}, tmp_path)
    app = create_app(services, plugins=None, info={"gpu": "test", "tier": "a"})
    with TestClient(app, headers={"X-Short-Drama": "1"}) as client:
        state = client.get("/api/state").json()
        assert state["chat_ready"] is False
        assert state["free_apis"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
        created = client.post("/api/projects", json={"title": "t", "novel": "第一章\n文"}).json()
        url = f"/api/projects/{created['id']}"
        assert client.get(url).json()["stages"][0]["next"] == "Choose a language model in Settings"
        client.post(f"{url}/run", json={"step": "plan"})
        assert "choose a chat API in Settings" in wait(client)[0]["error"]

        nvidia = state["free_apis"]["nvidia"]
        client.put("/api/settings/external-chat", json={"base_url": nvidia["base_url"], "model": nvidia["models"][0],
                                                        "api_key": "k"})
        assert client.get("/api/state").json()["chat_ready"] is True
        assert client.get(url).json()["stages"][0]["next"] == "Plan the drama from the novel"
        chat = services.chat(None)
        assert (chat.base_url, chat.model, chat.api_key, chat.local) == (
            "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4.1-flash", "k", False)


def test_a_failed_episode_step_does_not_cancel_other_episodes():
    from trtmc_short_drama.jobs import Jobs

    def execute(job):
        if job.step == "subtitles":
            raise RuntimeError("boom")

    jobs = Jobs(execute)
    try:
        submitted = [jobs.submit("p", step, episode, {}) for step, episode in
                     (("subtitles", 1), ("assemble", 1), ("shots", 2), ("assemble", 2))]
        for _ in range(200):
            if all(j.status not in {"queued", "running"} for j in submitted):
                break
            time.sleep(0.01)
        assert [j.status for j in submitted] == ["failed", "cancelled", "done", "done"]
    finally:
        jobs.close()


def test_a_non_loopback_app_needs_its_token(tmp_path, monkeypatch):
    services = Services(FakeEngine(), {}, tmp_path)
    app = create_app(services, plugins=None, info={"gpu": "test", "tier": "a"}, token="s3cret")
    with TestClient(app, headers={"X-Short-Drama": "1"}) as client:
        assert client.get("/api/state").status_code == 401
        assert client.get("/?token=wrong").status_code == 401
        assert client.get("/?token=s3cret").status_code == 200  # sets the cookie
        assert client.get("/api/state").status_code == 200
    with TestClient(app) as other:
        assert other.get("/api/state", headers={"X-Short-Drama-Token": "s3cret"}).status_code == 200
        assert other.get("/static/app.css").status_code == 200



def test_an_idea_project_writes_its_story_then_plans_and_revises(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        assert client.post("/api/projects", json={"title": "x"}).status_code == 400
        created = client.post("/api/projects", json={"title": "手机", "idea": "外卖骑手捡到一部手机", "episodes": 1}).json()
        url = f"/api/projects/{created['id']}"
        project = client.get(url).json()
        assert project["novel_chars"] == 0 and project["meta"]["idea"] == "外卖骑手捡到一部手机"
        assert project["stages"][0]["next"] == "Write the story from the idea"
        assert project["model_review"] is None  # the page tests it for truth
        client.post(f"{url}/run", json={"step": "plan"})
        assert "no story yet" in wait(client)[0]["error"]
        client.post(f"{url}/run", json={"step": "story"})
        assert wait(client)[0]["status"] == "done"
        assert client.get(f"{url}/novel").json()["novel"] == STORY
        client.post(f"{url}/run", json={"step": "plan"})
        wait(client)

        project = client.get(url).json()
        assert project["origins"] == {"e1s1": "adapted"} and project["usage"]["location:temple"] == ["e1s1"]
        assert project["stages"][1]["next"] == "Draw 2 stills" and project["estimates"] == {"still": None, "shot": None}
        assert client.post(f"{url}/run", json={"step": "revise", "params": {"target": "shot:../x", "instruction": "a"}}
                           ).status_code == 400
        client.post(f"{url}/run", json={"step": "revise", "params": {"target": "shot:e1s1", "instruction": "换句台词",
                                                                        "only": ["x"], "max_tokens": 1}})
        assert wait(client)[0]["status"] == "done"
        client.post(f"{url}/run", json={"step": "review"})
        wait(client)
        project = client.get(url).json()
        assert project["plan"]["episodes"][0]["shots"][0]["line"] == "你也配姓赵！"
        assert project["model_review"]["current"] and project["model_review"]["items"][0]["fix"] == "语气缓和"
        assert project["history"][0]["note"].startswith("before revising shot e1s1")
        name = project["history"][0]["name"]
        assert client.post(f"{url}/history/{name}/restore").status_code == 200
        assert client.post(f"{url}/history/..%2fplan.json/restore").status_code == 404
        project = client.get(url).json()
        assert project["plan"]["episodes"][0]["shots"][0]["line"] == "滚出去！" and not project["model_review"]["current"]
        assert client.put(f"{url}/novel", json={"novel": " "}).status_code == 400


def test_projects_are_listed_newest_first_renamed_and_deleted(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        first = client.post("/api/projects", json={"title": "一", "novel": "第一章\n文本", "episodes": 1}).json()["id"]
        second = client.post("/api/projects", json={"title": "二", "idea": "小熊猫种星星", "episodes": 1,
                                                    "style": "bright 3D cartoon animation"}).json()["id"]
        client.post(f"/api/projects/{first}/run", json={"step": "plan"})
        wait(client)
        listed = client.get("/api/projects").json()
        assert [p["id"] for p in listed["projects"]] == [first, second]  # the plan changed the first one last
        progress = listed["projects"][0]["progress"]
        assert progress["story"] and progress["planned"] and progress["stills"] == {"needed": 2, "drawn": 0, "approved": 0}
        assert progress["shots"] == {"total": 1, "rendered": 0} and progress["picture"] is None
        assert listed["projects"][1]["progress"]["story"] is False and listed["estimates"] == {"still": None, "shot": None}
        assert client.get(f"/api/projects/{second}").json()["meta"]["style"] == "bright 3D cartoon animation"

        assert client.put(f"/api/projects/{first}/meta", json={"title": " 新名字 "}).json()["title"] == "新名字"
        assert client.put(f"/api/projects/{first}/meta", json={"title": ""}).status_code == 400
        monkeypatch.setattr(client.app.state.jobs, "busy", lambda project=None: True)
        assert client.delete(f"/api/projects/{first}").status_code == 409
        monkeypatch.undo()
        assert client.delete(f"/api/projects/{first}").json() == {"status": "deleted"}
        assert not (tmp_path / "projects" / first).exists()
        assert client.delete("/api/projects/..%2f..%2fsettings").status_code == 404
        assert [p["id"] for p in client.get("/api/state").json()["projects"]] == [second]


def test_the_page_sees_the_gpu_its_server_log_and_each_phase(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        state = client.get("/api/state").json()
        assert state["engine"] == {"loaded": [], "loading": [], "since": None, "failure": ""}
        assert client.get("/api/logs/server?lines=1").json() == {"lines": ["ready"]}
        created = client.post("/api/projects", json={"title": "x", "novel": "第一章\n文本", "episodes": 1}).json()
        client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"})
        job = wait(client)[0]
        assert job["message"] == "writing the shots of episode 1" and job["phase_seconds"] is not None
        assert job["params"] == {} and job["finished"] is not None


def test_the_page_token_survives_a_restart(tmp_path):
    from trtmc_short_drama.cli import app_token

    token = app_token(tmp_path)
    assert app_token(tmp_path) == token and len(token) >= 32
    assert stat.S_IMODE((tmp_path / "app-token.json").stat().st_mode) == 0o600


def test_a_flow_saved_in_api_mode_plans_with_the_local_model(tmp_path):
    engine = FakeEngine()
    services = Services(engine, {"q": ("llm", Served("q", tmp_path / "q.bundle"))}, tmp_path)
    chat = services.chat(None)  # the flow's plan step names no model
    assert chat.model == "q" and engine.ensured == [["q"]]
