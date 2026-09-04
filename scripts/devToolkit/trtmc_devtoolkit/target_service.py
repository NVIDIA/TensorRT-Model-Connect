# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral execution-target lifecycle orchestration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from .models import DevToolkitError
from .providers import FrozenProviderRegistry
from .receipt import exclusive_lock, write_json
from .runner import Runner
from .target_contracts import ProvisionedTarget, TargetHandle, TargetPlan, _plain


def _policy_label(policy: object | None) -> str:
    if policy is None:
        return "provider-default"
    if isinstance(policy, Enum) and isinstance(policy.value, str) and policy.value:
        return policy.value
    return f"{type(policy).__module__}.{type(policy).__qualname__}"


class TargetService:
    def __init__(
        self,
        repository: Path,
        state_root: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
    ) -> None:
        self.repository = repository
        self.state_root = state_root
        self.providers = providers
        self.runner = runner

    def resolve(self, request: object) -> TargetPlan:
        provider_name = getattr(request, "provider", None)
        if not isinstance(provider_name, str) or not provider_name:
            raise DevToolkitError("Target request must declare a provider")
        provider = self.providers.target(provider_name)
        plan = provider.resolve(
            request,
            repository=self.repository,
            runner=self.runner,
        )
        if not isinstance(plan, TargetPlan) or plan.provider != provider.descriptor:
            raise DevToolkitError("Target provider returned an incompatible target plan")
        return plan

    def provision(
        self,
        plan: TargetPlan,
        *,
        policy: object | None = None,
    ) -> ProvisionedTarget:
        if not isinstance(plan, TargetPlan):
            raise DevToolkitError("Target provisioning requires a TargetPlan")
        state_dir = self.state_root / "targets" / plan.plan_id
        state_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(state_dir / ".provision.lock"):
            write_json(state_dir / "target-plan.json", plan.as_dict())
            (state_dir / "provision-receipt.json").unlink(missing_ok=True)
            (state_dir / "provision-failure.json").unlink(missing_ok=True)
            try:
                provider = self.providers.target(plan.provider.name)
                if provider.descriptor != plan.provider:
                    raise DevToolkitError("Registered target provider does not match target plan")
                handle = provider.provision(
                    plan,
                    policy=policy,
                    repository=self.repository,
                    state_dir=state_dir,
                    runner=self.runner,
                )
                if handle.provider != plan.provider or handle.plan_id != plan.plan_id:
                    raise DevToolkitError("Target provider returned an incompatible target handle")
                provider.attest(
                    handle,
                    repository=self.repository,
                    runner=self.runner,
                )
                receipt = write_json(
                    state_dir / "provision-receipt.json",
                    {
                        "schema_version": 1,
                        "status": "ready",
                        "plan_id": plan.plan_id,
                        "target_id": handle.target_id,
                        "action": handle.action,
                        "policy": handle.policy,
                        "attested": True,
                        "provider": plan.as_dict()["provider"],
                        "identity": _plain(handle.identity),
                        "observation": _plain(handle.observation),
                    },
                )
                return ProvisionedTarget(
                    provider=handle.provider,
                    plan_id=handle.plan_id,
                    target_id=handle.target_id,
                    action=handle.action,
                    policy=handle.policy,
                    identity=handle.identity,
                    observation=handle.observation,
                    execution_target=handle.execution_target,
                    receipt=receipt,
                    request=handle.request,
                )
            except Exception as error:
                write_json(
                    state_dir / "provision-failure.json",
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "plan_id": plan.plan_id,
                        "policy": _policy_label(policy),
                        "error_type": type(error).__name__,
                    },
                )
                raise

    def ensure(
        self,
        request: object,
        *,
        policy: object | None = None,
    ) -> ProvisionedTarget:
        return self.provision(self.resolve(request), policy=policy)

    def attest(self, target: ProvisionedTarget) -> None:
        provider = self.providers.target(target.provider.name)
        if provider.descriptor != target.provider:
            raise DevToolkitError("Registered target provider does not match provisioned target")
        provider.attest(
            TargetHandle(
                provider=target.provider,
                plan_id=target.plan_id,
                target_id=target.target_id,
                action=target.action,
                policy=target.policy,
                identity=target.identity,
                observation=target.observation,
                execution_target=target.execution_target,
                request=target.request,
            ),
            repository=self.repository,
            runner=self.runner,
        )
