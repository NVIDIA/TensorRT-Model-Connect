# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in catalogs that resolve toolchain intent into immutable artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path

from .builtin_providers import (
    ManagedArtifactToolchainSource,
    target_python_baseline,
    target_runtime_baseline,
)
from .models import DevToolkitError
from .resolution import (
    ArtifactPin,
    ContextLock,
    EnvironmentRequest,
    ProviderDescriptor,
    ToolchainCandidate,
)
from .runner import Runner


Fetch = Callable[[str], bytes | None]

_CUDA_COMPONENTS = (
    "cuda_crt",
    "cuda_cudart",
    "cuda_culibos",
    "cuda_nvcc",
    "libcublas",
    "libcurand",
    "libnvvm",
)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = next((value for name, value in attrs if name.lower() == "href"), None)
        if href:
            self.hrefs.append(href)


def _public_fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise DevToolkitError(f"NVIDIA catalog request failed with HTTP {error.code}") from error
    except (OSError, urllib.error.URLError) as error:
        raise DevToolkitError(f"NVIDIA catalog request failed: {type(error).__name__}") from error


def _artifact(name: str, url: str, digest: str) -> ArtifactPin | None:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return None
    return ArtifactPin(name, url, digest)


class NvidiaPackageIndexCatalog:
    """Resolve public NVIDIA TensorRT wheels and development headers."""

    descriptor = ProviderDescriptor(
        "nvidia-package-index",
        "trtmc-devtoolkit-nvidia-package-index==2",
        1,
    )

    def __init__(
        self,
        *,
        pypi_json: str = "https://pypi.org/pypi",
        nvidia_python: str = "https://pypi.nvidia.com",
        cuda_repository: str = "https://developer.download.nvidia.com/compute/cuda/repos",
        cuda_redist: str = "https://developer.download.nvidia.com/compute/cuda/redist",
        fetch: Fetch | None = None,
    ) -> None:
        self.pypi_json = pypi_json.rstrip("/")
        self.nvidia_python = nvidia_python.rstrip("/")
        self.cuda_repository = cuda_repository.rstrip("/")
        self.cuda_redist = cuda_redist.rstrip("/")
        self._fetch = fetch or _public_fetch

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        selected_catalog = request.toolchain_options.get("catalog")
        if (
            request.artifacts
            or request.toolchain not in {None, "managed-artifacts"}
            or selected_catalog not in {None, self.descriptor.name}
        ):
            return ()
        baseline = target_runtime_baseline(context, repository, runner)
        python = baseline or target_python_baseline(context, repository, runner)
        if python is None or python.python != request.python:
            return ()
        policy = request.cuda
        use_existing = baseline is not None and (
            policy.kind == "system-first"
            or (policy.kind == "system-only" and policy.version in {None, baseline.cuda})
            or (policy.kind == "exact" and policy.version == baseline.cuda)
        )
        if policy.kind == "system-only" and not use_existing:
            return ()
        if use_existing:
            assert baseline is not None
            cuda = baseline.cuda
            cuda_source = baseline.cuda_source
            cuda_root: str | None = baseline.cuda_root
            nvcc: str | None = baseline.nvcc
            cuda_artifacts: tuple[ArtifactPin, ...] = ()
            cuda_release: str | None = None
        else:
            cuda = policy.fallback if policy.kind == "system-first" else policy.version
            if cuda is None:
                return ()
            cuda_source = "managed"
            cuda_root = None
            nvcc = None
            cuda_artifacts, cuda_release = self._cuda_distribution(
                cuda,
                context.architecture,
            )
            if not cuda_artifacts:
                return ()
        artifacts, headers_version = self._distribution(
            request.tensorrt,
            cuda,
            request.python,
            context.architecture,
            str(context.identity.get("os_id", "unknown")),
            str(context.identity.get("os_version", "unknown")),
        )
        if not artifacts:
            return ()
        cuda_major = cuda.split(".", 1)[0]
        managed_names = tuple(artifact.name for artifact in cuda_artifacts)
        return (
            ToolchainCandidate(
                provider=ManagedArtifactToolchainSource.descriptor,
                origin="managed",
                cuda_source=cuda_source,
                tensorrt=request.tensorrt,
                cuda=cuda,
                python=request.python,
                identity={
                    "layout_schema": 3,
                    "catalog": {
                        "name": self.descriptor.name,
                        "implementation": self.descriptor.implementation,
                        "lock_schema": self.descriptor.lock_schema,
                    },
                    "cuda_module": f"nvidia.cu{cuda_major}",
                    "tensorrt_lib_distribution": f"tensorrt_cu{cuda_major}_libs",
                    "system_cuda_root": cuda_root,
                    "system_nvcc": nvcc,
                    "cuda_artifacts": managed_names,
                    "cuda_release": cuda_release,
                    "headers_package_version": headers_version,
                },
                artifacts=(*artifacts, *cuda_artifacts),
            ),
        )

    def _cuda_distribution(
        self,
        version: str,
        architecture: str,
    ) -> tuple[tuple[ArtifactPin, ...], str]:
        """Resolve the minimal complete native-build CUDA component closure."""

        platform_name = {
            "x86_64": "linux-x86_64",
            "aarch64": "linux-sbsa",
        }.get(architecture)
        if platform_name is None:
            return (), ""
        release = f"{version}.0"
        raw = self._fetch(f"{self.cuda_redist}/redistrib_{release}.json")
        if raw is None:
            return (), ""
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return (), ""
            if payload.get("release_label") != release:
                return (), ""
            component_names = (*_CUDA_COMPONENTS, self._cccl_component(payload))
            artifacts: list[ArtifactPin] = []
            for component_name in component_names:
                component = payload[component_name]
                platform = component[platform_name]
                artifact = _artifact(
                    f"cuda-component-{component_name}",
                    urllib.parse.urljoin(
                        f"{self.cuda_redist}/",
                        str(platform["relative_path"]),
                    ),
                    str(platform["sha256"]),
                )
                if artifact is None:
                    return (), ""
                artifacts.append(artifact)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return (), ""
        return tuple(artifacts), release

    @staticmethod
    def _cccl_component(payload: object) -> str:
        if not isinstance(payload, dict):
            raise TypeError
        if "cccl" in payload:
            return "cccl"
        if "cuda_cccl" in payload:
            return "cuda_cccl"
        raise KeyError("cccl")

    def _distribution(
        self,
        version: str,
        cuda: str,
        python: str,
        architecture: str,
        os_id: str,
        os_version: str,
    ) -> tuple[tuple[ArtifactPin, ...], str]:
        cuda_major = cuda.split(".", 1)[0]
        package = f"tensorrt-cu{cuda_major}"
        bindings_package = f"tensorrt-cu{cuda_major}-bindings"
        libraries_package = f"tensorrt-cu{cuda_major}-libs"
        python_source = self._pypi_file(
            package,
            version,
            lambda filename: filename.endswith(".tar.gz"),
            "tensorrt-python",
        )
        python_tag = python.replace(".", "")
        platform_tag = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(architecture)
        if platform_tag is None:
            return (), ""
        bindings = self._pypi_file(
            bindings_package,
            version,
            lambda filename: bool(
                re.search(
                    rf"-cp{re.escape(python_tag)}-none-manylinux_[^-]+_{platform_tag}\.whl$",
                    filename,
                )
            ),
            "tensorrt-bindings",
        )
        libraries = self._nvidia_wheel(
            libraries_package,
            version,
            platform_tag,
            "tensorrt-libs",
        )
        headers, headers_version = self._headers(
            version,
            cuda,
            os_id,
            os_version,
            architecture,
        )
        resolved = (python_source, bindings, libraries, headers)
        if any(item is None for item in resolved):
            return (), ""
        return tuple(item for item in resolved if item is not None), headers_version

    def _pypi_file(
        self,
        package: str,
        version: str,
        matches: Callable[[str], bool],
        name: str,
    ) -> ArtifactPin | None:
        raw = self._fetch(f"{self.pypi_json}/{package}/{version}/json")
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            files = [item for item in payload["urls"] if matches(str(item["filename"]))]
            if len(files) != 1:
                return None
            item = files[0]
            return _artifact(name, str(item["url"]), str(item["digests"]["sha256"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _nvidia_wheel(
        self,
        package: str,
        version: str,
        platform_tag: str,
        name: str,
    ) -> ArtifactPin | None:
        base = f"{self.nvidia_python}/{package}/"
        raw = self._fetch(base)
        if raw is None:
            return None
        parser = _Links()
        parser.feed(raw.decode("utf-8", errors="replace"))
        normalized = package.replace("-", "_")
        pattern = re.compile(
            rf"^{re.escape(normalized)}-{re.escape(version)}-py(?:2\.py3|3)-none-"
            rf"manylinux_[^-]+_{platform_tag}\.whl$"
        )
        matches: list[tuple[str, str]] = []
        for href in parser.hrefs:
            resolved = urllib.parse.urljoin(base, href)
            parsed = urllib.parse.urlsplit(resolved)
            filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            digest = urllib.parse.parse_qs(parsed.fragment).get("sha256", [])
            if pattern.fullmatch(filename) and len(digest) == 1:
                matches.append((urllib.parse.urldefrag(resolved).url, digest[0]))
        if len(matches) != 1:
            return None
        return _artifact(name, *matches[0])

    def _headers(
        self,
        version: str,
        cuda: str,
        os_id: str,
        os_version: str,
        architecture: str,
    ) -> tuple[ArtifactPin | None, str]:
        distribution = self._distribution_name(os_id, os_version)
        architecture_names = {
            "x86_64": ("x86_64", "amd64"),
            "aarch64": ("sbsa", "arm64"),
        }.get(architecture)
        if distribution is None or architecture_names is None:
            return None, ""
        repository_arch, deb_arch = architecture_names
        base = f"{self.cuda_repository}/{distribution}/{repository_arch}/"
        raw = self._fetch(f"{base}Packages.gz")
        if raw is None:
            return None, ""
        try:
            packages = gzip.decompress(raw).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None, ""
        cuda_major = cuda.split(".", 1)[0]
        matches: list[dict[str, str]] = []
        for stanza in packages.split("\n\n"):
            fields = {
                key: value
                for line in stanza.splitlines()
                if ": " in line
                for key, value in (line.split(": ", 1),)
            }
            package_version = fields.get("Version", "")
            if (
                fields.get("Package") == "libnvinfer-headers-dev"
                and fields.get("Architecture") in {deb_arch, "all"}
                and package_version.startswith(f"{version}-1+cuda{cuda_major}.")
                and fields.get("Filename")
                and fields.get("SHA256")
            ):
                matches.append(fields)
        if not matches:
            return None, ""
        matches.sort(
            key=lambda item: (
                f"+cuda{cuda}" in item["Version"],
                item["Version"],
            ),
            reverse=True,
        )
        selected = matches[0]
        url = urllib.parse.urljoin(base, selected["Filename"])
        return _artifact("tensorrt-headers", url, selected["SHA256"]), selected["Version"]

    @staticmethod
    def _distribution_name(os_id: str, os_version: str) -> str | None:
        compact = os_version.replace(".", "")
        if os_id == "ubuntu" and compact in {"2004", "2204", "2404"}:
            return f"ubuntu{compact}"
        if os_id == "debian" and compact in {"11", "12", "13"}:
            return f"debian{compact}"
        return None


class JsonToolchainCatalog:
    """Resolve team/private artifacts from explicit, digest-pinned manifests."""

    descriptor = ProviderDescriptor(
        "json-toolchain-catalog",
        "trtmc-devtoolkit-json-toolchain-catalog==1",
        1,
    )

    def __init__(self, manifests: Sequence[Path]) -> None:
        self.manifests = tuple(Path(path).resolve() for path in manifests)
        if not self.manifests:
            raise DevToolkitError("JSON toolchain catalog requires at least one manifest")

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]:
        selected_catalog = request.toolchain_options.get("catalog")
        if (
            request.artifacts
            or request.toolchain not in {None, "managed-artifacts"}
            or selected_catalog not in {None, self.descriptor.name}
        ):
            return ()
        baseline = target_runtime_baseline(context, repository, runner)
        python = baseline or target_python_baseline(context, repository, runner)
        if python is None or python.python != request.python:
            return ()
        candidates: list[ToolchainCandidate] = []
        record_ids: set[str] = set()
        for manifest in self.manifests:
            raw = self._read(manifest)
            digest = hashlib.sha256(raw).hexdigest()
            payload = self._payload(manifest, raw)
            for record in payload["toolchains"]:
                if not isinstance(record, Mapping):
                    raise DevToolkitError(f"Toolchain catalog {manifest} has a non-object record")
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise DevToolkitError(f"Toolchain catalog {manifest} has a record without id")
                if record_id in record_ids:
                    raise DevToolkitError(f"Duplicate toolchain catalog record id: {record_id}")
                record_ids.add(record_id)
                candidate = self._candidate(
                    request,
                    context,
                    baseline,
                    manifest,
                    digest,
                    record_id,
                    record,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def _read(manifest: Path) -> bytes:
        try:
            return manifest.read_bytes()
        except OSError as error:
            raise DevToolkitError(
                f"Could not read toolchain catalog {manifest}: {error}"
            ) from error

    @staticmethod
    def _payload(manifest: Path, raw: bytes) -> Mapping[str, object]:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DevToolkitError(f"Invalid toolchain catalog JSON {manifest}: {error}") from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("toolchains"), list)
        ):
            raise DevToolkitError(
                f"Toolchain catalog {manifest} requires schema_version 1 and toolchains[]"
            )
        return payload

    def _candidate(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        baseline,
        manifest: Path,
        manifest_digest: str,
        record_id: str,
        record: Mapping[str, object],
    ) -> ToolchainCandidate | None:
        if (
            record.get("tensorrt") != request.tensorrt
            or record.get("python") != request.python
            or record.get("architecture") != context.architecture
            or not self._matches_optional(record.get("os_id"), context.identity.get("os_id"))
            or not self._matches_optional(
                record.get("os_version"), context.identity.get("os_version")
            )
        ):
            return None
        cuda = record.get("cuda")
        if not isinstance(cuda, Mapping):
            raise DevToolkitError(f"Toolchain catalog record {record_id} requires cuda object")
        source = cuda.get("source")
        if source == "target":
            if baseline is None or not self._target_cuda_matches(cuda, baseline.cuda):
                return None
            if request.cuda.kind == "managed":
                return None
            if request.cuda.kind in {"exact", "system-only"} and request.cuda.version not in {
                None,
                baseline.cuda,
            }:
                return None
            resolved_cuda = baseline.cuda
            cuda_source = baseline.cuda_source
            cuda_root: str | None = baseline.cuda_root
            nvcc: str | None = baseline.nvcc
            cuda_artifact_names: tuple[str, ...] = ()
            cuda_release: str | None = None
        elif source == "managed":
            resolved_cuda = cuda.get("version")
            if not isinstance(resolved_cuda, str):
                raise DevToolkitError(
                    f"Toolchain catalog record {record_id} requires managed CUDA version"
                )
            if request.cuda.kind == "system-only":
                return None
            requested_cuda = (
                request.cuda.fallback
                if request.cuda.kind == "system-first"
                else request.cuda.version
            )
            if requested_cuda != resolved_cuda:
                return None
            cuda_source = "managed"
            cuda_root = None
            nvcc = None
            components = cuda.get("artifacts")
            if (
                not isinstance(components, list)
                or not components
                or any(not isinstance(name, str) or not name for name in components)
            ):
                raise DevToolkitError(
                    f"Toolchain catalog record {record_id} requires managed CUDA artifacts[]"
                )
            cuda_artifact_names = tuple(components)
            release = cuda.get("release")
            cuda_release = str(release) if release is not None else None
        else:
            raise DevToolkitError(
                f"Toolchain catalog record {record_id} CUDA source must be target or managed"
            )
        artifacts = self._artifacts(manifest, record_id, record.get("artifacts"))
        artifact_names = {artifact.name for artifact in artifacts}
        if any(name not in artifact_names for name in cuda_artifact_names):
            raise DevToolkitError(
                f"Toolchain catalog record {record_id} references an unknown CUDA artifact"
            )
        cuda_major = resolved_cuda.split(".", 1)[0]
        distribution = record.get(
            "tensorrt_lib_distribution",
            f"tensorrt_cu{cuda_major}_libs",
        )
        if not isinstance(distribution, str) or not distribution:
            raise DevToolkitError(
                f"Toolchain catalog record {record_id} has invalid TensorRT distribution"
            )
        return ToolchainCandidate(
            provider=ManagedArtifactToolchainSource.descriptor,
            origin="managed",
            cuda_source=cuda_source,
            tensorrt=request.tensorrt,
            cuda=resolved_cuda,
            python=request.python,
            identity={
                "layout_schema": 3,
                "catalog": {
                    "name": self.descriptor.name,
                    "implementation": self.descriptor.implementation,
                    "lock_schema": self.descriptor.lock_schema,
                    "manifest_sha256": manifest_digest,
                    "record_id": record_id,
                },
                "cuda_module": f"nvidia.cu{cuda_major}",
                "tensorrt_lib_distribution": distribution,
                "system_cuda_root": cuda_root,
                "system_nvcc": nvcc,
                "cuda_artifacts": cuda_artifact_names,
                "cuda_release": cuda_release,
            },
            artifacts=artifacts,
        )

    @staticmethod
    def _matches_optional(expected: object, actual: object) -> bool:
        if expected is None:
            return True
        if isinstance(expected, str):
            return expected == actual
        if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
            return actual in expected
        return False

    @staticmethod
    def _target_cuda_matches(cuda: Mapping[str, object], actual: str) -> bool:
        version = cuda.get("version")
        major = cuda.get("major")
        if version is not None and version != actual:
            return False
        if major is not None and major != actual.split(".", 1)[0]:
            return False
        return version is not None or major is not None

    @staticmethod
    def _artifacts(
        manifest: Path,
        record_id: str,
        raw_artifacts: object,
    ) -> tuple[ArtifactPin, ...]:
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise DevToolkitError(f"Toolchain catalog record {record_id} requires artifacts[]")
        artifacts: list[ArtifactPin] = []
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping):
                raise DevToolkitError(
                    f"Toolchain catalog record {record_id} has a non-object artifact"
                )
            name, uri, digest = raw.get("name"), raw.get("uri"), raw.get("sha256")
            if not all(isinstance(value, str) for value in (name, uri, digest)):
                raise DevToolkitError(
                    f"Toolchain catalog record {record_id} has an invalid artifact"
                )
            parsed = urllib.parse.urlsplit(uri)
            if not parsed.scheme:
                path = Path(uri)
                if not path.is_absolute():
                    path = manifest.parent / path
                uri = path.resolve().as_uri()
            artifacts.append(ArtifactPin(name, uri, digest))
        names = [artifact.name for artifact in artifacts]
        if len(names) != len(set(names)):
            raise DevToolkitError(
                f"Toolchain catalog record {record_id} has duplicate artifact names"
            )
        if "tensorrt-headers" not in names or not any(
            urllib.parse.urlsplit(artifact.uri).path.endswith((".whl", ".tar.gz"))
            for artifact in artifacts
        ):
            raise DevToolkitError(
                f"Toolchain catalog record {record_id} lacks TensorRT headers or Python packages"
            )
        return tuple(artifacts)
