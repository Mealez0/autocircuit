"""Typed transport for the Neuronpedia graph server.

This module deliberately contains no LLM/agent behavior. Agent code calls this
transport and receives structured Python values or explicit exceptions; it never
simulates API calls by printing prose.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import requests

DEFAULT_GRAPH_URL = "http://localhost:5004"
DEFAULT_AUTH_HEADER = "x-secret-key"
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


class NeuronpediaAPIError(RuntimeError):
    """Raised when the graph server cannot complete a request."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class GeneratedGraph:
    """Metadata for a graph response written to disk."""

    path: Path
    manifest_path: Path
    byte_count: int
    sha256: str
    content_type: str


class NeuronpediaGraphClient:
    """Small, synchronous client for Neuronpedia's graph server API.

    The current upstream graph-server contract exposes ``POST /generate-graph``
    and ``POST /steer`` and authenticates with an ``x-secret-key`` header. The
    auth header is configurable because hosted/proxied deployments sometimes use
    a different gateway header.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        secret: str | None = None,
        auth_header: str = DEFAULT_AUTH_HEADER,
        timeout_seconds: float = 120.0,
        retries: int = 0,
        session: requests.Session | None = None,
    ) -> None:
        resolved_url = (base_url or os.getenv("NEURONPEDIA_GRAPH_URL") or DEFAULT_GRAPH_URL).strip()
        if not resolved_url:
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if not auth_header.strip():
            raise ValueError("auth_header must not be empty")

        self.base_url = resolved_url.rstrip("/")
        self.secret = secret if secret is not None else os.getenv("NEURONPEDIA_GRAPH_SECRET")
        self.auth_header = auth_header.strip()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AutoCircuit-Agent/0.2",
        }
        if self.secret:
            headers[self.auth_header] = self.secret
        return headers

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @staticmethod
    def _error_body(response: requests.Response) -> str:
        text = response.text.strip()
        return text[:2000]

    @staticmethod
    def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def _request(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        *,
        stream: bool = False,
    ) -> requests.Response:
        attempts = self.retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self.session.post(
                    self._url(endpoint),
                    headers=self._headers(),
                    json=dict(payload),
                    timeout=self.timeout_seconds,
                    stream=stream,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(min(2**attempt, 8))
                continue

            if response.status_code < 400:
                return response

            body = self._error_body(response)
            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt + 1 >= attempts:
                response.close()
                raise NeuronpediaAPIError(
                    f"Neuronpedia request failed with HTTP {response.status_code}",
                    status_code=response.status_code,
                    body=body,
                )

            response.close()
            time.sleep(min(2**attempt, 8))

        raise NeuronpediaAPIError(f"Neuronpedia request failed: {last_error}") from last_error

    def generate_graph(
        self,
        *,
        prompt: str,
        model_id: str,
        output_path: str | Path,
        batch_size: int = 48,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.85,
        max_feature_nodes: int = 5000,
        slug_identifier: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> GeneratedGraph:
        """Generate an attribution graph and atomically persist the JSON response."""
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not model_id:
            raise ValueError("model_id must not be empty")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "model_id": model_id,
            "batch_size": batch_size,
            "max_n_logits": max_n_logits,
            "desired_logit_prob": desired_logit_prob,
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "max_feature_nodes": max_feature_nodes,
        }
        if slug_identifier:
            payload["slug_identifier"] = slug_identifier
        if extra:
            payload.update(extra)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self._request("/generate-graph", payload, stream=True)
        content_type = response.headers.get("Content-Type", "")

        fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        byte_count = 0
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            if byte_count == 0:
                raise NeuronpediaAPIError("Neuronpedia returned an empty graph response")

            # Fail early on the common case where a proxy returns text/HTML with
            # a 2xx status. Leading JSON whitespace is permitted.
            with open(tmp_name, "rb") as handle:
                first_non_whitespace = handle.read(4096).lstrip()[:1]
            if first_non_whitespace not in {b"{", b"["}:
                raise NeuronpediaAPIError(
                    "Neuronpedia graph response was not JSON",
                    body=f"content-type={content_type!r}",
                )

            os.replace(tmp_name, destination)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        finally:
            response.close()

        sha256 = digest.hexdigest()
        manifest_path = destination.with_name(f"{destination.name}.manifest.json")
        self._atomic_json_write(
            manifest_path,
            {
                "schema_version": "autocircuit-neuronpedia-graph-v1",
                "created_at": datetime.now(UTC).isoformat(),
                "request": {
                    "endpoint": self._url("/generate-graph"),
                    "payload": payload,
                },
                "response": {
                    "path": str(destination),
                    "byte_count": byte_count,
                    "sha256": sha256,
                    "content_type": content_type,
                },
            },
        )

        return GeneratedGraph(
            path=destination,
            manifest_path=manifest_path,
            byte_count=byte_count,
            sha256=sha256,
            content_type=content_type,
        )

    def steer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a graph-server steering intervention and return structured JSON."""
        response = self._request("/steer", payload)
        try:
            data = response.json()
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as exc:
            raise NeuronpediaAPIError(
                "Neuronpedia steering response was not valid JSON",
                body=self._error_body(response),
            ) from exc
        finally:
            response.close()

        if not isinstance(data, dict):
            raise NeuronpediaAPIError("Neuronpedia steering response must be a JSON object")
        return data
