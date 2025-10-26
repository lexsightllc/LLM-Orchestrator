# SPDX-License-Identifier: MPL-2.0
"""Content-addressable artifact storage utilities.

This module provides a minimal yet fully-typed artifact management subsystem
used by the sandbox and REPL.  It deliberately focuses on deterministic
serialization so artifacts can be hashed and reproduced across platforms.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:  # pragma: no cover - optional dependency
    import jsonschema
except ImportError:  # pragma: no cover - fallback when jsonschema is absent
    jsonschema = None

__all__ = [
    "Artifact",
    "ArtifactMetadata",
    "ArtifactRef",
    "ArtifactStorage",
    "ArtifactType",
    "create_json_artifact",
    "create_text_artifact",
    "create_binary_artifact",
    "get_default_storage",
    "set_default_storage",
]


class ArtifactType(str, Enum):
    """Supported artifact payload encodings."""

    JSON = "json"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class ArtifactMetadata:
    """Metadata captured for each stored artifact."""

    name: str
    artifact_type: ArtifactType
    content_type: str = "application/octet-stream"
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a JSON-serializable dictionary."""

        return {
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArtifactMetadata":
        """Deserialize :class:`ArtifactMetadata` from a dictionary."""

        return cls(
            name=payload["name"],
            artifact_type=ArtifactType(payload["artifact_type"]),
            content_type=payload["content_type"],
            created_at=payload.get("created_at", time.time()),
            updated_at=payload.get("updated_at", time.time()),
            size_bytes=payload.get("size_bytes", 0),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class Artifact:
    """In-memory representation of an artifact payload."""

    data: Any
    name: str
    artifact_type: ArtifactType
    content_type: Optional[str] = None
    metadata: Union[ArtifactMetadata, Dict[str, Any], None] = None

    def __post_init__(self) -> None:
        default_content_type = {
            ArtifactType.JSON: "application/json",
            ArtifactType.TEXT: "text/plain",
            ArtifactType.BINARY: "application/octet-stream",
        }[self.artifact_type]

        if isinstance(self.metadata, ArtifactMetadata):
            meta = self.metadata
            extra = meta.metadata
        else:
            extra = dict(self.metadata or {})
            meta = ArtifactMetadata(
                name=self.name,
                artifact_type=self.artifact_type,
                content_type=self.content_type or default_content_type,
                metadata=extra,
            )

        meta.name = self.name
        meta.artifact_type = self.artifact_type
        meta.content_type = self.content_type or meta.content_type or default_content_type

        payload_size = len(self._serialize_payload())
        meta.size_bytes = payload_size
        now = time.time()
        if meta.created_at == 0:
            meta.created_at = now
        meta.updated_at = now
        meta.metadata = extra

        self.content_type = meta.content_type
        self.metadata = meta

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the artifact for persistence."""

        return {
            "data": self._encode_for_storage(),
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "content_type": self.content_type,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Artifact":
        """Restore an :class:`Artifact` from serialized form."""

        artifact_type = ArtifactType(payload["artifact_type"])
        data = cls._decode_from_storage(artifact_type, payload["data"])
        metadata = ArtifactMetadata.from_dict(payload.get("metadata", {}))
        return cls(
            data=data,
            name=payload["name"],
            artifact_type=artifact_type,
            content_type=payload.get("content_type"),
            metadata=metadata,
        )

    def validate(self, schema: Dict[str, Any]) -> bool:
        """Validate the artifact against a JSON schema if available."""

        if jsonschema is None:
            return True
        try:
            jsonschema.validate(self.data, schema)
        except jsonschema.ValidationError:
            return False
        return True

    def _serialize_payload(self) -> bytes:
        """Serialize the payload to bytes for hashing."""

        if self.artifact_type is ArtifactType.JSON:
            return json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if self.artifact_type is ArtifactType.TEXT:
            return str(self.data).encode("utf-8")
        if isinstance(self.data, bytes):
            return self.data
        raise TypeError("Binary artifacts require bytes data")

    def _encode_for_storage(self) -> Union[str, bytes]:
        if self.artifact_type is ArtifactType.JSON:
            return json.dumps(self.data, sort_keys=True)
        if self.artifact_type is ArtifactType.TEXT:
            return str(self.data)
        return self.data

    @staticmethod
    def _decode_from_storage(artifact_type: ArtifactType, payload: Union[str, bytes]) -> Any:
        if artifact_type is ArtifactType.JSON:
            return json.loads(payload) if isinstance(payload, str) else json.loads(payload.decode("utf-8"))
        if artifact_type is ArtifactType.TEXT:
            return str(payload)
        return payload if isinstance(payload, bytes) else payload.encode("utf-8")


@dataclass
class ArtifactRef:
    """Reference to an artifact stored on disk."""

    artifact_id: str
    path: str
    metadata: ArtifactMetadata

    @property
    def uri(self) -> str:
        return f"artifact://{self.artifact_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=payload["artifact_id"],
            path=payload["path"],
            metadata=ArtifactMetadata.from_dict(payload["metadata"]),
        )


class ArtifactStorage:
    """Simple content-addressable store for artifacts."""

    def __init__(self, base_path: Union[str, os.PathLike[str]]) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "blobs").mkdir(exist_ok=True)
        (self.base_path / "refs").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def store(self, artifact: Artifact, ref_name: Optional[str] = None) -> ArtifactRef:
        payload = artifact._serialize_payload()
        artifact_id = hashlib.sha256(payload).hexdigest()
        blob_dir = self.base_path / "blobs" / artifact_id[:2]
        blob_dir.mkdir(parents=True, exist_ok=True)
        data_path = blob_dir / f"{artifact_id}.bin"
        meta_path = blob_dir / f"{artifact_id}.json"

        with self._lock:
            if not data_path.exists():
                data_path.write_bytes(payload)
                meta_path.write_text(json.dumps(artifact.to_dict(), sort_keys=True), encoding="utf-8")

            ref = ArtifactRef(
                artifact_id=artifact_id,
                path=str(data_path.relative_to(self.base_path)),
                metadata=artifact.metadata,
            )

            if ref_name:
                ref_path = self.base_path / "refs" / f"{ref_name}.json"
                ref_path.write_text(json.dumps(ref.to_dict(), sort_keys=True), encoding="utf-8")

        return ref

    def load(self, reference: Union[str, ArtifactRef]) -> Artifact:
        if isinstance(reference, ArtifactRef):
            return self._load_by_artifact_id(reference.artifact_id, raw=False)

        ref_path = self.base_path / "refs" / f"{reference}.json"
        if ref_path.exists():
            ref_payload = json.loads(ref_path.read_text(encoding="utf-8"))
            ref = ArtifactRef.from_dict(ref_payload)
            return self._load_by_artifact_id(ref.artifact_id, raw=False)

        return self._load_by_artifact_id(reference, raw=True)

    def delete(self, reference: Union[str, ArtifactRef]) -> None:
        if isinstance(reference, ArtifactRef):
            artifact_id = reference.artifact_id
        else:
            ref_path = self.base_path / "refs" / f"{reference}.json"
            if ref_path.exists():
                payload = json.loads(ref_path.read_text(encoding="utf-8"))
                artifact_id = payload["artifact_id"]
                ref_path.unlink()
            else:
                artifact_id = reference

        for candidate in (self.base_path / "refs").glob("*.json"):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("artifact_id") == artifact_id and candidate.exists():
                candidate.unlink()

        blob_dir = self.base_path / "blobs" / artifact_id[:2]
        data_path = blob_dir / f"{artifact_id}.bin"
        meta_path = blob_dir / f"{artifact_id}.json"

        with self._lock:
            if data_path.exists():
                data_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

    def _load_by_artifact_id(self, artifact_id: str, *, raw: bool) -> Artifact:
        blob_dir = self.base_path / "blobs" / artifact_id[:2]
        data_path = blob_dir / f"{artifact_id}.bin"
        meta_path = blob_dir / f"{artifact_id}.json"
        if not data_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Artifact {artifact_id} not found")

        artifact_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = Artifact.from_dict(artifact_payload)

        if raw:
            artifact_bytes = data_path.read_bytes()
            if artifact.artifact_type is ArtifactType.JSON:
                artifact.data = artifact_bytes
            elif artifact.artifact_type is ArtifactType.TEXT:
                artifact.data = artifact_bytes.decode("utf-8")
            else:
                artifact.data = artifact_bytes
        else:
            artifact.data = Artifact._decode_from_storage(
                artifact.artifact_type,
                artifact_payload["data"],
            )
        return artifact


_default_storage: Optional[ArtifactStorage] = None


def get_default_storage() -> ArtifactStorage:
    """Return the process-wide default artifact storage."""

    global _default_storage
    if _default_storage is None:
        base = Path(os.environ.get("LLM_ORCHESTRATOR_ARTIFACT_DIR", "data/artifacts"))
        _default_storage = ArtifactStorage(base)
    return _default_storage


def set_default_storage(storage: ArtifactStorage) -> None:
    """Override the global default storage."""

    global _default_storage
    _default_storage = storage


def create_json_artifact(data: Any, name: str, **metadata: Any) -> Artifact:
    """Create an artifact with JSON content."""

    return Artifact(data=data, name=name, artifact_type=ArtifactType.JSON, metadata=metadata)


def create_text_artifact(data: str, name: str, **metadata: Any) -> Artifact:
    """Create an artifact with text content."""

    return Artifact(data=data, name=name, artifact_type=ArtifactType.TEXT, metadata=metadata)


def create_binary_artifact(data: bytes, name: str, **metadata: Any) -> Artifact:
    """Create an artifact with binary content."""

    return Artifact(data=data, name=name, artifact_type=ArtifactType.BINARY, metadata=metadata)
