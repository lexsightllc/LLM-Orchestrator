"""Artifact management system for LLM Orchestrator."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import orjson
from pydantic import BaseModel, Field, validator

class ArtifactType(str, Enum):
    """Supported artifact types."""
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    IMAGE = "image"
    DATAFRAME = "dataframe"
    FILE = "file"

class ArtifactState(str, Enum):
    """State of an artifact in its lifecycle."""
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"

class ArtifactMetadata(BaseModel):
    """Metadata for an artifact."""
    name: str
    artifact_type: ArtifactType
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())
    state: ArtifactState = ArtifactState.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_ref: Optional[str] = None  # JSON Schema reference for validation
    
    class Config:
        json_loads = orjson.loads
        json_dumps = lambda x, **kw: orjson.dumps(x, **kw).decode()

class ArtifactRef(BaseModel):
    """Reference to an artifact in storage."""
    artifact_id: str  # Unique identifier (content-addressable hash)
    path: str  # Relative path in storage
    metadata: ArtifactMetadata
    
    @property
    def uri(self) -> str:
        """Get the URI for this artifact."""
        return f"artifact://{self.artifact_id}"

class Artifact:
    """Base class for all artifacts."""
    
    def __init__(
        self,
        data: Any,
        name: str,
        artifact_type: ArtifactType,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        schema_ref: Optional[str] = None,
    ):
        self.data = data
        self.metadata = ArtifactMetadata(
            name=name,
            artifact_type=artifact_type,
            content_type=content_type or self._get_default_content_type(artifact_type),
            metadata=metadata or {},
            schema_ref=schema_ref,
        )
        self._update_size()
    
    def _update_size(self) -> None:
        """Update the size metadata based on current data."""
        if isinstance(self.data, (str, bytes)):
            self.metadata.size_bytes = len(self.data)
        elif isinstance(self.data, (dict, list)):
            self.metadata.size_bytes = len(orjson.dumps(self.data))
        else:
            self.metadata.size_bytes = len(str(self.data).encode('utf-8'))
    
    @staticmethod
    def _get_default_content_type(artifact_type: ArtifactType) -> str:
        """Get default content type for an artifact type."""
        return {
            ArtifactType.JSON: "application/json",
            ArtifactType.TEXT: "text/plain",
            ArtifactType.BINARY: "application/octet-stream",
            ArtifactType.IMAGE: "image/png",
            ArtifactType.DATAFRAME: "application/vnd.apache.arrow.file",
            ArtifactType.FILE: "application/octet-stream",
        }[artifact_type]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Artifact':
        """Create an artifact from a dictionary."""
        return cls(
            data=data.get('data'),
            name=data.get('name', 'unnamed'),
            artifact_type=ArtifactType(data.get('artifact_type')),
            content_type=data.get('content_type'),
            metadata=data.get('metadata', {}),
            schema_ref=data.get('schema_ref')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to a dictionary."""
        return {
            'data': self.data,
            'name': self.metadata.name,
            'artifact_type': self.metadata.artifact_type.value,
            'content_type': self.metadata.content_type,
            'metadata': self.metadata.metadata,
            'schema_ref': self.metadata.schema_ref
        }
    
    def validate(self, schema: Optional[Dict] = None) -> bool:
        """Validate the artifact against a JSON Schema."""
        # If no schema provided, try to load from schema_ref
        if schema is None and self.metadata.schema_ref:
            # Schema references are not yet supported – log and skip validation
            logger.warning(
                "Schema loading from reference '%s' not implemented; skipping validation",
                self.metadata.schema_ref,
            )
            return True
        
        if schema is None:
            return True  # No schema to validate against
            
        # Use jsonschema for validation
        try:
            import jsonschema
            jsonschema.validate(instance=self.data, schema=schema)
            return True
        except ImportError:
            logger.warning("jsonschema not installed, validation skipped")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

class ArtifactStorage:
    """Content-addressable storage for artifacts."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize with a base storage path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create required directories
        (self.base_path / "blobs").mkdir(exist_ok=True)
        (self.base_path / "refs").mkdir(exist_ok=True)
    
    def _get_blob_path(self, content_hash: str) -> Path:
        """Get the path for a blob based on its content hash."""
        # Use first 2 chars as directory for better filesystem performance
        dir_name = content_hash[:2]
        dir_path = self.base_path / "blobs" / dir_name
        dir_path.mkdir(exist_ok=True)
        return dir_path / content_hash
    
    def _get_ref_path(self, ref_name: str) -> Path:
        """Get the path for a named reference."""
        return self.base_path / "refs" / f"{ref_name}.json"
    
    def _calculate_hash(self, data: bytes) -> str:
        """Calculate the content hash for data."""
        return hashlib.sha256(data).hexdigest()
    
    def store(self, artifact: Artifact, ref_name: Optional[str] = None) -> ArtifactRef:
        """Store an artifact and return its reference."""
        # Serialize the artifact data
        if isinstance(artifact.data, (str, bytes)):
            data = artifact.data if isinstance(artifact.data, bytes) else artifact.data.encode('utf-8')
        else:
            data = orjson.dumps(artifact.data)
        
        # Calculate content hash
        content_hash = self._calculate_hash(data)
        blob_path = self._get_blob_path(content_hash)
        
        # Write the data if it doesn't exist
        if not blob_path.exists():
            with open(blob_path, 'wb') as f:
                f.write(data)
        
        # Create artifact reference
        artifact_ref = ArtifactRef(
            artifact_id=content_hash,
            path=str(blob_path.relative_to(self.base_path)),
            metadata=artifact.metadata
        )
        
        # Update metadata with final size
        artifact.metadata.size_bytes = blob_path.stat().st_size
        artifact.metadata.state = ArtifactState.READY
        artifact.metadata.updated_at = time.time()
        
        # Store the reference if a name is provided
        if ref_name:
            ref_path = self._get_ref_path(ref_name)
            with open(ref_path, 'wb') as f:
                f.write(artifact_ref.json().encode('utf-8'))
        
        return artifact_ref
    
    def load(self, ref: Union[str, ArtifactRef]) -> Artifact:
        """Load an artifact by its reference."""
        if isinstance(ref, str):
            # It's a reference name or artifact ID
            if '/' in ref or '\\' in ref:
                # It's a path
                ref_path = self.base_path / ref
            elif len(ref) == 64:  # SHA-256 hash length
                # It's an artifact ID
                ref_path = self._get_blob_path(ref)
            else:
                # It's a named reference
                ref_path = self._get_ref_path(ref)
                with open(ref_path, 'rb') as f:
                    ref = ArtifactRef.parse_raw(f.read())
                ref_path = self.base_path / ref.path
        else:
            # It's already an ArtifactRef
            ref_path = self.base_path / ref.path
        
        # Load the artifact data
        with open(ref_path, 'rb') as f:
            data = f.read()
        
        # Parse the data based on content type
        content_type = ref.metadata.content_type if isinstance(ref, ArtifactRef) else None
        
        if content_type == 'application/json' or (isinstance(ref, ArtifactRef) and ref.metadata.artifact_type == ArtifactType.JSON):
            data = orjson.loads(data)
        elif content_type and content_type.startswith('text/'):
            data = data.decode('utf-8')
        
        # Create and return the artifact
        if isinstance(ref, ArtifactRef):
            return Artifact(
                data=data,
                name=ref.metadata.name,
                artifact_type=ref.metadata.artifact_type,
                content_type=ref.metadata.content_type,
                metadata=ref.metadata.metadata,
                schema_ref=ref.metadata.schema_ref
            )
        else:
            # For raw artifact IDs, create a basic artifact
            return Artifact(
                data=data,
                name=Path(ref_path).name,
                artifact_type=ArtifactType.FILE,
                content_type=content_type or 'application/octet-stream'
            )
    
    def delete(self, ref: Union[str, ArtifactRef]) -> None:
        """Delete an artifact by its reference."""
        if isinstance(ref, str):
            if '/' in ref or '\\' in ref:
                # It's a path
                path = self.base_path / ref
                if path.exists():
                    path.unlink()
            elif len(ref) == 64:  # SHA-256 hash length
                # It's an artifact ID
                path = self._get_blob_path(ref)
                if path.exists():
                    path.unlink()
            else:
                # It's a named reference
                ref_path = self._get_ref_path(ref)
                if ref_path.exists():
                    ref_path.unlink()
        else:
            # It's an ArtifactRef
            path = self.base_path / ref.path
            if path.exists():
                path.unlink()
            
            # Also remove any named references
            for ref_file in (self.base_path / "refs").glob("*.json"):
                with open(ref_file, 'rb') as f:
                    try:
                        artifact_ref = ArtifactRef.parse_raw(f.read())
                        if artifact_ref.artifact_id == ref.artifact_id:
                            ref_file.unlink()
                    except Exception:
                        continue

# Global storage instance
_default_storage: Optional[ArtifactStorage] = None

def get_default_storage() -> ArtifactStorage:
    """Get or create the default artifact storage."""
    global _default_storage
    if _default_storage is None:
        # Default to a local directory
        storage_dir = Path.home() / ".llm_orchestrator" / "artifacts"
        _default_storage = ArtifactStorage(storage_dir)
    return _default_storage

def set_default_storage(storage: ArtifactStorage) -> None:
    """Set the default artifact storage."""
    global _default_storage
    _default_storage = storage

# Factory functions for common artifact types
def create_json_artifact(
    data: Any,
    name: str = "data.json",
    schema_ref: Optional[str] = None,
    **metadata
) -> Artifact:
    """Create a JSON artifact."""
    return Artifact(
        data=data,
        name=name,
        artifact_type=ArtifactType.JSON,
        content_type="application/json",
        metadata=metadata,
        schema_ref=schema_ref
    )

def create_text_artifact(
    text: str,
    name: str = "text.txt",
    **metadata
) -> Artifact:
    """Create a text artifact."""
    return Artifact(
        data=text,
        name=name,
        artifact_type=ArtifactType.TEXT,
        content_type="text/plain",
        metadata=metadata
    )

# Add logging
import logging
import time

logger = logging.getLogger(__name__)
