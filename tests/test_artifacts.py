"""Tests for the artifact management system."""
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from orchestrator.artifacts import (
    Artifact, ArtifactType, ArtifactStorage, ArtifactRef, ArtifactMetadata,
    create_json_artifact, create_text_artifact, get_default_storage, set_default_storage
)

@pytest.fixture
def temp_storage():
    """Create a temporary storage for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ArtifactStorage(temp_dir)
        yield storage
        # Cleanup is handled by the tempdir context

@pytest.fixture
def sample_artifact():
    """Create a sample artifact for testing."""
    return Artifact(
        data={"key": "value"},
        name="test.json",
        artifact_type=ArtifactType.JSON,
        content_type="application/json",
        metadata={"test": "data"}
    )

def test_artifact_creation():
    """Test creating an artifact."""
    artifact = Artifact(
        data={"test": 123},
        name="test.json",
        artifact_type=ArtifactType.JSON
    )
    
    assert artifact.data == {"test": 123}
    assert artifact.metadata.name == "test.json"
    assert artifact.metadata.artifact_type == ArtifactType.JSON
    assert isinstance(artifact.metadata.created_at, datetime)
    assert isinstance(artifact.metadata.updated_at, datetime)
    assert artifact.metadata.size_bytes > 0
    assert len(artifact.metadata.content_hash) > 0

def test_json_artifact_factory():
    """Test the create_json_artifact factory function."""
    artifact = create_json_artifact(
        {"key": "value"},
        name="test.json",
        test_meta="test"
    )
    
    assert artifact.metadata.artifact_type == ArtifactType.JSON
    assert artifact.metadata.content_type == "application/json"
    assert artifact.data == {"key": "value"}
    assert artifact.metadata.metadata.get("test_meta") == "test"

def test_text_artifact_factory():
    """Test creating a text artifact using the factory function."""
    artifact = create_text_artifact("test text", "test.txt")
    
    assert artifact.metadata.artifact_type == ArtifactType.TEXT
    assert artifact.metadata.content_type == "text/plain"
    assert artifact.data == "test text"

def test_artifact_storage_store_and_load(temp_storage, sample_artifact):
    """Test storing and loading an artifact."""
    # Store the artifact
    artifact_ref = temp_storage.store(sample_artifact, "test_ref")
    
    # Load the artifact
    loaded_artifact = temp_storage.load(artifact_ref)
    
    # Verify the loaded artifact
    assert loaded_artifact.data == {"key": "value"}
    assert loaded_artifact.metadata.name == "test.json"
    assert loaded_artifact.metadata.artifact_type == ArtifactType.JSON

def test_artifact_storage_load_by_ref_name(temp_storage, sample_artifact):
    """Test loading an artifact by reference name."""
    # Store with a named reference
    temp_storage.store(sample_artifact, "test_ref")
    
    # Load by reference name
    loaded_artifact = temp_storage.load("test_ref")
    
    # Verify the loaded artifact
    assert loaded_artifact.data == {"key": "value"}

def test_artifact_storage_load_by_artifact_id(temp_storage, sample_artifact):
    """Test loading an artifact by its content hash."""
    # Store the artifact
    artifact_ref = temp_storage.store(sample_artifact, "test_ref")
    
    # Load by artifact ID (content hash)
    loaded_artifact = temp_storage.load(artifact_ref.artifact_id)
    
    # Verify the loaded artifact
    assert loaded_artifact.data == {"key": "value"}

def test_artifact_storage_delete(temp_storage, sample_artifact):
    """Test deleting an artifact."""
    # Store the artifact with a named reference
    artifact_ref = temp_storage.store(sample_artifact, "test_ref")
    
    # Delete the artifact
    temp_storage.delete(artifact_ref)
    
    # Verify the artifact is deleted
    with pytest.raises(FileNotFoundError):
        temp_storage.load(artifact_ref)
    
    # Verify the reference is also deleted
    ref_path = Path(temp_storage.base_path) / "refs" / "test_ref.json"
    assert not ref_path.exists()

def test_artifact_validation():
    """Test artifact validation against a schema."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name"]
    }
    
    # Valid artifact
    valid_artifact = Artifact(
        data={"name": "test", "age": 30},
        name="valid.json",
        schema=schema
    )
    
    # Should not raise
    valid_artifact.validate()
    
    # Invalid artifact
    invalid_artifact = Artifact(
        data={"name": 123},  # Invalid type
        name="invalid.json",
        schema=schema
    )
    
    with pytest.raises(jsonschema.ValidationError):
        invalid_artifact.validate()

def test_default_storage():
    """Test getting and setting the default storage."""
    # Get the default storage
    default_storage = get_default_storage()
    assert default_storage is not None
    
    # Create a new storage
    with tempfile.TemporaryDirectory() as temp_dir:
        new_storage = ArtifactStorage(temp_dir)
        set_default_storage(new_storage)
        
        # Verify the default storage was updated
        assert get_default_storage() == new_storage
        
        # Reset to original
        set_default_storage(default_storage)

def test_artifact_serialization(sample_artifact):
    """Test serialization and deserialization of artifacts."""
    # Convert to dict and back
    artifact_dict = sample_artifact.to_dict()
    new_artifact = Artifact.from_dict(artifact_dict)
    
    assert new_artifact.data == sample_artifact.data
    assert new_artifact.metadata.name == sample_artifact.metadata.name
    assert new_artifact.metadata.artifact_type == sample_artifact.metadata.artifact_type

def test_artifact_ref_uri():
    """Test the URI generation for artifact references."""
    ref = ArtifactRef(
        artifact_id="a1b2c3",
        path="blobs/a1/a1b2c3",
        metadata=ArtifactMetadata(
            name="test.json",
            artifact_type=ArtifactType.JSON
        )
    )
    
    assert ref.uri == "artifact://a1b2c3"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
