# SPDX-License-Identifier: MPL-2.0
"""
Demo of the artifact management system.

This script demonstrates:
1. Creating different types of artifacts
2. Storing and loading artifacts
3. Using content-addressable storage
4. Schema validation
"""
import asyncio
import json
import logging
import tempfile
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.artifacts import (
    Artifact, ArtifactType, ArtifactStorage,
    create_json_artifact, create_text_artifact
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run the artifact management demo."""
    logger.info("Starting artifact management demo...")
    
    # Create a temporary directory for storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ArtifactStorage(temp_dir)
        
        logger.info("\n=== Example 1: Storing and Loading Artifacts ===")
        
        # Create a JSON artifact
        json_data = {
            "name": "John Doe",
            "age": 30,
            "skills": ["Python", "Machine Learning", "Cloud"]
        }
        
        json_artifact = create_json_artifact(
            json_data,
            name="user_profile.json",
            schema_ref="schemas/user_profile_v1.json",
            author="demo",
            tags=["user", "profile"]
        )
        
        # Store the artifact
        json_ref = storage.store(json_artifact, "user_profile_latest")
        logger.info(f"Stored JSON artifact with ID: {json_ref.artifact_id}")
        logger.info(f"Artifact URI: {json_ref.uri}")
        logger.info(f"Artifact path: {json_ref.path}")
        
        # Create a text artifact
        text_content = """
        This is a sample text document.
        It contains multiple lines of text.
        """
        
        text_artifact = create_text_artifact(
            text_content,
            name="sample.txt",
            description="Sample text document"
        )
        
        # Store the text artifact
        text_ref = storage.store(text_artifact, "sample_text")
        logger.info(f"\nStored text artifact with ID: {text_ref.artifact_id}")
        
        logger.info("\n=== Example 2: Loading Artifacts ===")
        
        # Load artifacts by reference
        loaded_json = storage.load("user_profile_latest")
        loaded_text = storage.load(text_ref.artifact_id)
        
        logger.info(f"Loaded JSON data: {json.dumps(loaded_json.data, indent=2)}")
        logger.info(f"Loaded text content: {loaded_text.data[:50]}...")
        
        logger.info("\n=== Example 3: Schema Validation ===")
        
        # Define a schema
        user_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["name", "age"]
        }
        
        # Test validation
        is_valid = loaded_json.validate(user_schema)
        logger.info(f"JSON data is valid against schema: {is_valid}")
        
        # Test with invalid data
        invalid_data = {"name": "Jane", "age": -5}  # Invalid age
        invalid_artifact = create_json_artifact(invalid_data)
        is_valid = invalid_artifact.validate(user_schema)
        logger.info(f"Invalid data validation result: {is_valid}")
        
        logger.info("\n=== Example 4: Content-Addressable Storage ===")
        
        # Store the same content again
        same_json_artifact = create_json_artifact(json_data)
        same_json_ref = storage.store(same_json_artifact)
        
        # Should have the same content hash
        logger.info(f"Original artifact ID: {json_ref.artifact_id}")
        logger.info(f"Same content artifact ID: {same_json_ref.artifact_id}")
        logger.info(f"Same content? {json_ref.artifact_id == same_json_ref.artifact_id}")
        
        # Different content gets different hash
        different_artifact = create_json_artifact({"different": "data"})
        different_ref = storage.store(different_artifact)
        logger.info(f"Different content artifact ID: {different_ref.artifact_id}")
        
        logger.info("\nDemo complete!")
        logger.info(f"Artifacts stored in: {temp_dir}")

if __name__ == "__main__":
    asyncio.run(main())
