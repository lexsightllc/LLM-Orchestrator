"""
Demo of the tool versioning system.

This script demonstrates:
1. Loading tools from a directory
2. Version resolution
3. Tool execution
4. Migration between versions
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.tools import tool_registry, ToolResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run the tool versioning demo."""
    logger.info("Starting tool versioning demo...")
    
    # Load tools from the examples directory
    tools_dir = Path(__file__).parent / "tools"
    tool_registry.load_from_directory(str(tools_dir))
    
    logger.info("\n=== Tool Versions ===")
    logger.info("Available calculator versions:")
    for name, versions in tool_registry._tools.items():
        logger.info(f"- {name}: {', '.join(versions.keys())}")
    
    # Example 1: Execute a specific version
    logger.info("\n=== Example 1: Execute Specific Version ===")
    calculator = tool_registry.get_tool("calculator", "1.0.0")
    result = calculator.implementation({
        "operation": "add",
        "a": 5,
        "b": 3
    })
    logger.info(f"5 + 3 = {result.data['result']}")
    
    # Example 2: Use version constraints
    logger.info("\n=== Example 2: Version Constraints ===")
    try:
        # This will get the latest 1.x version
        calculator = tool_registry.get_tool("calculator", "^1.0.0")
        logger.info(f"Using calculator version: {calculator.version}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Example 3: Handle migrations
    logger.info("\n=== Example 3: Migration Demo ===")
    try:
        # Simulate a request for v1.0.0 with v1.1.0 parameters
        calculator = tool_registry.get_tool("calculator", "1.0.0")
        
        # This would be the v1.1.0 request format
        v11_params = {
            "operation": "power",
            "a": 2,
            "b": 3,
            "round": 2
        }
        
        # The tool should handle the migration
        result = calculator.implementation(v11_params)
        logger.info(f"2^3 = {result}")
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
    
    logger.info("\nDemo complete!")

if __name__ == "__main__":
    asyncio.run(main())
