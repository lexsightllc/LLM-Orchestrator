# SPDX-License-Identifier: MPL-2.0
"""
Demo of the REPL interface for LLM Orchestrator.

This script demonstrates:
1. Starting an interactive REPL session
2. Registering and executing commands
3. Handling command arguments
4. Using command groups
"""
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.repl import REPL, CommandRegistry
from orchestrator.artifacts import ArtifactStorage, create_json_artifact
from orchestrator.tools import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoCommands:
    """Example command group for the demo."""
    
    def __init__(self, repl):
        self.repl = repl
        self.registry = CommandRegistry()
        self.counter = 0
        
        # Register commands
        self._register_commands()
    
    def _register_commands(self):
        """Register commands for this group."""
        
        @self.registry.register(
            "hello",
            aliases=["hi"],
            help="Say hello"
        )
        async def hello_cmd(repl, args):
            """Say hello to someone.
            
            Usage: hello [name]
            """
            name = args[0] if args else "there"
            print(f"Hello, {name}!")
            return True
        
        @self.registry.register(
            "counter",
            aliases=["count", "c"],
            help="Increment and display a counter"
        )
        async def counter_cmd(repl, args):
            """Increment and display a counter.
            
            Usage: counter [increment]
            """
            increment = int(args[0]) if args and args[0].isdigit() else 1
            self.counter += increment
            print(f"Counter: {self.counter}")
            return True
        
        @self.registry.register(
            "artifact",
            aliases=["a"],
            help="Create and manage artifacts"
        )
        async def artifact_cmd(repl, args):
            """Create and manage artifacts.
            
            Usage:
              artifact create <name> <value>  Create a JSON artifact
              artifact list                  List all artifacts
            """
            if not args:
                print("Missing subcommand. Use 'help artifact' for usage.")
                return True
                
            subcmd = args[0].lower()
            
            if subcmd == "create" and len(args) >= 3:
                # Create a JSON artifact
                name = args[1]
                value = " ".join(args[2:])
                
                artifact = create_json_artifact(
                    {"value": value},
                    name=f"{name}.json",
                    description=f"Demo artifact: {name}"
                )
                
                # Store the artifact
                ref = repl.storage.store(artifact, name)
                print(f"Created artifact: {ref.uri}")
                print(f"  Path: {ref.path}")
                print(f"  Type: {ref.metadata.artifact_type}")
                print(f"  Size: {ref.metadata.size_bytes} bytes")
                
            elif subcmd == "list":
                # List artifacts (simplified)
                print("Artifacts:")
                for ref_file in (repl.storage.base_path / "refs").glob("*.json"):
                    print(f"- {ref_file.stem}")
                    
            else:
                print(f"Unknown subcommand: {subcmd}")
                print("Use 'help artifact' for usage.")
            
            return True

async def main():
    """Run the REPL demo."""
    logger.info("Starting REPL demo...")
    
    # Set up storage and tool registry
    storage = ArtifactStorage("demo_artifacts")
    tool_registry = ToolRegistry()
    
    # Create and configure the REPL
    repl = REPL(storage=storage, tool_registry=tool_registry)
    
    # Add demo commands
    demo_commands = DemoCommands(repl)
    repl.command_groups.append(demo_commands)
    
    # Add the demo commands to the REPL's command registry
    repl._build_command_registry()
    
    # Print welcome message
    print("\n" + "=" * 50)
    print("LLM Orchestrator - Interactive REPL Demo")
    print("=" * 50)
    print("\nTry these commands:")
    print("  hello [name]      - Say hello")
    print("  counter [inc]     - Increment and display a counter")
    print("  artifact create <name> <value> - Create a JSON artifact")
    print("  artifact list     - List artifacts")
    print("  help              - Show help")
    print("  exit              - Exit the REPL\n")
    
    # Run the REPL
    await repl.run()
    
    logger.info("Demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
