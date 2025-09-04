"""
REPL (Read-Eval-Print Loop) interface for LLM Orchestrator.

Provides an interactive command-line interface for controlling and monitoring the orchestrator.
"""
from __future__ import annotations

import asyncio
import cmd
import inspect
import json
import logging
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from ..artifacts import Artifact, ArtifactStorage, get_default_storage
from ..tools import ToolRegistry, ToolVersion, tool_registry

logger = logging.getLogger(__name__)

# Type variables for generic command handlers
T = TypeVar('T', bound='BaseCommand')
CommandHandler = Callable[[List[str]], Optional[bool]]

class CommandMetadata(BaseModel):
    """Metadata for REPL commands."""
    name: str
    aliases: List[str] = Field(default_factory=list)
    help: str = ""
    hidden: bool = False
    requires_agent: bool = False

@dataclass
class Command:
    """A REPL command with handler and metadata."""
    handler: CommandHandler
    metadata: CommandMetadata

class CommandError(Exception):
    """Base exception for command-related errors."""
    pass

class CommandRegistry:
    """Registry for REPL commands."""
    
    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        help: str = "",
        hidden: bool = False,
        requires_agent: bool = False
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Decorator to register a command handler."""
        def decorator(func: CommandHandler) -> CommandHandler:
            metadata = CommandMetadata(
                name=name,
                aliases=aliases or [],
                help=help,
                hidden=hidden,
                requires_agent=requires_agent
            )
            
            command = Command(handler=func, metadata=metadata)
            self._commands[name] = command
            
            # Register aliases
            for alias in aliases or []:
                self._aliases[alias] = name
            
            return func
        return decorator
    
    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        # Check exact match
        if name in self._commands:
            return self._commands[name]
        
        # Check aliases
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        
        return None
    
    def get_commands(self) -> List[Command]:
        """Get all registered commands (excluding hidden ones)."""
        return [
            cmd for cmd in self._commands.values()
            if not cmd.metadata.hidden
        ]

class BaseCommand:
    """Base class for command groups."""
    
    def __init__(self, repl: 'REPL'):
        self.repl = repl
        self.registry = CommandRegistry()
        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register commands for this command group."""
        for name in dir(self):
            if name.startswith('do_'):
                method = getattr(self, name)
                if callable(method):
                    # Extract command metadata from docstring
                    doc = inspect.getdoc(method) or ""
                    lines = [line.strip() for line in doc.split('\n')]
                    help_text = lines[0] if lines else ""
                    
                    # Parse command name and aliases from docstring
                    cmd_name = name[3:].replace('_', '-')
                    aliases = []
                    
                    # Look for @aliases in docstring
                    for line in lines[1:]:
                        if line.startswith('@aliases:'):
                            aliases = [a.strip() for a in line[9:].split(',')]
                            break
                    
                    # Register the command
                    self.registry.register(
                        name=cmd_name,
                        aliases=aliases,
                        help=help_text,
                        hidden=cmd_name.startswith('_')
                    )(method)
    
    async def execute(self, command: str, args: List[str]) -> bool:
        """Execute a command."""
        cmd = self.registry.get_command(command)
        if not cmd:
            return False
        
        # Check if command requires an active agent
        if cmd.metadata.requires_agent and not self.repl.agent:
            print("Error: No active agent. Use 'agent create' first.")
            return True
        
        try:
            # Call the command handler
            result = await cmd.handler(args)
            return result is not False
        except Exception as e:
            logger.exception(f"Error executing command '{command}': {e}")
            print(f"Error: {e}")
            return True

class SystemCommands(BaseCommand):
    """System-level commands."""
    
    async def do_help(self, args: List[str]) -> bool:
        """Show help for commands."""
        if not args:
            # Show all commands
            print("\nAvailable commands (type 'help <command>' for details):\n")
            
            # Group commands by category
            categories: Dict[str, List[Command]] = {}
            for cmd in self.registry.get_commands():
                category = cmd.metadata.name.split(' ', 1)[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(cmd)
            
            # Print commands by category
            for category, cmds in categories.items():
                print(f"{category}:")
                for cmd in sorted(cmds, key=lambda c: c.metadata.name):
                    print(f"  {cmd.metadata.name:<20} {cmd.metadata.help}")
                print()
        else:
            # Show help for specific command
            cmd = self.registry.get_command(args[0])
            if cmd:
                print(f"\n{cmd.metadata.name}: {cmd.metadata.help}")
                if cmd.metadata.aliases:
                    print(f"Aliases: {', '.join(cmd.metadata.aliases)}")
                print()
                
                # Show command docstring
                doc = inspect.getdoc(cmd.handler)
                if doc:
                    print(inspect.cleandoc(doc))
                    print()
            else:
                print(f"Unknown command: {args[0]}")
        
        return True
    
    async def do_exit(self, args: List[str]) -> bool:
        """Exit the REPL."""
        print("Exiting...")
        self.repl.should_exit = True
        return True
    
    async def do_clear(self, args: List[str]) -> bool:
        """Clear the screen."""
        print("\033[H\033[J", end="")  # ANSI escape sequence to clear screen
        return True

class REPL:
    """Interactive REPL for LLM Orchestrator."""
    
    def __init__(
        self,
        storage: Optional[ArtifactStorage] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        self.storage = storage or get_default_storage()
        self.tool_registry = tool_registry or tool_registry
        self.agent = None
        self.should_exit = False

        # Initialize command groups
        self.command_groups: List[BaseCommand] = [
            SystemCommands(self),
            # Add more command groups here
        ]
        
        # Build command registry
        self.commands: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}
        self._build_command_registry()
        self._executed_commands = 0
    
    def _build_command_registry(self) -> None:
        """Build the command registry from all command groups."""
        for group in self.command_groups:
            for cmd_name, cmd in group.registry._commands.items():
                self.commands[cmd_name] = cmd
                for alias in cmd.metadata.aliases:
                    self.aliases[alias] = cmd_name
    
    async def execute_command(self, command_line: str) -> bool:
        """Execute a single command line."""
        if not command_line.strip():
            return True
        
        # Parse command and arguments
        parts = shlex.split(command_line)
        if not parts:
            return True
        
        command = parts[0].lower()
        args = parts[1:]
        
        # Handle command aliases
        if command in self.aliases:
            command = self.aliases[command]
        
        # Find and execute the command
        for group in self.command_groups:
            if await group.execute(command, args):
                if command == "exit" and self._executed_commands == 0:
                    raise asyncio.CancelledError
                return True
        
        # Command not found
        print(f"Unknown command: {command}. Type 'help' for a list of commands.")
        return False
    
    async def run(self) -> None:
        """Run the REPL."""
        print("LLM Orchestrator REPL")
        print("Type 'help' for a list of commands, 'exit' to quit.\n")
        
        while not self.should_exit:
            try:
                # Get input with prompt
                try:
                    command_line = input("orchestrator> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nUse 'exit' to quit.")
                    continue

                # Execute the command
                await self.execute_command(command_line)
                self._executed_commands += 1

            except asyncio.CancelledError:
                if self._executed_commands:
                    break
                raise
            except Exception as e:
                logger.exception("Error in REPL:")
                print(f"Error: {e}")

async def start_repl(
    storage: Optional[ArtifactStorage] = None,
    tool_registry: Optional[ToolRegistry] = None
) -> None:
    """Start the REPL with the given dependencies."""
    repl = REPL(storage=storage, tool_registry=tool_registry)
    await repl.run()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Start the REPL
    asyncio.run(start_repl())
