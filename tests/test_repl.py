"""Tests for the REPL interface."""
import asyncio
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from orchestrator.repl import REPL, CommandRegistry, Command, CommandMetadata, SystemCommands

class TestREPL(unittest.IsolatedAsyncioTestCase):
    """Test the REPL interface."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.repl = REPL()
        self.mock_input = []
        self.mock_output = []
        
        # Patch input and print for testing
        self.input_patcher = patch('builtins.input', side_effect=self.mock_input_func)
        self.print_patcher = patch('builtins.print', side_effect=self.mock_print_func)
        self.input_patcher.start()
        self.print_patcher.start()
    
    async def asyncTearDown(self):
        """Tear down test fixtures."""
        self.input_patcher.stop()
        self.print_patcher.stop()
    
    def mock_input_func(self, prompt=None):
        """Mock input function."""
        if not self.mock_input:
            self.repl.should_exit = True
            return "exit"
        return self.mock_input.pop(0)
    
    def mock_print_func(self, *args, **kwargs):
        """Mock print function."""
        self.mock_output.append(" ".join(str(a) for a in args))
    
    def assertOutputContains(self, text):
        """Assert that the output contains the given text."""
        output = "\n".join(self.mock_output)
        self.assertIn(text, output)
    
    async def test_help_command(self):
        """Test the help command."""
        self.mock_input = ["help", "exit"]
        
        await self.repl.run()
        
        self.assertOutputContains("Available commands")
        self.assertOutputContains("help")
        self.assertOutputContains("exit")
    
    async def test_unknown_command(self):
        """Test handling of unknown commands."""
        self.mock_input = ["nonexistent_command", "exit"]
        
        await self.repl.run()
        
        self.assertOutputContains("Unknown command")
    
    async def test_exit_command(self):
        """Test the exit command."""
        self.mock_input = ["exit"]
        
        with self.assertRaises(asyncio.CancelledError):
            await self.repl.run()
        
        self.assertTrue(self.repl.should_exit)
    
    async def test_clear_command(self):
        """Test the clear screen command."""
        self.mock_input = ["clear", "exit"]
        
        await self.repl.run()
        
        # Check for ANSI clear screen sequence
        self.assertIn("\033[H\033[J", "\n".join(self.mock_output))

class TestCommandRegistry(unittest.TestCase):
    """Test the command registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = CommandRegistry()
        
        @self.registry.register("test", aliases=["t"], help="Test command")
        async def test_cmd(repl, args):
            return True
        
        self.test_cmd = test_cmd
    
    def test_register_command(self):
        """Test registering a command."""
        self.assertIn("test", self.registry._commands)
        self.assertEqual(self.registry._commands["test"].handler, self.test_cmd)
        self.assertEqual(self.registry._commands["test"].metadata.aliases, ["t"])
    
    def test_get_command(self):
        """Test getting a command by name or alias."""
        # By name
        cmd = self.registry.get_command("test")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.handler, self.test_cmd)
        
        # By alias
        cmd = self.registry.get_command("t")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.handler, self.test_cmd)
    
    def test_get_commands(self):
        """Test getting all non-hidden commands."""
        # Add a hidden command
        @self.registry.register("hidden", hidden=True)
        async def hidden_cmd(repl, args):
            return True
        
        commands = self.registry.get_commands()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].metadata.name, "test")

class TestSystemCommands(unittest.IsolatedAsyncioTestCase):
    """Test system commands."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.repl = REPL()
        self.cmds = SystemCommands(self.repl)
        self.mock_output = []
        
        # Patch print for testing
        self.print_patcher = patch('builtins.print', side_effect=self.mock_print_func)
        self.print_patcher.start()
    
    async def asyncTearDown(self):
        """Tear down test fixtures."""
        self.print_patcher.stop()
    
    def mock_print_func(self, *args, **kwargs):
        """Mock print function."""
        self.mock_output.append(" ".join(str(a) for a in args))
    
    async def test_help_specific_command(self):
        """Test getting help for a specific command."""
        await self.cmds.do_help(["help"])
        
        output = "\n".join(self.mock_output)
        self.assertIn("help:", output)
        self.assertIn("Show help for commands", output)
    
    async def test_exit_command(self):
        """Test the exit command."""
        result = await self.cmds.do_exit([])
        self.assertTrue(result)
        self.assertTrue(self.repl.should_exit)

if __name__ == "__main__":
    unittest.main()
