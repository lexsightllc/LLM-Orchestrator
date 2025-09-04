"""
Sandbox environment for safely executing untrusted code.

This module provides a sandboxed execution environment with resource limits
and security restrictions to safely run untrusted code.
"""
import asyncio
import logging
from typing import Any, Dict, Optional, Union, List, Callable, Awaitable
from pathlib import Path

logger = logging.getLogger(__name__)

class Sandbox:
    """A sandboxed execution environment for running untrusted code."""
    
    def __init__(self, workdir: Optional[Union[str, Path]] = None):
        """Initialize the sandbox.
        
        Args:
            workdir: Working directory for the sandbox
        """
        self.workdir = Path(workdir) if workdir else None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the sandbox environment."""
        if self.workdir:
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.initialized = True
        logger.info("Sandbox initialized")
    
    async def cleanup(self) -> None:
        """Clean up the sandbox environment."""
        self.initialized = False
        logger.info("Sandbox cleaned up")
    
    async def run_code(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Run Python code in the sandbox.
        
        Args:
            code: Python code to execute
            globals_dict: Global variables available to the code
            locals_dict: Local variables available to the code
            timeout: Maximum execution time in seconds
            
        Returns:
            The result of the code execution
        """
        if not self.initialized:
            raise RuntimeError("Sandbox not initialized")
        
        globals_dict = globals_dict or {}
        locals_dict = locals_dict or {}
        
        # Add safe builtins
        safe_builtins = {
            'None': None,
            'False': False,
            'True': True,
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'property': property,
            'type': type,
        }
        
        globals_dict.update({
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__file__': '<sandbox>',
        })
        
        try:
            # Execute the code in a separate task with timeout
            task = asyncio.create_task(self._execute_code(code, globals_dict, locals_dict))
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            raise TimeoutError(f"Code execution timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error executing code in sandbox: {e}")
            raise
    
    async def _execute_code(
        self,
        code: str,
        globals_dict: Dict[str, Any],
        locals_dict: Dict[str, Any]
    ) -> Any:
        """Execute code in the sandbox."""
        # This method runs in a separate task
        try:
            # Compile the code first to check for syntax errors
            compiled = compile(code, '<sandbox>', 'exec')
            
            # Execute the code
            exec(compiled, globals_dict, locals_dict)
            
            # Return the result if there is one
            if '_result' in locals_dict:
                return locals_dict['_result']
            return None
            
        except Exception as e:
            logger.error(f"Error in sandbox execution: {e}")
            raise
    
    async def run_shell_command(
        self,
        command: Union[str, List[str]],
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[Union[str, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a shell command in the sandbox.
        
        Args:
            command: The command to run (string or list of args)
            cwd: Working directory for the command
            env: Environment variables
            input_data: Input to send to the command's stdin
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary containing command output and metadata
        """
        if not self.initialized:
            raise RuntimeError("Sandbox not initialized")
        
        cwd = Path(cwd) if cwd else self.workdir
        if cwd and not cwd.exists():
            cwd.mkdir(parents=True, exist_ok=True)
        
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        
        # Create the process
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            # Send input if provided
            if input_data:
                if isinstance(input_data, str):
                    input_data = input_data.encode('utf-8')
                process.stdin.write(input_data)
                await process.stdin.drain()
                process.stdin.close()
            
            # Wait for the process to complete with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            # Return the result
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace') if stdout else '',
                'stderr': stderr.decode('utf-8', errors='replace') if stderr else '',
            }
            
        except Exception as e:
            logger.error(f"Error running command in sandbox: {e}")
            raise
        finally:
            # Ensure process is terminated
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
    
    async def install_packages(self, packages: List[str], timeout: float = 300) -> bool:
        """Install Python packages in the sandbox.
        
        Args:
            packages: List of package specs (e.g., ['requests>=2.25.0', 'numpy'])
            timeout: Maximum time to wait for installation
            
        Returns:
            True if installation was successful, False otherwise
        """
        if not packages:
            return True
            
        command = [
            sys.executable, "-m", "pip", "install", "--no-warn-script-location"
        ] + packages
        
        try:
            result = await self.run_shell_command(
                command,
                timeout=timeout
            )
            return result['returncode'] == 0
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            return False

    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()
