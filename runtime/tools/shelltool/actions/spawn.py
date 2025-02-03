"""Spawn a process."""

import subprocess
import time
import typing as t
from pathlib import Path

from composio.tools.base.local import LocalAction
from pydantic import BaseModel, Field

# pylint: disable=consider-using-with,unspecified-encoding


class SpawnRequest(BaseModel):
    """Execute command request."""

    cmd: str = Field(
        ...,
        description='Command to be executed.',
        examples=[
            '/bin/python /home/user/server.py',
            'node /home/user/server.js',
            'yarn start',
        ],
    )
    working_dir: t.Optional[str] = Field(
        None,
        description=(
            'Directory where this command should be executed, '
            'if not provided the current directory will be used'
        ),
        examples=[
            '/home/user',
            './',
        ],
    )


class SpawnResponse(BaseModel):
    """Shell execution response."""

    stdout: str = Field(
        ...,
        description='Path to the file containing the stdout stream',
    )
    stderr: str = Field(
        ...,
        description='Path to the file containing the stderr stream.',
    )
    pid: str = Field(
        ...,
        description='Path to the file containing the process ID for the spawned process',
    )


class SpawnProcess(LocalAction[SpawnRequest, SpawnResponse]):
    """
    Spawn a process.

    Use this action to launch processes on background, for example launch a
    python process using cmd: python path/to/script.py
    Please use this action for launching long running commands.
    The outputs of the command and the process ID are stored in the shared files
    """

    _tags = ['workspace', 'shell']

    def execute(self, request: SpawnRequest, metadata: t.Dict) -> SpawnResponse:
        """Execute a shell command."""
        env = self.shells.get().environment

        workspace_dir = env.get('MLDevBench_WORKSPACE_DIR')
        if workspace_dir is None:
            raise ValueError(
                'MLDevBench_WORKSPACE_DIR is not set in the environment variables'
            )

        # Create .spawn_process directory if it doesn't exist
        spawn_dir = Path(workspace_dir) / '.spawn_process'
        spawn_dir.mkdir(exist_ok=True)

        # Create a unique directory name based on the command
        cmd_hash = hash(request.cmd)
        base_dir_name = f'process_{cmd_hash}'
        process_dir = spawn_dir / base_dir_name

        # If directory exists, append timestamp to make it unique
        if process_dir.exists():
            timestamp = hex(int(time.time()))[2:]  # Remove '0x' prefix
            process_dir = spawn_dir / f'{base_dir_name}_{timestamp}'

        process_dir.mkdir(parents=True)

        stdout = process_dir / 'stdout.txt'
        stderr = process_dir / 'stderr.txt'
        process = subprocess.Popen(
            request.cmd,
            start_new_session=True,
            stdout=stdout.open('w+', buffering=1),
            stderr=stderr.open('w+', buffering=1),
            env=env,
            bufsize=1,
            shell=True,  # Required for composite commands using && or ;
            executable='/bin/bash',  # Explicitly use bash for better command support
            text=True,  # Use text mode instead of binary
            cwd=str(request.working_dir or workspace_dir),
        )
        pid = process_dir / 'pid.txt'
        pid.write_text(str(process.pid))

        return SpawnResponse(stdout=str(stdout), stderr=str(stderr), pid=str(pid))
