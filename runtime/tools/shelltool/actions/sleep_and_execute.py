"""Tool for sleeping and optionally executing shell commands."""

import time
import typing as t

from composio.tools.base.local import LocalAction
from composio.tools.env.constants import EXIT_CODE, STDERR, STDOUT
from pydantic import Field

from .exec import ShellExecResponse, ShellRequest


class SleepAndExecuteRequest(ShellRequest):
    """Sleep and execute command request."""

    duration: float = Field(
        ...,
        description='Duration to sleep in seconds',
    )
    cmd: str = Field(
        default='',
        description='Optional command to be executed after sleeping',
    )


class SleepAndExecute(LocalAction[SleepAndExecuteRequest, ShellExecResponse]):
    """
    Sleep for a specified duration and optionally execute a shell command afterwards.

    Examples:
      1. Sleep for 5 seconds: duration=5
      2. Sleep for 2 seconds and then list files: duration=2, cmd='ls -l'

    Note: The environment doesnt support interactive commands like `vim`, `watch` etc.
    """

    _tags = ['workspace', 'shell']

    def execute(
        self, request: SleepAndExecuteRequest, metadata: t.Dict
    ) -> ShellExecResponse:
        """Sleep for the specified duration and optionally execute a command."""
        self.logger.debug(f'Sleeping for {request.duration} seconds')
        time.sleep(request.duration)

        shell = self.shells.get(id=request.shell_id)

        if not request.cmd:
            # If no command provided, return empty response
            return ShellExecResponse(
                stdout='',
                stderr='',
                exit_code=0,
                current_shell_pwd=f'{shell.exec(cmd="pwd")[STDOUT].strip()}',
            )

        self.logger.debug(f'Executing {request.cmd} @ {shell}')
        output = shell.exec(cmd=request.cmd)

        return ShellExecResponse(
            stdout=output[STDOUT],
            stderr=output[STDERR],
            exit_code=int(output[EXIT_CODE]),
            current_shell_pwd=f'{shell.exec(cmd="pwd")[STDOUT].strip()}',
        )
