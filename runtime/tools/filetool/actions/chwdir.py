from typing import Dict

from composio.tools.base.local import LocalAction
from composio.tools.env.constants import EXIT_CODE, STDERR, STDOUT
from composio.tools.local.filetool.actions.base_action import (
    BaseFileRequest,
    BaseFileResponse,
    include_cwd,
)
from pydantic import Field


class ChwdirRequest(BaseFileRequest):
    """Request to change the current working directory."""

    path: str = Field(
        ...,
        description='The path to change the current working directory to. '
        "Can be absolute, relative to the current working directory, or use '..' to navigate up the directory tree.",
    )

    shell_id: str = Field(
        default='',
        description=(
            'ID of the shell where this command will be executed, if not '
            'provided the recent shell will be used to execute the action'
        ),
    )


class MLDevBenchChwdirResponse(BaseFileResponse):
    """Response to change the current working directory."""

    stdout: str = Field(
        ...,
        description='Output captured from the execution of the command',
    )
    stderr: str = Field(
        ...,
        description='Errors captured during execution of the command',
    )
    exit_code: int = Field(
        ...,
        description='Exit code of the command',
    )
    current_shell_pwd: str = Field(
        default='',
        description="Current shell's working directory",
    )


class CWD(LocalAction[ChwdirRequest, MLDevBenchChwdirResponse]):
    """
    Changes the current working directory of the file manager and the shell to the specified
    path. The shell and file tool commands after this action will be executed in this new directory.

    Can result in:
    - PermissionError: If the user doesn't have permission to access the directory.
    - FileNotFoundError: If the directory or any parent directory does not exist.
    - RuntimeError: If the path cannot be resolved due to a loop or other issues.
    """

    @include_cwd  # type: ignore
    def execute(
        self, request: ChwdirRequest, metadata: Dict
    ) -> MLDevBenchChwdirResponse:
        try:
            self.filemanagers.get(request.file_manager_id).chdir(request.path)
            shell = self.shells.get(id=request.shell_id)
            output = shell.exec(cmd=f'cd {request.path}')

            return MLDevBenchChwdirResponse(
                stdout=output[STDOUT],
                stderr=output[STDERR],
                exit_code=int(output[EXIT_CODE]),
                current_shell_pwd=f'{shell.exec(cmd="pwd")[STDOUT].strip()}',
            )
        except PermissionError as e:
            return MLDevBenchChwdirResponse(
                stderr=f'Permission denied: {str(e)}', exit_code=1
            )
        except FileNotFoundError as e:
            return MLDevBenchChwdirResponse(
                stderr=f'Directory not found: {str(e)}', exit_code=1
            )
        except RuntimeError as e:
            return MLDevBenchChwdirResponse(
                stderr=f'Unable to resolve path: {str(e)}', exit_code=1
            )
        except OSError as e:
            return MLDevBenchChwdirResponse(
                stderr=f'Unable to resolve path: {str(e)}', exit_code=1
            )
