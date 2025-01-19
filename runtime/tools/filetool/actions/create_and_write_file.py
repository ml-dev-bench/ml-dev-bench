import logging
import typing as t
from pathlib import Path

from composio.tools.base.local import LocalAction
from composio.tools.local.filetool.actions.base_action import (
    BaseFileRequest,
    BaseFileResponse,
    include_cwd,
)
from pydantic import Field, field_validator


class WriteRequest(BaseFileRequest):
    """Request to write a file, creating it if it doesn't exist."""

    file_path: str = Field(
        ...,  # Changed from Optional to required
        description='The path to the file that will be created and/or edited.',
    )
    content: str = Field(
        ...,
        description='The content that will be written to the file.',
    )

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if v.strip() == '':
            raise ValueError('Path cannot be empty or just whitespace')
        if v in ('.', '..'):
            raise ValueError('Path cannot be "." or ".."')
        return v


# ... WriteResponse class remains the same ...
class WriteResponse(BaseFileResponse):
    """Response to write a file."""

    message: str = Field(
        default='',
        description='Status of the file write operation',
    )
    error: str = Field(
        default='',
        description='Error message if any',
    )


class CreateAndWriteFile(LocalAction[WriteRequest, WriteResponse]):
    """
    Create and write content to a file.

    If the file doesn't exist, it will be created first.
    If it exists, it will be overwritten with the new content.
    """

    @include_cwd  # type: ignore
    def execute(self, request: WriteRequest, metadata: t.Dict) -> WriteResponse:
        try:
            filemanager = self.filemanagers.get(request.file_manager_id)

            # Ensure parent directory exists
            file_path = Path(request.file_path)
            if (
                str(file_path.parent) != '.'
                and str(file_path.parent) != str(filemanager.working_dir)
                and not Path(filemanager.working_dir, file_path.parent).exists()
            ):
                # create parent directory if its not the current working directory
                logging.info(f'Creating parent directory: {file_path.parent}')
                filemanager.create_directory(file_path.parent)

            # Create or get the file
            file = filemanager.create(path=request.file_path)

            # Write the content
            file.write(text=request.content)
            return WriteResponse(
                message=f'File {request.file_path} created successfully!'
            )
        except FileNotFoundError as e:
            return WriteResponse(error=f'File not found: {str(e)}')
        except PermissionError as e:
            return WriteResponse(error=f'Permission denied: {str(e)}')
        except OSError as e:
            return WriteResponse(error=f'OS error occurred: {str(e)}')
