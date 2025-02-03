import typing as t

from composio.tools.base.local import LocalAction
from composio.tools.local.filetool.actions.base_action import (
    BaseFileRequest,
    BaseFileResponse,
    include_cwd,
)
from pydantic import Field


class GetDirectoryTreeRequest(BaseFileRequest):
    """Request model for getting directory tree."""

    max_limit: t.Optional[int] = Field(
        default=5,
        description='Maximum number of items to show per directory level. If None, shows all items.',
    )
    show_hidden: bool = Field(
        default=False,
        description='Whether to show hidden files and directories (starting with ".")',
    )


class GetDirectoryTreeResponse(BaseFileResponse):
    """Response model for directory tree."""

    tree: str = Field(
        default='', description='ASCII representation of the directory tree'
    )
    error: str = Field(
        default='', description='Error message if any occurred during tree generation'
    )


class GetDirectoryTree(LocalAction[GetDirectoryTreeRequest, GetDirectoryTreeResponse]):
    """Get a visual ASCII representation of the current working directory structure.

    Returns a tree-like ASCII diagram showing files and folders in the current working directory,
    with indentation to indicate nesting levels. This helps visualize the hierarchy and organization
    of files and directories in a human-readable format.

    Example output:
    .
    ├── project/
    │   ├── src/
    │   │   ├── main.py
    │   │   └── utils.py
    │   ├── tests/
    │   │   └── test_main.py
    │   └── README.md
    └── requirements.txt
    """

    @include_cwd
    def execute(
        self, request: GetDirectoryTreeRequest, metadata: t.Dict
    ) -> GetDirectoryTreeResponse:
        """Execute the action."""
        try:
            filemanager = self.filemanagers.get(request.file_manager_id)
            tree = filemanager.tree(
                max_limit=request.max_limit, show_hidden=request.show_hidden
            )
            return GetDirectoryTreeResponse(tree=tree)
        except Exception as e:
            return GetDirectoryTreeResponse(
                error=f'Failed to generate directory tree: {str(e)}'
            )
