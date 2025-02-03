import typing as t

from composio.tools.base.local import LocalAction, LocalTool

from .actions.chwdir import CWD
from .actions.create_and_write_file import CreateAndWriteFile
from .actions.get_directory_tree import GetDirectoryTree


class MLDevBenchFileTool(LocalTool, autoload=True):  # type: ignore[call-arg]
    """File I/O tool."""

    logo = 'https://raw.githubusercontent.com/ComposioHQ/composio/master/python/docs/imgs/logos/filetool.png'

    @classmethod
    def actions(cls) -> t.List[t.Type[LocalAction]]:
        """Return the list of actions."""
        return [
            CreateAndWriteFile,
            CWD,
            GetDirectoryTree,
        ]
