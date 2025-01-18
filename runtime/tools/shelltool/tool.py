"""Tool for executing shell commands."""

import typing as t

from composio.tools.base.local import LocalAction, LocalTool

from .actions.exec import ExecCommand
from .actions.sleep_and_execute import SleepAndExecute
from .actions.spawn import SpawnProcess


class MLDevBenchShellTool(LocalTool, autoload=True):  # type: ignore[call-arg]
    """Tool for executing shell commands."""

    logo = 'https://raw.githubusercontent.com/ComposioHQ/composio/master/python/docs/imgs/logos/shelltool.png'

    @classmethod
    def actions(cls) -> t.List[t.Type[LocalAction]]:
        """Returns list of actions."""
        return [
            ExecCommand,
            SpawnProcess,
            SleepAndExecute,
        ]
