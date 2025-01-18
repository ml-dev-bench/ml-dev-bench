from typing import Any, Dict, Optional

from calipers.framework.base import BaseRuntime


class MLDevBenchRuntime(BaseRuntime):
    def __init__(self, runtime: Optional[BaseRuntime] = None) -> None:
        self._runtime: Optional[BaseRuntime] = runtime

    def execute_action(
        self, action: str, request_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._runtime is None:
            raise RuntimeError('Runtime not initialized')
        return self._runtime.execute_action(action, request_data, metadata)
