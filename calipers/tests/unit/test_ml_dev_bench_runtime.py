from unittest.mock import MagicMock

from calipers.framework.base import BaseRuntime
from calipers.runtime.ml_dev_bench_runtime import MLDevBenchRuntime


def test_ml_dev_bench_runtime_initialization():
    mock_runtime = MagicMock(spec=BaseRuntime)
    runtime = MLDevBenchRuntime(runtime=mock_runtime)
    assert runtime._runtime == mock_runtime


def test_ml_dev_bench_runtime_execute_action():
    mock_runtime = MagicMock(spec=BaseRuntime)
    runtime = MLDevBenchRuntime(runtime=mock_runtime)

    action = 'test_action'
    request_data = {'key': 'value'}
    metadata = {'meta': 'data'}

    runtime.execute_action(action, request_data, metadata)
    mock_runtime.execute_action.assert_called_once_with(action, request_data, metadata)
