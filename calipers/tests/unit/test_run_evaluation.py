# import json
# import os
# import shutil
# import tempfile
# from datetime import datetime, timedelta
# from pathlib import Path
# from typing import Any, Dict
# from unittest.mock import MagicMock, patch

# import litellm
# import pytest
# import yaml

# import calipers
# from calipers.framework import (
#     BaseAgent,
#     BaseEvaluationTask,
#     EvalRegistry,
#     EvaluationFramework,
#     EvaluationResult,
#     RunResult,
# )
# from calipers.framework.base import BaseRuntime
# from calipers.framework.config import TaskConfig
# from calipers.metrics import (
#     CounterMetric,
#     MetricsTracker,
#     RuntimeMetric,
#     StepsMetric,
#     TokenCostMetric,
#     TokensMetric,
# )
# from calipers.scripts.run_evaluation import (
#     main,
#     print_results,
#     run_evaluation,
#     save_results,
# )


# class MockAgent(BaseAgent):
#     agent_id = 'test_agent'
#     description = 'Mock agent for testing'

#     def __init__(self, config):
#         super().__init__(config)
#         self._runtime = None

#     def uses_litellm(self) -> bool:
#         return self.config.config.get('config', {}).get('uses_litellm', False)

#     async def run(self, task: str) -> dict:
#         # Simulate a LiteLLM call by directly calling the callback if it exists
#         if self.uses_litellm() and litellm.callbacks:
#             mock_response = {
#                 'standard_logging_object': {
#                     'response': {
#                         'usage': {
#                             'completion_tokens': 50,
#                             'prompt_tokens': 25,
#                             'total_tokens': 75,
#                         }
#                     }
#                 },
#                 'response_cost': 0.001,
#             }
#             start_time = datetime.now()
#             end_time = start_time + timedelta(seconds=0.5)

#             for callback in litellm.callbacks:
#                 await callback.async_log_success_event(
#                     mock_response, None, start_time, end_time
#                 )

#         return {'success': True, 'token_usage': 100}

#     @property
#     def runtime(self) -> BaseRuntime:
#         if self._runtime is None:
#             self._runtime = MagicMock(spec=BaseRuntime)
#         return self._runtime


# class MockTask(BaseEvaluationTask):
#     task_id = 'task1'
#     description = 'Mock task for testing'
#     categories = {'test', 'mock'}

#     def __init__(self, config: TaskConfig):
#         super().__init__(config)
#         # Add all required metrics
#         self.metrics.add_metric(TokensMetric())
#         self.metrics.add_metric(RuntimeMetric())
#         self.metrics.add_metric(TokenCostMetric())
#         self.metrics.add_metric(StepsMetric())

#     async def run(self, agent: BaseAgent) -> dict:
#         return await agent.run('test task')

#     async def validate(self, agent_output: dict, runtime: BaseRuntime) -> dict:
#         # Update metrics based on agent output
#         if 'token_usage' in agent_output:
#             self.update_metric('tokens', agent_output['token_usage'])

#         if 'steps_completed' in agent_output:
#             self.update_metric('steps', agent_output['steps_completed'])

#         return {
#             'success': agent_output.get('success', False),
#             'validation_details': agent_output,
#         }


# class MockCallbackHandler:
#     """Mock callback handler for LiteLLM tests"""

#     def __init__(self):
#         self.metrics_tracker: MetricsTracker = MetricsTracker()
#         self.reset_called = False

#     async def async_log_success_event(
#         self,
#         kwargs: Dict[str, Any],
#         response_obj: Any,
#         start_time: datetime,
#         end_time: datetime,
#     ) -> None:
#         # Update steps
#         self.metrics_tracker.update('steps', 1)

#         # Update duration
#         if start_time and end_time:
#             duration = (end_time - start_time).total_seconds()
#             self.metrics_tracker.update('runtime', duration)

#         # Update cost
#         cost = kwargs.get('response_cost')
#         if cost is not None:
#             self.metrics_tracker.update('token_cost', cost)

#         # Update token metrics
#         usage = (
#             kwargs.get('standard_logging_object', {})
#             .get('response', {})
#             .get('usage', {})
#         )
#         if usage:
#             for key, value in usage.items():
#                 metric_name = f'token__{key}'
#                 if metric_name not in self.metrics_tracker.metrics:
#                     self.metrics_tracker.add_metric(
#                         CounterMetric(
#                             name=metric_name,
#                             description=f'Token usage metric for {key}',
#                             unit='tokens',
#                         )
#                     )
#                 self.metrics_tracker.update(metric_name, value)

#     async def async_log_failure_event(
#         self,
#         kwargs: Dict[str, Any],
#         response_obj: Any,
#         start_time: datetime,
#         end_time: datetime,
#     ) -> None:
#         await self.async_log_success_event(kwargs, response_obj, start_time, end_time)

#     def get_accumulated_metrics(self) -> Dict[str, Any]:
#         return self.metrics_tracker.get_all()

#     def reset(self) -> None:
#         self.reset_called = True
#         self.metrics_tracker.reset()


# @pytest.fixture
# def register_mocks():
#     """Register mock classes with EvalRegistry"""
#     # Register mock classes
#     EvalRegistry._tasks['task1'] = MockTask
#     EvalRegistry._agents['test_agent'] = MockAgent

#     yield

#     # Cleanup registry
#     EvalRegistry._tasks.pop('task1', None)
#     EvalRegistry._agents.pop('test_agent', None)


# @pytest.fixture
# def mock_results():
#     """Create mock evaluation results"""
#     from datetime import datetime, timedelta

#     start_time = datetime.now()
#     run1 = RunResult(
#         success=True,
#         validation_details={'test': 'details1'},
#         metrics={
#             'tokens': {
#                 'value': 100,
#                 'unit': 'tokens',
#                 'description': 'Number of tokens used',
#             },
#             'duration': {
#                 'value': 1.0,
#                 'unit': 'seconds',
#                 'description': 'Run duration',
#             },
#         },
#         start_time=start_time,
#         end_time=start_time + timedelta(seconds=1),
#     )

#     run2 = RunResult(
#         success=False,
#         validation_details={'test': 'details2'},
#         metrics={
#             'tokens': {
#                 'value': 200,
#                 'unit': 'tokens',
#                 'description': 'Number of tokens used',
#             },
#             'duration': {
#                 'value': 2.0,
#                 'unit': 'seconds',
#                 'description': 'Run duration',
#             },
#         },
#         start_time=start_time + timedelta(seconds=1),
#         end_time=start_time + timedelta(seconds=2),
#     )

#     return [
#         EvaluationResult(
#             task_id='test_task',
#             agent_id='test_agent',
#             categories={'test', 'mock'},
#             runs=[run1, run2],
#         )
#     ]


# def test_save_results(mock_results, tmp_path):
#     """Test saving results to JSON file"""
#     config = {'agent': {'id': 'test_agent'}, 'tasks': [{'id': 'task1'}]}
#     commit_hash = 'test123'

#     save_results(mock_results, tmp_path, config, commit_hash)

#     # Find the saved file
#     result_files = list(tmp_path.glob('eval_results_*.json'))
#     assert len(result_files) == 1

#     # Load and verify contents
#     with open(result_files[0]) as f:
#         data = json.load(f)

#     assert 'metadata' in data
#     assert data['metadata']['commit_hash'] == commit_hash
#     assert data['metadata']['config'] == config
#     assert len(data['results']) == 1

#     result = data['results'][0]
#     assert result['task_id'] == 'test_task'
#     assert result['success_rate'] == 0.5
#     assert len(result['runs']) == 2


# def test_print_results(mock_results, caplog):
#     """Test results printing"""
#     caplog.set_level('INFO')

#     print_results(mock_results)

#     # Verify log output
#     assert 'test_task' in caplog.text
#     assert 'Success Rate: 0.50' in caplog.text
#     assert 'tokens' in caplog.text
#     # Verify numeric formatting
#     assert 'mean: ' in caplog.text
#     assert 'min: ' in caplog.text
#     assert 'max: ' in caplog.text


# @pytest.fixture
# def test_workspace(tmp_path):
#     """Create and manage test workspace"""
#     workspace = tmp_path / 'test_workspace'
#     workspace.mkdir(exist_ok=True)

#     yield workspace

#     # Cleanup
#     if workspace.exists():
#         shutil.rmtree(workspace)


# @pytest.mark.asyncio
# async def test_main_with_categories(tmp_path, test_workspace, register_mocks):
#     """Test main function with category filtering"""
#     config = {
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#             'workspace_dir': str(test_workspace),  # Use test workspace
#         },
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'categories': ['test', 'mock'],
#                 'config': {},
#                 'workspace_dir': str(test_workspace),  # Use test workspace
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         import yaml

#         yaml.dump(config, f)

#     # Mock command line arguments
#     with patch('argparse.ArgumentParser.parse_args') as mock_args:
#         args = MagicMock()
#         args.config = str(config_file)
#         args.categories = [['test']]
#         args.tasks = None
#         args.output_dir = str(tmp_path)
#         args.commit_hash = 'test123'
#         args.langchain_project = None
#         mock_args.return_value = args

#         # Run main
#         await main()

#         # Verify results file was created
#         result_files = list(tmp_path.glob('eval_results_*.json'))
#         assert len(result_files) == 1

#         # Load and verify results
#         with open(result_files[0]) as f:
#             results_data = json.load(f)

#         # Check metadata
#         assert 'metadata' in results_data
#         assert results_data['metadata']['commit_hash'] == 'test123'
#         assert results_data['metadata']['config'] == config

#         # Check results
#         assert 'results' in results_data
#         results = results_data['results']
#         assert len(results) == 1  # One task matched the category

#         # Verify task details
#         task_result = results[0]
#         assert task_result['task_id'] == 'task1'
#         assert task_result['agent_id'] == 'test_agent'
#         assert set(task_result['categories']) == {'test', 'mock'}
#         assert len(task_result['runs']) == 1  # Default num_runs=1


# @pytest.mark.asyncio
# async def test_main_with_tasks(tmp_path, test_workspace, register_mocks):
#     """Test main function with specific tasks"""
#     config = {
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#             'workspace_dir': str(test_workspace),  # Use test workspace
#         },
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'config': {},
#                 'workspace_dir': str(test_workspace),  # Use test workspace
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         import yaml

#         yaml.dump(config, f)

#     # Mock command line arguments
#     with patch('argparse.ArgumentParser.parse_args') as mock_args:
#         args = MagicMock()
#         args.config = str(config_file)
#         args.categories = None
#         args.tasks = ['task1']
#         args.output_dir = str(tmp_path)
#         args.commit_hash = 'test123'
#         args.langchain_project = None
#         mock_args.return_value = args

#         # Run main
#         await main()


# @pytest.mark.asyncio
# async def test_main_validation_error():
#     """Test main function exits with error on validation failure"""
#     config = {
#         'workspace_dir': '/test/workspace',
#         'agent': {'id': 'test_agent'},
#         'tasks': [{'id': 'task1'}],
#     }

#     config_file = 'test_config.yaml'

#     # Mock all the dependencies
#     with (
#         patch('calipers.scripts.run_evaluation.open', create=True) as mock_open,
#         patch('yaml.safe_load') as mock_yaml,
#         patch('calipers.scripts.run_evaluation.run_evaluation') as mock_run,
#     ):
#         # Setup mocks
#         mock_open.return_value.__enter__.return_value = 'file'
#         mock_yaml.return_value = config

#         # Create a real EvaluationResult for the mock
#         start_time = datetime.now()
#         run_result = RunResult(
#             success=False,
#             validation_details={},
#             metrics={},
#             start_time=start_time,
#             end_time=start_time + timedelta(seconds=1),
#         )
#         eval_result = EvaluationResult(
#             task_id='task1',
#             agent_id='test_agent',
#             categories=set(),
#             runs=[run_result],
#         )

#         # Mock run_evaluation to return a real result
#         mock_run.return_value = [eval_result]

#         # Mock args after setting up run_evaluation mock
#         with patch('argparse.ArgumentParser.parse_args') as mock_args:
#             mock_args.return_value.config = config_file
#             mock_args.return_value.tasks = None
#             mock_args.return_value.output_dir = 'test_output'
#             mock_args.return_value.commit_hash = None

#             with pytest.raises(SystemExit) as exc_info:
#                 await main()

#             assert exc_info.value.code == 1


# @pytest.mark.asyncio
# async def test_output_json_format(tmp_path, test_workspace, register_mocks):
#     """Test the structure and content of the output JSON file"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#             'workspace_dir': str(test_workspace),
#         },
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'categories': ['test', 'mock'],
#                 'config': {},
#                 'workspace_dir': str(test_workspace),
#             }
#         ],
#         'num_runs': 2,
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Run evaluation
#     with patch('argparse.ArgumentParser.parse_args') as mock_args:
#         mock_args.return_value.config = str(config_file)
#         mock_args.return_value.tasks = None
#         mock_args.return_value.output_dir = str(tmp_path)
#         mock_args.return_value.commit_hash = 'test123'

#         # Mock save_results to avoid serialization issues
#         with patch('calipers.scripts.run_evaluation.save_results') as mock_save:
#             await run_evaluation(
#                 config_path=str(config_file),
#                 output_dir=str(tmp_path),
#                 commit_hash='test123',
#             )

#             # Verify save_results was called with correct data
#             assert mock_save.called
#             saved_results = mock_save.call_args[0][0]
#             assert len(saved_results) == 1
#             assert saved_results[0].task_id == 'task1'
#             assert len(saved_results[0].runs) == 2


# @pytest.mark.asyncio
# async def test_category_filter_evaluation(tmp_path, test_workspace, register_mocks):
#     """Test evaluation with category filters"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'category_filters': [
#             ['test', 'mock']
#         ],  # Use category_filters instead of categories
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock save_results to avoid serialization issues
#     with patch('calipers.scripts.run_evaluation.save_results'):
#         results = await run_evaluation(str(config_file))
#         assert len(results) == 1
#         assert results[0].task_id == 'task1'


# @pytest.mark.asyncio
# async def test_mutually_exclusive_config(tmp_path, test_workspace, register_mocks):
#     """Test that config cannot have both category_filters and tasks"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'category_filters': [['test', 'mock']],
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'categories': ['test', 'mock'],
#                 'config': {},
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock save_results to avoid serialization issues
#     with patch('calipers.scripts.run_evaluation.save_results'):
#         with pytest.raises(ValueError) as exc_info:
#             await run_evaluation(str(config_file))

#         assert 'Config cannot specify both category_filters and tasks' in str(
#             exc_info.value
#         )


# @pytest.mark.asyncio
# async def test_no_tasks_selected(tmp_path, test_workspace, register_mocks):
#     """Test handling of config with no tasks selected"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock save_results to avoid serialization issues
#     with patch('calipers.scripts.run_evaluation.save_results'):
#         results = await run_evaluation(str(config_file))
#         assert len(results) == 0


# @pytest.mark.asyncio
# async def test_task_package_import_logging(tmp_path, test_workspace, register_mocks):
#     """Test logging of task package imports"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'task_packages': ['test.package'],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock imports, save_results, and logger
#     with (
#         patch('builtins.__import__') as mock_import,
#         patch('calipers.scripts.run_evaluation.save_results'),
#         patch('calipers.scripts.run_evaluation.run_logger') as mock_logger,
#     ):

#         def mock_import_effect(name, *args, **kwargs):
#             if name == 'test.package':
#                 raise ImportError('Test error')
#             return MagicMock()

#         mock_import.side_effect = mock_import_effect

#         # Run evaluation with explicit commit hash
#         await run_evaluation(
#             str(config_file),
#             commit_hash='test123',  # Pass explicit hash instead of mocking git
#         )

#         # Verify logging
#         mock_logger.info.assert_any_call('Importing task package: test.package')
#         mock_logger.warning.assert_any_call(
#             'Failed to import task package test.package: Test error'
#         )


# @pytest.mark.asyncio
# async def test_task_package_config_parsing(tmp_path, test_workspace, register_mocks):
#     """Test task package config is properly parsed from YAML"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {'id': 'test_agent'},
#         'task_packages': ['package1', 'package2'],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock imports, save_results, and git
#     with (
#         patch('builtins.__import__'),
#         patch('calipers.scripts.run_evaluation.save_results'),
#     ):
#         # Load config and verify task_packages field
#         with open(config_file) as f:
#             loaded_config = yaml.safe_load(f)

#         assert 'task_packages' in loaded_config
#         assert loaded_config['task_packages'] == ['package1', 'package2']


# @pytest.mark.asyncio
# async def test_task_package_import(tmp_path, test_workspace, register_mocks):
#     """Test importing task packages from config"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'task_packages': [
#             'ml_dev_bench.tasks.array_generation',
#             'nonexistent.package',
#         ],
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'config': {},
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock imports and save_results
#     with (
#         patch('builtins.__import__') as mock_import,
#         patch('calipers.scripts.run_evaluation.save_results'),
#     ):

#         def mock_import_effect(name, *args, **kwargs):
#             if name == 'nonexistent.package':
#                 raise ImportError('Package not found')
#             return MagicMock()

#         mock_import.side_effect = mock_import_effect

#         # Run evaluation with explicit commit hash
#         results = await run_evaluation(
#             str(config_file),
#             commit_hash='test123',  # Pass explicit hash instead of mocking git
#         )

#         # Verify imports were attempted
#         mock_import.assert_any_call('ml_dev_bench.tasks.array_generation')
#         mock_import.assert_any_call('nonexistent.package')

#         # Verify evaluation still runs despite failed import
#         assert len(results) == 1
#         assert results[0].task_id == 'task1'


# @pytest.mark.asyncio
# async def test_relative_path_conversion(tmp_path, test_workspace, register_mocks):
#     """Test conversion of relative paths to absolute paths for output_dir and workspace_dir"""
#     # Create a config with relative paths
#     config = {
#         'workspace_dir': 'relative/workspace',
#         'output_dir': 'relative/output',
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'config': {},
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock imports and save_results
#     with patch('calipers.scripts.run_evaluation.save_results') as mock_save:
#         # Get the root directory for comparison

#         eval_parent_dir = Path(calipers.__path__[0]).parent

#         # Run evaluation
#         await run_evaluation(
#             config_path=str(config_file),
#             output_dir='relative/output',
#         )

#         # Verify workspace_dir was converted to absolute path
#         expected_workspace = os.path.abspath(
#             os.path.join(eval_parent_dir, 'relative/workspace')
#         )
#         mock_save.assert_called_once()
#         saved_config = mock_save.call_args[0][2]
#         assert str(saved_config['workspace_dir']) == expected_workspace

#         # Reset mock for next test
#         mock_save.reset_mock()

#         # Test with output_dir specified in function call
#         await run_evaluation(
#             config_path=str(config_file),
#             output_dir='another/relative/path',
#         )

#         # Verify output_dir was converted to absolute path
#         expected_output = os.path.abspath(
#             os.path.join(eval_parent_dir, 'another/relative/path')
#         )
#         assert str(mock_save.call_args[0][1]) == expected_output


# @pytest.mark.asyncio
# async def test_absolute_paths_unchanged(tmp_path, test_workspace, register_mocks):
#     """Test that absolute paths are not modified"""
#     # Create absolute paths
#     abs_workspace = os.path.abspath('/absolute/workspace')
#     abs_output = os.path.abspath('/absolute/output')

#     config = {
#         'workspace_dir': abs_workspace,
#         'output_dir': abs_output,
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'tasks': [
#             {
#                 'id': 'task1',
#                 'config': {},
#             }
#         ],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock save_results
#     with patch('calipers.scripts.run_evaluation.save_results') as mock_save:
#         # Run evaluation
#         await run_evaluation(
#             config_path=str(config_file),
#             output_dir=abs_output,
#         )

#         # Verify paths remained absolute and unchanged
#         saved_config = mock_save.call_args[0][2]
#         assert saved_config['workspace_dir'] == abs_workspace
#         assert mock_save.call_args[0][1] == abs_output


# @pytest.mark.asyncio
# async def test_main_with_litellm_metrics(tmp_path, test_workspace, register_mocks):
#     """Test main function with LiteLLM metrics collection"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'agent': {
#             'id': 'test_agent',
#             'config': {'uses_litellm': True},
#         },
#         'tasks': [{'id': 'task1'}],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock command line arguments
#     with patch('argparse.ArgumentParser.parse_args') as mock_args:
#         args = MagicMock()
#         args.config = str(config_file)
#         args.categories = None
#         args.tasks = ['task1']
#         args.output_dir = str(tmp_path)
#         args.commit_hash = 'test123'
#         args.langchain_project = None
#         mock_args.return_value = args

#         # Create a task instance that will be used throughout the test
#         task = MockTask(TaskConfig(id='task1', workspace_dir=test_workspace))

#         # Clear any existing callbacks
#         litellm.callbacks = []

#         with (
#             patch.object(EvalRegistry, 'get_task', return_value=MockTask),
#             patch.object(MockTask, '__new__', return_value=task),
#         ):
#             # Run main
#             await main()

#             # Verify results file was created
#             result_files = list(tmp_path.glob('eval_results_*.json'))
#             assert len(result_files) == 1

#             # Load and verify results
#             with open(result_files[0]) as f:
#                 data = json.load(f)

#             # Check metadata
#             assert 'metadata' in data
#             assert data['metadata']['commit_hash'] == 'test123'
#             assert data['metadata']['config'] == config

#             # Check results
#             assert 'results' in data
#             results = data['results']
#             assert len(results) == 1

#             # Verify task details
#             task_result = results[0]
#             assert task_result['task_id'] == 'task1'
#             assert task_result['agent_id'] == 'test_agent'
#             assert len(task_result['runs']) == 1

#             # Verify metrics were captured
#             run_metrics = task_result['runs'][0]['metrics']
#             assert run_metrics['runtime']['value'] == pytest.approx(
#                 0.5
#             )  # From agent's mock duration
#             assert run_metrics['token_cost']['value'] == 0.001  # From agent's mock cost
#             assert run_metrics['steps']['value'] == 1  # One callback call
#             assert run_metrics['token__completion_tokens']['value'] == 50
#             assert run_metrics['token__prompt_tokens']['value'] == 25
#             assert run_metrics['token__total_tokens']['value'] == 75
#             assert (
#                 run_metrics['tokens']['value'] == 100
#             )  # From MockTask's validate method


# @pytest.mark.asyncio
# async def test_workspace_cloning(tmp_path, test_workspace, register_mocks):
#     """Test workspace cloning functionality"""
#     # Create some test files in the workspace
#     test_file = test_workspace / 'test.txt'
#     test_file.write_text('original content')

#     config = {
#         'workspace_dir': str(test_workspace),
#         'clone_workspace': True,
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'tasks': [{'id': 'task1', 'config': {}}],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Run evaluation with mocked framework to return results
#     mock_result = EvaluationResult(
#         task_id='task1',
#         agent_id='test_agent',
#         categories=set(),
#         runs=[
#             RunResult(
#                 success=True,
#                 metrics={},
#                 validation_details={},
#                 start_time=datetime.now(),
#                 end_time=datetime.now(),
#             )
#         ],
#     )

#     temp_dir = '/tmp/workspace_clone_test'

#     with (
#         patch('tempfile.mkdtemp', return_value=temp_dir),
#         patch('shutil.copytree') as mock_copytree,
#         patch('shutil.rmtree') as mock_rmtree,
#         patch('calipers.scripts.run_evaluation.save_results'),
#         patch.object(EvaluationFramework, 'evaluate', return_value=mock_result),
#     ):
#         results = await run_evaluation(str(config_file))

#         # Verify original workspace is unchanged
#         assert test_file.read_text() == 'original content'

#         # Verify copytree was called with correct arguments
#         mock_copytree.assert_called_once_with(
#             str(test_workspace), temp_dir, dirs_exist_ok=True
#         )

#         # Verify rmtree was called for cleanup
#         mock_rmtree.assert_called_once_with(temp_dir, ignore_errors=True)

#         # Verify we got results
#         assert len(results) == 1
#         assert results[0].task_id == 'task1'


# @pytest.mark.asyncio
# async def test_workspace_clone_cleanup(tmp_path, test_workspace, register_mocks):
#     """Test that cloned workspace is properly cleaned up"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'clone_workspace': True,
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'tasks': [{'id': 'task1', 'config': {}}],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     temp_dirs = []
#     temp_dir = tempfile.mkdtemp(prefix='workspace_clone_')
#     temp_dirs.append(temp_dir)

#     # Mock framework to avoid recursion
#     mock_result = EvaluationResult(
#         task_id='task1',
#         agent_id='test_agent',
#         categories=set(),
#         runs=[
#             RunResult(
#                 success=True,
#                 metrics={},
#                 validation_details={},
#                 start_time=datetime.now(),
#                 end_time=datetime.now(),
#             )
#         ],
#     )

#     # Run evaluation with mocked framework
#     with (
#         patch('tempfile.mkdtemp', return_value=temp_dir),
#         patch('calipers.scripts.run_evaluation.save_results'),
#         patch.object(EvaluationFramework, 'evaluate', return_value=mock_result),
#     ):
#         await run_evaluation(str(config_file))

#         # Verify temp directory was created and then cleaned up
#         assert not os.path.exists(temp_dir)


# @pytest.mark.asyncio
# async def test_workspace_clone_error_handling(tmp_path, test_workspace, register_mocks):
#     """Test error handling during workspace cloning"""
#     config = {
#         'workspace_dir': str(test_workspace),
#         'clone_workspace': True,
#         'agent': {
#             'id': 'test_agent',
#             'config': {},
#         },
#         'tasks': [{'id': 'task1', 'config': {}}],
#     }

#     config_file = tmp_path / 'test_config.yaml'
#     with open(config_file, 'w') as f:
#         yaml.dump(config, f)

#     # Mock copytree to raise an error
#     with (
#         patch('shutil.copytree', side_effect=OSError('Mock copy error')),
#         patch('calipers.scripts.run_evaluation.save_results'),
#         patch('calipers.scripts.run_evaluation.run_logger'),
#     ):
#         with pytest.raises(OSError, match='Mock copy error'):
#             await run_evaluation(str(config_file))
