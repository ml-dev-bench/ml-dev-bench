from datetime import datetime

import pytest

from calipers.callbacks.litellm_callbacks.metrics_callback import (
    MetricsCallbackHandler,
)
from calipers.metrics import (
    MetricsTracker,
    RuntimeMetric,
    StepsMetric,
    TokenCostMetric,
)


@pytest.fixture
def metrics_tracker():
    tracker = MetricsTracker()
    tracker.add_metric(RuntimeMetric())
    tracker.add_metric(TokenCostMetric())
    tracker.add_metric(StepsMetric())
    return tracker


@pytest.fixture
def handler(metrics_tracker):
    return MetricsCallbackHandler(metrics_tracker)


@pytest.fixture
def sample_usage_data():
    return {
        'standard_logging_object': {
            'response': {
                'usage': {
                    'completion_tokens': 100,
                    'prompt_tokens': 50,
                    'total_tokens': 150,
                    'prompt_tokens_details': {
                        'cached_tokens': 0,
                        'audio_tokens': None,
                    },
                }
            }
        },
        'response_cost': 0.05,
    }


def test_handler_initialization():
    """Test that handler is properly initialized with default metrics"""
    handler = MetricsCallbackHandler()

    # Test metrics property returns MetricsTracker
    assert isinstance(handler.metrics_tracker, MetricsTracker)

    # Test both access methods return same data
    metrics_via_property = handler.metrics_tracker.get_all()
    metrics_via_method = handler.get_accumulated_metrics()
    assert metrics_via_property == metrics_via_method

    # Verify required metrics exist
    assert RuntimeMetric.name in metrics_via_property
    assert TokenCostMetric.name in metrics_via_property
    assert StepsMetric.name in metrics_via_property


def test_metrics_property_returns_same_tracker(metrics_tracker):
    """Test that metrics property returns the same tracker instance"""
    handler = MetricsCallbackHandler(metrics_tracker)
    assert handler.metrics_tracker is metrics_tracker


def test_handler_calculate_duration(handler):
    """Test duration calculation"""
    start = datetime.now()
    end = datetime.now()

    # Valid duration
    duration = handler._calculate_duration(start, end)
    assert isinstance(duration, float)
    assert duration >= 0

    # None inputs
    assert handler._calculate_duration(None, end) is None
    assert handler._calculate_duration(start, None) is None
    assert handler._calculate_duration(None, None) is None

    # Invalid inputs
    assert handler._calculate_duration('invalid', end) is None
    assert handler._calculate_duration(start, 'invalid') is None


def test_handler_extract_token_metrics(handler, sample_usage_data):
    """Test token metrics extraction"""
    metrics = handler._extract_token_metrics(sample_usage_data)
    assert metrics is not None
    assert metrics['completion_tokens'] == 100
    assert metrics['prompt_tokens'] == 50
    assert metrics['total_tokens'] == 150
    assert metrics['prompt_tokens_details__cached_tokens'] == 0
    assert metrics['prompt_tokens_details__audio_tokens'] is None

    # Test with invalid data
    assert handler._extract_token_metrics({}) is None
    assert handler._extract_token_metrics(None) is None


@pytest.mark.asyncio
async def test_handler_success_events(handler, sample_usage_data):
    """Test both sync and async success events"""
    start_time = datetime.now()
    end_time = datetime.now()

    # Test sync success event
    handler.log_success_event(sample_usage_data, None, start_time, end_time)

    # Test async success event
    await handler.async_log_success_event(sample_usage_data, None, start_time, end_time)

    # Verify metrics were accumulated - test both access methods
    metrics_via_property = handler.metrics_tracker.get_all()
    metrics_via_method = handler.get_accumulated_metrics()
    assert metrics_via_property == metrics_via_method

    # Use metrics_via_property for assertions
    assert RuntimeMetric.name in metrics_via_property
    assert metrics_via_property[RuntimeMetric.name]['value'] > 0

    assert TokenCostMetric.name in metrics_via_property
    assert metrics_via_property[TokenCostMetric.name]['value'] == pytest.approx(
        0.10
    )  # 2 events * 0.05

    assert StepsMetric.name in metrics_via_property
    assert metrics_via_property[StepsMetric.name]['value'] == 2  # 2 events

    token_metric_name = 'token__completion_tokens'
    assert token_metric_name in metrics_via_property
    assert (
        metrics_via_property[token_metric_name]['value'] == 200
    )  # 2 events * 100 tokens


@pytest.mark.asyncio
async def test_handler_failure_events(handler, sample_usage_data):
    """Test both sync and async failure events"""
    start_time = datetime.now()
    end_time = datetime.now()

    # Test sync failure event
    handler.log_failure_event(sample_usage_data, None, start_time, end_time)

    # Test async failure event
    await handler.async_log_failure_event(sample_usage_data, None, start_time, end_time)

    # Verify metrics were accumulated - test both access methods
    metrics_via_property = handler.metrics_tracker.get_all()
    metrics_via_method = handler.get_accumulated_metrics()
    assert metrics_via_property == metrics_via_method

    # Use metrics_via_property for assertions
    assert RuntimeMetric.name in metrics_via_property
    assert metrics_via_property[RuntimeMetric.name]['value'] > 0

    assert TokenCostMetric.name in metrics_via_property
    assert metrics_via_property[TokenCostMetric.name]['value'] == pytest.approx(
        0.10
    )  # 2 events * 0.05

    assert StepsMetric.name in metrics_via_property
    assert metrics_via_property[StepsMetric.name]['value'] == 2  # 2 events

    token_metric_name = 'token__completion_tokens'
    assert token_metric_name in metrics_via_property
    assert (
        metrics_via_property[token_metric_name]['value'] == 200
    )  # 2 events * 100 tokens


def test_handler_reset(handler, sample_usage_data):
    """Test that handler reset clears all metrics"""
    start_time = datetime.now()
    end_time = datetime.now()

    # Add some test metrics
    handler.log_success_event(sample_usage_data, None, start_time, end_time)

    # Verify metrics were recorded - test both access methods
    metrics_via_property = handler.metrics_tracker.get_all()
    metrics_via_method = handler.get_accumulated_metrics()
    assert metrics_via_property == metrics_via_method

    assert metrics_via_property[RuntimeMetric.name]['value'] > 0
    assert metrics_via_property[TokenCostMetric.name]['value'] > 0
    assert metrics_via_property[StepsMetric.name]['value'] == 1  # 1 event
    assert metrics_via_property['token__completion_tokens']['value'] > 0

    # Reset the metrics
    handler.reset()

    # Verify all metrics were reset to initial values
    reset_metrics = handler.metrics_tracker.get_all()
    assert reset_metrics[RuntimeMetric.name]['value'] == 0.0
    assert reset_metrics[TokenCostMetric.name]['value'] == 0.0
    assert reset_metrics[StepsMetric.name]['value'] == 0
    assert reset_metrics['token__completion_tokens']['value'] == 0
