import pytest

from calipers.metrics import (
    CostMetric,
    CounterMetric,
    DurationMetric,
    MetricsRegistry,
    MetricsTracker,
)


def test_counter_metric():
    """Test counter metric behavior"""
    counter = CounterMetric(name='test_counter', description='Test counter')

    # Test initial state
    assert counter.get_value() == 0

    # Test increment
    counter.update(1)
    assert counter.get_value() == 1

    # Test multiple increments
    counter.update(2)
    counter.update(3)
    assert counter.get_value() == 6

    # Test reset
    counter.reset()
    assert counter.get_value() == 0


def test_duration_metric():
    """Test duration metric behavior"""
    duration = DurationMetric(name='test_duration', description='Test duration')

    # Test initial state
    assert duration.get_value() == 0.0

    # Test update
    duration.update(1.5)
    assert duration.get_value() == 1.5

    # Test override
    duration.update(2.5)
    assert duration.get_value() == 2.5

    # Test reset
    duration.reset()
    assert duration.get_value() == 0.0


def test_cost_metric():
    """Test cost metric behavior"""
    cost = CostMetric(name='test_cost', description='Test cost')

    # Test initial state
    assert cost.get_value() == 0.0

    # Test accumulation
    cost.update(0.1)
    cost.update(0.2)
    assert cost.get_value() == pytest.approx(0.3)

    # Test reset
    cost.reset()
    assert cost.get_value() == 0.0


def test_metrics_tracker():
    """Test metrics tracker functionality"""
    tracker = MetricsTracker()

    # Add metrics
    counter = CounterMetric(name='test_counter', description='Test counter')
    duration = DurationMetric(name='test_duration', description='Test duration')
    tracker.add_metric(counter)
    tracker.add_metric(duration)

    # Update metrics
    tracker.update('test_counter', 1)
    tracker.update('test_duration', 2.5)

    # Get all metrics
    metrics = tracker.get_all()
    assert metrics['test_counter']['value'] == 1
    assert metrics['test_counter']['unit'] == 'count'
    assert metrics['test_duration']['value'] == 2.5
    assert metrics['test_duration']['unit'] == 'seconds'

    # Test reset
    tracker.reset()
    metrics = tracker.get_all()
    assert metrics['test_counter']['value'] == 0
    assert metrics['test_duration']['value'] == 0.0


def test_metrics_registry():
    """Test metrics registry functionality"""

    # Test registration
    @MetricsRegistry.register
    class TestMetric(CounterMetric):
        name = 'test_metric'
        description = 'Test metric'

    # Get registered metric
    metric_class = MetricsRegistry.get_metric('test_metric')
    assert metric_class is TestMetric

    # Test non-existent metric
    assert MetricsRegistry.get_metric('non_existent') is None
