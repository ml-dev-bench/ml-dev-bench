from calipers.metrics import CounterMetric, MetricsRegistry


@MetricsRegistry.register
class LabelboxFileMetric(CounterMetric):
    name = 'labelbox_files'
    description = 'Number of correctly created Labelbox files'
    unit = 'files'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)
