from calipers.metrics import CounterMetric, MetricsRegistry


@MetricsRegistry.register
class WandBFileMetric(CounterMetric):
    name = 'wandb_files'
    description = 'Number of correctly created WandB files'
    unit = 'files'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)
