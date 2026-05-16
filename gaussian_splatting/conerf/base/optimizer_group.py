from typing import List


class OptimizerGroup:
    def __init__(self,):
        self._optimizers = dict()
        self._schedulers = dict()

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def schedulers(self):
        return self._schedulers

    def add_optimizer(self, name, optimizer):
        self._optimizers[name] = optimizer

    def add_scheduler(self, name, scheduler):
        self._schedulers[name] = scheduler

    def step_and_zero_grad(self, set_to_none: bool = True, **kwargs):
        for name, optimizer in self._optimizers.items():
            if name.find('sparse') >= 0:
                optimizer.step(kwargs["visibility_mask"])
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=set_to_none)

    def schedule(self):
        for _, scheduler in self._schedulers.items():
            scheduler.step()
