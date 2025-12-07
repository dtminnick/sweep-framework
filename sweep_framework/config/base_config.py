
# sweep_framework/config/base_config.py
from abc import ABC, abstractmethod

class BaseConfig(ABC):
    """
    Abstract base class for model configurations.
    Defines interface for building models, optimizers, and schedulers.
    """

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_optimizer(self, parameters):
        pass

    @abstractmethod
    def build_scheduler(self, optimizer):
        pass
