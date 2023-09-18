# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Scheduler Classes"""

from functools import wraps
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, Callable

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.configs.base_config import InstantiateConfig


def delay(cls, lr_init: float) -> Callable:
    """wrapper for making scheduler enabled with delay"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> float:
            step = args[0]
            learning_factor = 0.0
            if cls.config.delay_steps != 0:
                if step <= cls.config.delay_steps:
                    learning_factor = cls.config.lr_pre_warmup / lr_init
                else:
                    compenastion = cls.config.delay_steps * (
                        1 - ((step - cls.config.delay_steps) / cls.config.max_steps)
                    )
                    learning_factor = func(step - compenastion, *args, **kwargs)
            else:
                learning_factor = func(step, *args, **kwargs)
            return learning_factor
        return wrapper
    return decorator


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config"""

    _target: Type = field(default_factory=lambda: Scheduler)
    """target class to instantiate"""


class Scheduler:
    """Base scheduler"""

    config: SchedulerConfig

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """


@dataclass
class MultiStepSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: MultiStepScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    gamma: float = 0.33
    """The learning rate decay factor."""
    milestones: Tuple[int, ...] = (500000, 750000, 900000)
    """The milestone steps at which to decay the learning rate."""


class MultiStepScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: MultiStepSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )
        return scheduler


@dataclass
class ExponentialDecaySchedulerConfig(SchedulerConfig):
    """Config for exponential decay scheduler with warmup"""

    _target: Type = field(default_factory=lambda: ExponentialDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""
    delay_steps: int = 0
    """Number of delay steps."""


class ExponentialDecayScheduler(Scheduler):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: ExponentialDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            self.config.lr_final = lr_init

        @delay(self, lr_init)
        def func(step, *args, **kwargs) -> float:
            lr = 0
            if step > self.config.max_steps:
                lr = self.config.lr_final
            elif self.config.warmup_steps > 0 and self.config.warmup_steps > step:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.config.lr_pre_warmup
                        + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                assert self.config.lr_final is not None
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(self.config.lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


@dataclass
class CosineDecaySchedulerConfig(SchedulerConfig):
    """Config for cosine decay schedule"""

    _target: Type = field(default_factory=lambda: CosineDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    learning_rate_alpha: float = 0.05
    """Learning rate alpha value"""
    max_steps: int = 300000
    """The maximum number of steps."""
    delay_steps: int = 0
    """Number of delay steps."""


class CosineDecayScheduler(Scheduler):
    """Cosine decay scheduler with linear warmup"""

    config: CosineDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:

        @delay(self, lr_init)
        def func(step: int):
            if self.config.warmup_steps > 0 and self.config.warmup_steps > step:
                learning_factor = (
                    self.config.lr_pre_warmup
                    + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                ) / lr_init
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
