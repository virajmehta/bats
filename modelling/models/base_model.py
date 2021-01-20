"""
Base class for all models.

Author: Ian Char
Date: 8/26/2020
"""
import abc
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from modelling.utils.torch_utils import torch_to, Standardizer


class BaseModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class for models."""

    def __init__(self, data_dimensions: Sequence[int]):
        super(BaseModel, self).__init__()
        self.standardizer = Standardizer([(torch.zeros(dd), torch.ones(dd))
                                           for dd in data_dimensions])
        self._model_is_fresh = False

    def set_standardization(
            self,
            standardizers:Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Set the mean and standard deviation for standardizing inputs.
        Args:
            standardizers: List of (mean, std) tuples corresponding to the
                sequence received in the batchces for forward.
        """
        for data_loc, pair in enumerate(standardizers):
            self.standardizer.set_stats(pair[0], pair[1], data_loc)

    def standardize_batch(
            self,
            batch: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        """Standardize a batch."""
        return [self.standardizer.standardize(b, idx)
                for idx, b in enumerate(batch)]

    def unstandardize_batch(
            self,
            batch: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        """Unstandardize a batch."""
        return [self.standardizer.unstandardize(b, idx)
                for idx, b in enumerate(batch)]

    def forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        batch = [torch_to(b) for b in batch]
        batch = self.standardize_batch(batch)
        return self.model_forward(batch)

    def save_model(self, save_path: str) -> None:
        """Save the model to the path."""
        save_dict = self.state_dict()
        torch.save(save_dict, save_path)

    def load_model(
            self,
            load_path: str,
            map_location: Optional[str]=None
    ) -> None:
        """Load the model from the path."""
        model_dict = torch.load(load_path, map_location=map_location)
        self.load_state_dict(model_dict)
        self._loaded = True

    def reset(self):
        self._model_is_fresh = False

    @abc.abstractmethod
    def loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss from the output of the network and targets.
        Returns the loss and additional stats.
        """
