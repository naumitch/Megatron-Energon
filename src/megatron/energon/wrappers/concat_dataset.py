# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class ConcatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    def __init__(
        self,
        *datasets: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ):
        """Construct a concatenated dataset."""
        super().__init__(datasets, worker_config=worker_config)
        assert len(self) >= 0, "Datasets must be finite."

    def reset_state_own(self) -> None:
        return

    def len_worker(self, worker_idx: int | None = None) -> int:
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        for ds_idx, dataset in enumerate(self.datasets):
            for sample in dataset:
                yield add_sample_restore_key(
                    sample,
                    ds_idx,
                    src=self,
                )

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        # ConcatDataset always prepends (ClassName, ds_idx) in __iter__, even for a single wrapped
        # dataset, so always strip the prefix (the base len==1 fast-path would forward it unstripped).
        id, ds_idx = restore_key[:2]
        assert id == type(self).__name__
        assert isinstance(ds_idx, int)
        return add_sample_restore_key(
            self.datasets[ds_idx].restore_sample(restore_key[2:]),
            ds_idx,
            src=self,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
