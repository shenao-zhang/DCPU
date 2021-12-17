"""Implementation of a Logger class."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorboardX

def safe_make_dir(dir_name: str) -> str: ...

class Logger(object):
    statistics: List[Dict[str, float]]  # statistic[i_episode] = Summary(i_episode)
    current: Dict[str, Tuple[int, float]]  # Dict[key, (count, value)]
    all: Dict[str, List[float]]
    writer: Optional[tensorboardX.SummaryWriter]
    episode: int
    keys: set
    log_dir: str
    def __init__(
        self, name: str, comment: str = ..., tensorboard: bool = ...
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Dict[str, float]]: ...
    def __getitem__(self, item: int) -> Dict[str, float]: ...
    def __str__(self) -> str: ...
    def get(self, key: str) -> List[float]: ...
    def update(self, **kwargs: Any) -> None: ...
    def end_episode(self, **kwargs: Any) -> None: ...
    def save_hparams(self, hparams: Dict) -> None: ...
    def export_to_json(self) -> None: ...
    def load_from_json(self, log_dir: Optional[str] = ...) -> None: ...
    def log_hparams(self, hparams: Dict, metrics: Optional[Dict] = ...) -> None: ...
    def delete_directory(self) -> None: ...
    def change_log_dir(self, new_log_dir: str) -> None: ...