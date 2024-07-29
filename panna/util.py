import os
import random
import gc
import logging
from typing import List, Optional

import numpy as np
from PIL.Image import Image
from torch import Generator
from torch import cuda


def get_generator(seed: Optional[int] = None) -> Generator:
    if seed:
        return Generator().manual_seed(seed)
    max_seed = np.iinfo(np.int32).max
    return Generator().manual_seed(random.randint(0, max_seed))


def clear_cache() -> None:
    gc.collect()
    cuda.empty_cache()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger
