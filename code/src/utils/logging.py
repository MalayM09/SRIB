from __future__ import annotations

import logging
import sys


def get_logger(name: str = "srib", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.propagate = False
    return log
