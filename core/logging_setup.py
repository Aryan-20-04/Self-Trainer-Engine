"""
core/logging_setup.py
=====================
One-call logging configuration for Self-Trainer.

Usage
-----
From main.py or any entry point, call this ONCE before doing anything else:

    from core.logging_setup import setup_logging
    setup_logging()                          # INFO to console
    setup_logging(level="DEBUG")             # verbose, show CV scores etc.
    setup_logging(log_file="trainer.log")    # also write to a file
    setup_logging(level="WARNING")           # silent except warnings/errors

Logger hierarchy
----------------
All Self-Trainer loggers sit under the "self_trainer" namespace:

    self_trainer                        ← root for the whole project
    self_trainer.engine                 ← SelfTrainerEngine
    self_trainer.core.config            ← EngineConfig
    self_trainer.core.evaluator         ← cross-validation
    self_trainer.monitoring.baseline    ← histogram baseline
    self_trainer.monitoring.drift_detector
    self_trainer.monitoring.experiment_tracker

This means:
  - logging.getLogger("self_trainer").setLevel(logging.DEBUG)
    turns on everything in one line.
  - logging.getLogger("self_trainer.monitoring.drift_detector").setLevel(logging.WARNING)
    silences drift debug output while keeping everything else verbose.

Level guide
-----------
DEBUG   — per-fold CV scores, PSI per feature, sample counts
INFO    — high-level pipeline steps, model scores, file paths
WARNING — drift detected, non-finite values converted to null, skipped columns
ERROR   — PSI calculation failures (column skipped but pipeline continues)
CRITICAL— (not used; unrecoverable failures raise exceptions instead)
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure the 'self_trainer' logger hierarchy.

    Parameters
    ----------
    level : str
        Minimum log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        Default: 'INFO'.
    log_file : str, optional
        If provided, log output is also written to this file path
        in addition to the console.
    fmt : str
        Log record format string.
    datefmt : str
        Date/time format for log records.

    Returns
    -------
    logging.Logger
        The configured root 'self_trainer' logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("self_trainer")
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if setup_logging() is called twice
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────────
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging to file: %s", log_file)

    root_logger.info("Self-Trainer logging configured (level=%s).", level.upper())
    return root_logger