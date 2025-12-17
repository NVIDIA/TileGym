# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unified logging management system for TileGym project
Provides deduplicated warnings, hierarchical logging, performance monitoring and other features
"""

import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set


class TileGymLogFormatter(logging.Formatter):
    """Custom formatter that handles caller location information and colors"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, *args, use_colors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and hasattr(os.sys.stderr, "isatty") and os.sys.stderr.isatty()

    def format(self, record):
        # Add location information
        if hasattr(record, "caller_filename") and hasattr(record, "caller_lineno"):
            record.location = f"{record.caller_filename}:{record.caller_lineno}"
        else:
            # Fallback to default location info
            record.location = f"{os.path.basename(record.pathname)}:{record.lineno}"

        # Format the message
        formatted = super().format(record)

        # Add colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS["RESET"]
            # Color the entire message
            formatted = f"{color}{formatted}{reset}"

        return formatted


def _get_caller_info(skip_frames: int = 2) -> Dict[str, Any]:
    """Get caller information for logging"""
    frame = inspect.currentframe()
    try:
        # Skip the specified number of frames to get to the actual caller
        for _ in range(skip_frames):
            if frame:
                frame = frame.f_back

        if frame:
            return {
                "caller_filename": os.path.basename(frame.f_code.co_filename),
                "caller_lineno": frame.f_lineno,
                "caller_funcname": frame.f_code.co_name,
            }
        return {}
    finally:
        del frame


def _get_log_level_from_env() -> int:
    """Get log level from environment variable TILEGYM_LOG_LEVEL"""
    env_level = os.getenv("TILEGYM_LOG_LEVEL", "INFO").upper()

    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # Alternative spelling
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # Alternative spelling
    }

    if env_level in level_map:
        return level_map[env_level]
    else:
        # If invalid level specified, warn and use INFO as default
        print(
            f"Warning: Invalid TILEGYM_LOG_LEVEL '{env_level}'. Valid levels are: {', '.join(level_map.keys())}. Using INFO as default."
        )
        return logging.INFO


class TileGymLogger:
    """Unified logger manager for TileGym project"""

    def __init__(self, name: str = "tilegym"):
        self.logger = logging.getLogger(name)
        self._warned_messages: Set[str] = set()
        self._message_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Set default format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Use custom formatter that handles caller info
            formatter = TileGymLogFormatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] [%(location)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # Prevent propagation to avoid duplicate messages
            self.logger.propagate = False

            # Set log level from environment variable or default to INFO
            self.logger.setLevel(_get_log_level_from_env())

    def warn_once(self, message: str, category: Optional[str] = None, _auto_caller_info: bool = True, **kwargs):
        """Warning method that only warns once"""
        key = f"{category}:{message}" if category else message

        with self._lock:
            if key not in self._warned_messages:
                self._warned_messages.add(key)
                if category:
                    formatted_message = f"[{category}] {message}"
                else:
                    formatted_message = message

                # Get caller information for better location tracking
                if _auto_caller_info and "extra" not in kwargs:
                    caller_info = _get_caller_info(skip_frames=2)
                    if caller_info:
                        kwargs.setdefault("extra", {}).update(caller_info)

                self.logger.warning(formatted_message, **kwargs)

    def warn_limited(
        self,
        message: str,
        max_count: int = 5,
        category: Optional[str] = None,
        _auto_caller_info: bool = True,
        **kwargs,
    ):
        """Warning method with limited count"""
        key = f"{category}:{message}" if category else message

        with self._lock:
            self._message_counts[key] += 1
            count = self._message_counts[key]

            if count <= max_count:
                if category:
                    formatted_message = f"[{category}] {message}"
                else:
                    formatted_message = message

                if count == max_count:
                    formatted_message += " (This warning will not be shown again)"
                elif count > 1:
                    formatted_message += f" (shown {count}/{max_count} times)"

                # Get caller information for better location tracking
                if _auto_caller_info and "extra" not in kwargs:
                    caller_info = _get_caller_info(skip_frames=2)
                    if caller_info:
                        kwargs.setdefault("extra", {}).update(caller_info)

                self.logger.warning(formatted_message, **kwargs)

    def info(self, *args, **kwargs):
        """Standard info log - supports multiple arguments like print()"""
        if len(args) == 0:
            message = ""
        elif len(args) == 1:
            message = str(args[0])
        else:
            # Join multiple arguments with space, like print()
            message = " ".join(str(arg) for arg in args)
        self.logger.info(message, **kwargs)

    def debug(self, *args, **kwargs):
        """Standard debug log - supports multiple arguments like print()"""
        if len(args) == 0:
            message = ""
        elif len(args) == 1:
            message = str(args[0])
        else:
            # Join multiple arguments with space, like print()
            message = " ".join(str(arg) for arg in args)
        self.logger.debug(message, **kwargs)

    def error(self, *args, **kwargs):
        """Standard error log - supports multiple arguments like print()"""
        if len(args) == 0:
            message = ""
        elif len(args) == 1:
            message = str(args[0])
        else:
            # Join multiple arguments with space, like print()
            message = " ".join(str(arg) for arg in args)
        self.logger.error(message, **kwargs)

    def warning(self, *args, **kwargs):
        """Standard warning log - supports multiple arguments like print()"""
        if len(args) == 0:
            message = ""
        elif len(args) == 1:
            message = str(args[0])
        else:
            # Join multiple arguments with space, like print()
            message = " ".join(str(arg) for arg in args)
        self.logger.warning(message, **kwargs)

    def get_warning_stats(self) -> Dict[str, int]:
        """Get warning statistics"""
        with self._lock:
            return dict(self._message_counts)

    def reset_warning_cache(self):
        """Reset warning cache"""
        with self._lock:
            self._warned_messages.clear()
            self._message_counts.clear()


# Global logger instance
_global_logger = TileGymLogger()


def get_logger(name: Optional[str] = None) -> TileGymLogger:
    """Get logger instance"""
    if name is None:
        return _global_logger
    else:
        return TileGymLogger(name)


# Convenience functions
def warn_once(message: str, category: Optional[str] = None, **kwargs):
    """Global warn_once function"""
    # Add extra skip frame since we're going through this wrapper
    caller_info = _get_caller_info(skip_frames=2)
    if caller_info:
        kwargs.setdefault("extra", {}).update(caller_info)
    _global_logger.warn_once(message, category, _auto_caller_info=False, **kwargs)


def warn_limited(message: str, max_count: int = 5, category: Optional[str] = None, **kwargs):
    """Global warn_limited function"""
    # Add extra skip frame since we're going through this wrapper
    caller_info = _get_caller_info(skip_frames=2)
    if caller_info:
        kwargs.setdefault("extra", {}).update(caller_info)
    _global_logger.warn_limited(message, max_count, category, _auto_caller_info=False, **kwargs)


def info(*args, **kwargs):
    """Global info function - supports multiple arguments like print()"""
    _global_logger.info(*args, **kwargs)


def debug(*args, **kwargs):
    """Global debug function - supports multiple arguments like print()"""
    _global_logger.debug(*args, **kwargs)


def error(*args, **kwargs):
    """Global error function - supports multiple arguments like print()"""
    _global_logger.error(*args, **kwargs)


def warning(*args, **kwargs):
    """Global warning function - supports multiple arguments like print()"""
    _global_logger.warning(*args, **kwargs)


# Decorator support
def log_function_call(level: str = "debug", include_args: bool = False):
    """Decorator for logging function calls"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            if include_args:
                message = f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            else:
                message = f"Calling {func.__name__}"

            getattr(logger, level)(message)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                getattr(logger, level)(f"{func.__name__} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
                raise

        return wrapper

    return decorator


def deprecated(message: str = "", category: str = "DEPRECATED"):
    """Decorator to mark function as deprecated"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            dep_message = message or f"{func.__name__} is deprecated and will be removed in a future version"
            warn_once(dep_message, category)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Context manager
class LogContext:
    """Log context manager for temporarily changing log level"""

    def __init__(self, logger: TileGymLogger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.logger.level
        self.logger.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.logger.setLevel(self.old_level)


def set_log_level(level: str):
    """Set global log level"""
    _global_logger.logger.setLevel(getattr(logging, level.upper()))


def get_current_log_level() -> str:
    """Get current log level as string"""
    level_num = _global_logger.logger.level
    level_map = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    return level_map.get(level_num, "UNKNOWN")


def reload_log_level_from_env():
    """Reload log level from environment variable"""
    new_level = _get_log_level_from_env()
    _global_logger.logger.setLevel(new_level)
    return get_current_log_level()


def get_env_log_level() -> str:
    """Get the log level specified in environment variable"""
    return os.getenv("TILEGYM_LOG_LEVEL", "INFO").upper()


def set_env_log_level(level: str):
    """Set environment variable and update current log level"""
    os.environ["TILEGYM_LOG_LEVEL"] = level.upper()
    return reload_log_level_from_env()


def get_warning_stats() -> Dict[str, int]:
    """Get global warning statistics"""
    return _global_logger.get_warning_stats()


def reset_warning_cache():
    """Reset global warning cache"""
    _global_logger.reset_warning_cache()
