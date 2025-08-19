import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Optional, Any, Callable, List
import sys
from datetime import datetime
import json


class LogLevel(Enum):
    """Enumeration of log levels with priorities."""
    DEBUG = (10, "DEBUG")
    INFO = (20, "INFO")
    WARNING = (30, "WARNING")
    ERROR = (40, "ERROR")
    CRITICAL = (50, "CRITICAL")
    SUCCESS = (25, "SUCCESS")
    PROGRESS = (15, "PROGRESS")
    
    def __init__(self, level: int, level_name: str):
        self.level = level
        self.level_name = level_name


@dataclass
class LogMessage:
    """Structured log message with metadata."""
    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    thread_id: Optional[int] = None
    module: Optional[str] = None
    function: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.thread_id is None:
            self.thread_id = threading.get_ident()


class LogFormatter(ABC):
    """Abstract base class for log formatters."""
    
    @abstractmethod
    def format(self, message: LogMessage) -> str:
        """Format a log message."""
        pass


class ConsoleFormatter(LogFormatter):
    """Console formatter with colors."""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.colors = {
            LogLevel.DEBUG: "\033[36m",      # Cyan
            LogLevel.INFO: "\033[37m",       # White
            LogLevel.WARNING: "\033[33m",    # Yellow
            LogLevel.ERROR: "\033[31m",      # Red
            LogLevel.CRITICAL: "\033[35m",   # Magenta
            LogLevel.SUCCESS: "\033[32m",    # Green
            LogLevel.PROGRESS: "\033[34m",   # Blue
        }
        self.reset = "\033[0m"
    
    def format(self, message: LogMessage) -> str:
        """Format message for console output."""
        timestamp = message.timestamp.strftime("%H:%M:%S")
        
        prefix_parts = [f"[{timestamp}]", f"[{message.level.level_name}]"]
        
        if message.module:
            prefix_parts.append(f"[{message.module}]")
        
        prefix = " ".join(prefix_parts)
        
        if self.use_colors and message.level in self.colors:
            prefix = f"{self.colors[message.level]}{prefix}{self.reset}"
            msg = f"{self.colors[message.level]}{message.message}{self.reset}"
        else:
            msg = message.message
        
        return f"{prefix} {msg}"


class FileFormatter(LogFormatter):
    """File formatter for structured logging."""
    
    def format(self, message: LogMessage) -> str:
        """Format message for file output."""
        timestamp = message.timestamp.isoformat()
        thread_info = f"[T{message.thread_id}]" if message.thread_id else ""
        module_info = f"[{message.module}]" if message.module else ""
        
        base_msg = f"{timestamp} {message.level.level_name}{thread_info}{module_info} {message.message}"
        
        if message.extra_data:
            extra = json.dumps(message.extra_data, default=str)
            base_msg += f" | {extra}"
        
        return base_msg


class LogHandler(ABC):
    """Abstract base class for log handlers."""
    
    def __init__(self, formatter: LogFormatter, min_level: LogLevel = LogLevel.INFO):
        self.formatter = formatter
        self.min_level = min_level
    
    @abstractmethod
    def emit(self, message: LogMessage) -> None:
        """Emit a log message."""
        pass
    
    def should_emit(self, message: LogMessage) -> bool:
        """Check if message should be emitted based on level."""
        return message.level.level >= self.min_level.level


class ConsoleHandler(LogHandler):
    """Handler for console output."""
    
    def __init__(self, formatter: Optional[LogFormatter] = None, 
                 min_level: LogLevel = LogLevel.INFO,
                 stream=None):
        if formatter is None:
            formatter = ConsoleFormatter()
        super().__init__(formatter, min_level)
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
    
    def emit(self, message: LogMessage) -> None:
        """Emit message to console."""
        if not self.should_emit(message):
            return
        
        formatted = self.formatter.format(message)
        with self._lock:
            print(formatted, file=self.stream, flush=True)


class FileHandler(LogHandler):
    """Handler for file output."""
    
    def __init__(self, filepath: Path, formatter: Optional[LogFormatter] = None,
                 min_level: LogLevel = LogLevel.DEBUG,
                 max_size_mb: int = 10):
        if formatter is None:
            formatter = FileFormatter()
        super().__init__(formatter, min_level)
        self.filepath = filepath
        self.max_size_mb = max_size_mb
        self._lock = threading.Lock()
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, message: LogMessage) -> None:
        """Emit message to file."""
        if not self.should_emit(message):
            return
        
        formatted = self.formatter.format(message)
        
        with self._lock:
            self._rotate_if_needed()
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
    
    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds size limit."""
        if not self.filepath.exists():
            return
        
        size_mb = self.filepath.stat().st_size / (1024 * 1024)
        if size_mb > self.max_size_mb:
            backup_path = self.filepath.with_suffix(f".{int(time.time())}.log")
            self.filepath.rename(backup_path)


class ProgressTracker:
    """Thread-safe progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.completed = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update(self, increment: int = 1) -> Dict[str, Any]:
        """Update progress and return stats."""
        with self._lock:
            self.completed += increment
            elapsed = time.time() - self.start_time
            percentage = (self.completed / self.total) * 100
            
            if self.completed > 0:
                eta = (elapsed / self.completed) * (self.total - self.completed)
            else:
                eta = 0
            
            return {
                'completed': self.completed,
                'total': self.total,
                'percentage': percentage,
                'elapsed': elapsed,
                'eta': eta,
                'rate': self.completed / elapsed if elapsed > 0 else 0
            }


def create_progress_logger(total: int, description: str = "Progress"):
    """Return a callable that logs progress updates using the global logger.

    Parameters
    ----------
    total : int
        Total number of steps.
    description : str
        Prefix label for progress messages.
    """
    tracker = ProgressTracker(total, description)

    def _update(increment: int = 1) -> None:
        stats = tracker.update(increment)
        get_logger().progress(
            f"{description}: {stats['completed']}/{stats['total']} "
            f"({stats['percentage']:.1f}%) ETA {stats['eta']:.1f}s"
        )

    return _update


class Logger:
    """Advanced thread-safe logger with multiple handlers and formatters."""
    
    def __init__(self, name: str = "BCILogger"):
        self.name = name
        self.handlers: List[LogHandler] = []
        self.message_queue: Queue = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._lock = threading.Lock()
        self.filters: List[Callable[[LogMessage], bool]] = []
        
        self.stats = {
            'messages_logged': 0,
            'messages_filtered': 0,
            'start_time': None
        }
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler."""
        with self._lock:
            self.handlers.append(handler)
    
    def add_filter(self, filter_func: Callable[[LogMessage], bool]) -> None:
        """Add a message filter function."""
        with self._lock:
            self.filters.append(filter_func)
    
    def start(self) -> None:
        """Start the logging worker thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the logging worker thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.message_queue.put(None)
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
    
    def _worker(self) -> None:
        """Worker thread for processing log messages."""
        while self.is_running:
            try:
                message = self.message_queue.get(timeout=0.1)
                if message is None:
                    break
                
                if self._should_filter(message):
                    self.stats['messages_filtered'] += 1
                    continue
                
                for handler in self.handlers:
                    try:
                        handler.emit(message)
                    except Exception as e:
                        print(f"Logger handler error: {e}", file=sys.stderr)
                
                self.stats['messages_logged'] += 1
                
            except Empty:
                continue
            except Exception as e:
                print(f"Logger worker error: {e}", file=sys.stderr)
    
    def _should_filter(self, message: LogMessage) -> bool:
        """Check if message should be filtered."""
        return any(filter_func(message) for filter_func in self.filters)
    
    def log(self, level: LogLevel, message: str, 
            module: Optional[str] = None,
            function: Optional[str] = None,
            **extra_data) -> None:
        """Log a message."""
        if not self.is_running:
            self.start()
        
        log_message = LogMessage(
            level=level,
            message=message,
            module=module,
            function=function,
            extra_data=extra_data
        )
        
        self.message_queue.put(log_message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message."""
        self.log(LogLevel.SUCCESS, message, **kwargs)
    
    def progress(self, message: str, **kwargs) -> None:
        """Log progress message."""
        self.log(LogLevel.PROGRESS, message, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        current_time = time.time()
        uptime = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            **self.stats,
            'uptime': uptime,
            'handlers_count': len(self.handlers),
            'queue_size': self.message_queue.qsize(),
            'is_running': self.is_running
        }


_global_logger: Optional[Logger] = None


def get_logger(name: str = "BCI") -> Logger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = Logger(name)
        
        console_handler = ConsoleHandler(
            formatter=ConsoleFormatter(use_colors=True),
            min_level=LogLevel.INFO
        )
        _global_logger.add_handler(console_handler)
    
    return _global_logger


def setup_file_logging(log_dir: Path = Path("logs"), 
                      filename: str = "bci.log",
                      min_level: LogLevel = LogLevel.DEBUG) -> None:
    """Setup file logging for the global logger."""
    logger = get_logger()
    
    file_handler = FileHandler(
        filepath=log_dir / filename,
        min_level=min_level
    )
    logger.add_handler(file_handler)


def setup_development_logging() -> None:
    """Setup logging configuration for development."""
    logger = get_logger()
    
    debug_handler = ConsoleHandler(
        formatter=ConsoleFormatter(use_colors=True),
        min_level=LogLevel.DEBUG
    )
    logger.add_handler(debug_handler)
    
    setup_file_logging()


def create_progress_logger(total: int, description: str = "Processing") -> Callable[[int], None]:
    """Create a progress logging function."""
    logger = get_logger()
    tracker = ProgressTracker(total, description)
    
    def log_progress(increment: int = 1) -> None:
        stats = tracker.update(increment)
        
        message = (f"{description}: {stats['completed']}/{stats['total']} "
                  f"({stats['percentage']:.1f}%) - "
                  f"ETA: {stats['eta']:.1f}s")
        
        logger.progress(message, **stats)
    
    return log_progress
