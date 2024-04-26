import logging
import os
from contextlib import contextmanager
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_zero_only
from rich.logging import RichHandler


class CustomRichHandler(RichHandler):
    """_summary_

    Args:
        RichHandler (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        """_summary_

        Args:
            record (_type_): _description_
        """
        record.pathname, record.lineno, record.funcName = self.find_custom_caller(stacklevel=7)
        record.filename = os.path.basename(record.pathname)
        super().emit(record)

    def find_custom_caller(self, stacklevel=1):
        """Find the stack frame of the caller so that we can note the source file name, line number
        and function name."""
        f = logging.currentframe()
        # Go back stacklevel steps
        while stacklevel and f is not None:
            f = f.f_back
            stacklevel -= 1
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == logging._srcfile:
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno, co.co_name)
            break
        return rv


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
        handler = CustomRichHandler(rich_tracebacks=True),
        level = logging.DEBUG
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)

        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only
        
        self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            if current_rank is not None and not self.rank_zero_only:
                msg = f"[rank: {current_rank}] {msg}"
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)


@contextmanager
def suppress_logging(logger, level=logging.CRITICAL + 1):
    """_summary_

    Args:
        logger (_type_): _description_
        level (_type_, optional): _description_. Defaults to logging.CRITICAL+1.
    """
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)
