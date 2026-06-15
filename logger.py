import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"


class DailyRotatingFileHandler(logging.Handler):
    """每天自动切换日志文件，文件名格式：YYYY-MM-DD.log"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self._log_dir = os.path.abspath(LOG_DIR)
        os.makedirs(self._log_dir, exist_ok=True)
        self._current_date = None
        self._file_handler = None

    def _ensure_handler(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            log_path = os.path.join(self._log_dir, f"{today}.log")
            self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
            if self.formatter:
                self._file_handler.setFormatter(self.formatter)
        return self._file_handler

    def emit(self, record):
        handler = self._ensure_handler()
        handler.emit(record)

    def close(self):
        if self._file_handler:
            self._file_handler.close()
        super().close()


def get_logger(name: str) -> logging.Logger:
    """获取一个按天轮转的日志器，日志文件保存在 logs/YYYY-MM-DD.log"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # 文件日志 - 按天生成独立文件
    file_handler = DailyRotatingFileHandler()
    file_handler.setLevel(logging.DEBUG)

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 统一格式
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
