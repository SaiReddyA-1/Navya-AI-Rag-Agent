import os
import logging
from logging.handlers import RotatingFileHandler
from core.config import settings

def setup_enterprise_logger(name: str) -> logging.Logger:
    """
    Creates a standardized enterprise logger that outputs to both the console 
    and a rotating file to ensure zero data loss during massive batch ingestions.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # 1. Pull LOG_LEVEL from core.config
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 2. Set the Enterprise Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. Add Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 4. Add RotatingFileHandler (Max 5MB per file, keep last 3 backups)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "enterprise_rag.log")
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5*1024*1024, # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
