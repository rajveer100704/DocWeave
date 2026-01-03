import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Define formatter
        formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

        # File handler with rotation (UTF-8 encoding for Unicode support)
        file_handler = RotatingFileHandler(
            log_file_path, 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with Unicode error handling
        class SafeStreamHandler(logging.StreamHandler):
            """StreamHandler that safely handles Unicode encoding errors on Windows."""
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    # Try to write normally first
                    try:
                        stream.write(msg + self.terminator)
                        self.flush()
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # If encoding fails, sanitize the message by replacing problematic characters
                        # This ensures Windows console (cp1252) can handle the output
                        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                        try:
                            stream.write(safe_msg + self.terminator)
                            self.flush()
                        except Exception:
                            # Last resort: write a simplified message
                            safe_msg_simple = f"[Log message contains non-ASCII characters: {record.levelname}]"
                            stream.write(safe_msg_simple + self.terminator)
                            self.flush()
                except Exception:
                    self.handleError(record)
        
        console_handler = SafeStreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# Configure the logger
configure_logger()

# Re-export the logging module so other modules can use: from src.logger import logging
# After configuration, the root logger is set up with file and console handlers
# Other modules can use logging.info(), logging.error(), etc. as normal