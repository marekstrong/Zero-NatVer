
import logging
import colorlog

# Default log level
global_log_level = logging.DEBUG

# Configure the color log handler and formatter
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
))

# Main logger configuration
logger = colorlog.getLogger("llm_logger")
logger.propagate = False
if not logger.hasHandlers():
    logger.addHandler(handler)
    logger.setLevel(global_log_level)

def get_logger(name=__name__):
    """Retrieve a child logger with the current global log level."""
    child_logger = logger.getChild(name)
    child_logger.setLevel(global_log_level)
    return child_logger

def set_global_log_level(level):
    """Set the global log level for all loggers."""
    global global_log_level
    global_log_level = level

    # Update the level on the main logger and its children
    logger.setLevel(global_log_level)

    # Update all existing handlers to use the new level
    for handler in logger.handlers:
        handler.setLevel(global_log_level)
    print(f"Global log level set to {logging.getLevelName(global_log_level)}")
