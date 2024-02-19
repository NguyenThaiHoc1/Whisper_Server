import logging
import sys

# get formatters
from app.logging.log_formatter import file_formatter, system_formatter

# get logger
logger = logging.getLogger("Personal-Notebook")

# create handlers
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('app.log')

# set formatters
stream_handler.setFormatter(
    system_formatter
)

file_handler.setFormatter(
    file_formatter
)

# add handlers to the logger
logger.handlers = [
    stream_handler,
    file_handler
]  # or addhandlers

# set log-level
logger.setLevel(logging.INFO)
