import logging
import os
# Configure the logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger with the name of the current module or script
logger = logging.getLogger(__name__)

# Create a FileHandler and set the log file path
if not os.path.exists("logs/"):
    os.makedirs('logs/')
log_file_path = 'logs/my_log_file.log'
file_handler = logging.FileHandler(log_file_path)

# Create a formatter and set it for the FileHandler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)