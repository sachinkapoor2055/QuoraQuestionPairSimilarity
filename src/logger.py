import logging
import os
from datetime import datetime


# Define the directory where log files will be stored
log_directory = os.path.join(os.getcwd(), "logs")

# Ensure the log directory exists, creating it if it doesn't
os.makedirs(log_directory, exist_ok=True)

# Generate the log file name based on the current date
log_file_name = f"{datetime.now().strftime('%m_%d_%Y')}.log"

# Create the full path to the log file
log_file_path = os.path.join(log_directory, log_file_name)

# Configure logging with a file handler
logging.basicConfig(
    filename=log_file_path,  # Specify the file path for logging
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Specify the format for log messages
    level=logging.INFO,  # Set the logging level to INFO
)