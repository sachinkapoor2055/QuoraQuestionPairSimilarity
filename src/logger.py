import logging
import os
from datetime import datetime


log_directory = os.path.join(os.getcwd(), "logs")

# Ensure the directory exists (including parent directories)
os.makedirs(log_directory, exist_ok=True)

log_file_name = f"{datetime.now().strftime('%m_%d_%Y')}.log"

log_file_path = os.path.join(log_directory, log_file_name)


logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)