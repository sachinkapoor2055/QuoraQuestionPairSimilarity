import sys
from src.logger import logging 

def error_message_detail(error, error_detail: sys):
    if error_detail:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in '{file_name}' at line {line_number}: {type(error).__name__} - {str(error)}"
        return error_message
    else:
        return f"Error occurred: {type(error).__name__} - {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = None):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
