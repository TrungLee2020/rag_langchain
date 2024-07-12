import logging
import logging.handlers
import os

# Đường dẫn đến file log
LOG_DIR = "logs"
LOG_FILE = "logs.txt"

# Tạo thư mục log
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.DEBUG,  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Định dạng log
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE)),  # Ghi log vào file
        logging.StreamHandler()  # Hiển thị log ra console
            ]
                )
# logger
def get_logger(name):
    return logging.getLogger(name)
