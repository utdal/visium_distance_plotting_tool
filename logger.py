from datetime import datetime
import logging, logging.handlers, os

log_folder_path = os.path.join(os.getcwd(), 'Logs')
log_file = log_folder_path+"TRAP_analysis_log_"+str(datetime.now().strftime("%d-%m-%Y"))+".txt"
try:
    my_logger = logging.getLogger(__name__)
    my_logger.setLevel(getattr(logging, 'INFO'))
    handler = logging.handlers.RotatingFileHandler(log_file, mode='w', maxBytes=1048576, encoding=None)
    handler.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s : %(message)s"))
    my_logger.addHandler(handler)
except Exception as e:
    my_logger.exception(e)
