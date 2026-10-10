import os
import logging
from source.utils import get_metrics


class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        "INFO": "\033[32m",  # Green
        "ERROR": "\033[31m",  # Red
        "WARNING": "\033[33m",  # Yellow
        "DEBUG": "\033[34m",  # Blue
        "CRITICAL": "\033[41m",  # Background Red
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        log_message = super().format(record)

        # Apply color only to the log level name
        log_color = self.COLOR_CODES.get(record.levelname, self.RESET)
        reset_color = self.RESET
        colored_levelname = f"{log_color}{record.levelname}{reset_color}"

        # Replace the log level name in the message with the colored one
        log_message = log_message.replace(record.levelname, colored_levelname)
        return log_message


class PCBLogger:
    def __init__(
        self, console_output=True, logging_path="/home/osama/PCB_Inspection/App/logs"
    ):
        self.main_logger = logging.getLogger("Main_Logs")
        self.main_logger.setLevel(logging.INFO)
        self.main_logger.propagate = False

        self.exception_logger = logging.getLogger("Exception_Logs")
        self.exception_logger.setLevel(logging.ERROR)
        self.exception_logger.propagate = False

        self.time_logger = logging.getLogger("Timing_Logs")
        self.time_logger.setLevel(logging.INFO)
        self.time_logger.propagate = False

        self.input_logger = logging.getLogger("Input_Logs")
        self.input_logger.setLevel(logging.INFO)
        self.input_logger.propagate = False

        self.color_formatter = ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        os.makedirs(logging_path, exist_ok=True)

        # Set initial log file paths
        self.main_log_path = f"/home/osama/PCB_Inspection/App/logs/main.log"
        self.exception_log_path = f"/home/osama/PCB_Inspection/App/logs/exception.log"
        self.timing_log_path = f"/home/osama/PCB_Inspection/App/logs/timings.log"
        self.input_log_path = f"/home/osama/PCB_Inspection/App/logs/inputs.log"

        # Add file handlers
        self._add_file_handler(self.main_logger, self.main_log_path)
        self._add_file_handler(self.exception_logger, self.exception_log_path)
        self._add_file_handler(self.time_logger, self.timing_log_path)
        self._add_file_handler(self.input_logger, self.input_log_path)

        if console_output:
            self._add_console_handler()

    def _add_file_handler(
        self,
        logger,
        file_path,
    ):
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    def _add_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.color_formatter)
        self.main_logger.addHandler(console_handler)
        self.exception_logger.addHandler(
            console_handler
        )  # Add console handler to exception logger as well

    def update_log_paths(
        self,
        new_panel_id,
        new_pcb_id,
        base_dir="/home/osama/PCB_Inspection/App/results",
        console_output=True,
    ):
        # Remove existing file handlers

        # Update panel and PCB IDs and file paths
        self.panel_id = new_panel_id
        self.pcb_id = new_pcb_id
        self.base_dir = base_dir

        # Set up the directory structure
        self.panel_dir = os.path.join(self.base_dir, f"Panel_{self.panel_id}")
        self.pcb_dir = os.path.join(self.panel_dir, f"PCB_{self.pcb_id}")
        self.defect_dir = os.path.join(self.pcb_dir, "defects")

        os.makedirs(self.panel_dir, exist_ok=True)
        os.makedirs(self.pcb_dir, exist_ok=True)
        os.makedirs(self.defect_dir, exist_ok=True)

        # Define log files for main and exception logs
        self.main_log_path = os.path.join(self.pcb_dir, f"PCB_{self.pcb_id}_main.log")
        self.exception_log_path = os.path.join(
            self.pcb_dir, f"PCB_{self.pcb_id}_exception.log"
        )
        self.timing_log_path = os.path.join(self.pcb_dir, f"PCB_{self.pcb_id}_time.log")

        for logger in [
            self.main_logger,
            self.exception_logger,
            self.time_logger,
            self.input_logger,
        ]:
            logger.handlers = []  # Clear existing handlers
            if logger == self.main_logger:
                self._add_file_handler(logger, self.main_log_path)
            elif logger == self.exception_logger:
                self._add_file_handler(logger, self.exception_log_path)
            elif (
                logger == self.time_logger
            ):  # Ensure timing_logger is defined elsewhere in your class
                self._add_file_handler(logger, self.timing_log_path)
            elif (
                logger == self.input_logger
            ):  # Ensure timing_logger is defined elsewhere in your class
                self._add_file_handler(logger, self.input_log_path)

            # self._add_file_handler(logger, self.main_log_path if logger == self.main_logger else self.exception_log_path)

        if console_output:
            self._add_console_handler()

    def log_main(self, message, type="info"):
        if type == "info":
            self.main_logger.info(message)
        else:
            self.main_logger.error(message)

    def log_exception(self, message):
        self.exception_logger.error(message)

    def log_time(self, message):
        self.time_logger.info(message)

    def log_inputs(self, test_image):
        metrics = get_metrics(test_image)
        self.input_logger.info(metrics)
