import logging
import logging.config
import os
import yaml

from poe_api.config import CONFIG

server_log_filename = None
console_log_level = None


def get_log_config():
    global server_log_filename, console_log_level
    with open("logging_config", "r") as f:
        log_config = yaml.safe_load(f.read())
    log_config["handlers"]["file_handler"]["filename"] = server_log_filename
    log_config["handlers"]["console_handler"]["level"] = console_log_level
    return log_config


def setup_logger():
    global server_log_filename, console_log_level

    log_dir = CONFIG.log_dir

    os.makedirs(log_dir, exist_ok=True)
    server_log_filename = os.path.join(log_dir, f"latest.log")
    console_log_level = CONFIG.log_level

    log_config = get_log_config()
    logging.config.dictConfig(log_config)
