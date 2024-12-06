import sys
import loguru
from loguru import logger
import streamlit as st


def configure_logging(log_fmt: str, log_lvl: str) -> loguru.logger: # type: ignore
    """
    Create logging handlers and set the logging level as defined in the config file
    """
    level = log_lvl
    logger.remove()  # Remove any existing handlers
    logger.add("debug.log", rotation='50 MB', format=log_fmt, retention='10 days', level=level)
    logger.add(sys.stdout, format=log_fmt, level=level)
    return logger


def set_initial_session_state(name: str, value: any): # type: ignore
    """
    Sets the initial state of a session variable -
    this is only done once per script execution.
    :param name: Name of the session variable
    :param value: Value for the session variable
    """
    if name not in st.session_state:
        st.session_state[name] = value
