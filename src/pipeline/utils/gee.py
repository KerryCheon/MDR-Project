# GEE Initialization Utility

import os
import ee
from .logger import get_logger


def initialize_ee(logger=None):
    """Initializes Google Earth Engine using the GEE_PROJECT_ID environment variable.

    Args:
        logger: Optional logger instance. If not provided, a default child logger is used.

    Raises:
        ValueError: If GEE_PROJECT_ID environment variable is not set.
        Exception: If initialization or authentication fails.
    """
    if logger is None:
        logger = get_logger().getChild("gee")

    gee_project_id = os.environ.get("GEE_PROJECT_ID")
    if not gee_project_id:
        err_msg = "GEE_PROJECT_ID environment variable is not set. Please specify GEE_PROJECT_ID in your environment."
        logger.error(err_msg)
        raise ValueError(err_msg)

    try:
        ee.Initialize(project=gee_project_id)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=gee_project_id)
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine with project ID '{gee_project_id}': {e}")
            raise
