# GEE Initialization Utility

import os
import json
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

    gee_project_id = os.environ.get("GEE_PROJECT_ID", "mdr-project-500504")
    if not gee_project_id:
        err_msg = "GEE_PROJECT_ID environment variable is not set. Please specify GEE_PROJECT_ID in your environment."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 1. Try standard initialization
    try:
        ee.Initialize(project=gee_project_id)
        logger.debug(f"Earth Engine initialized with project '{gee_project_id}'.")
        return
    except Exception as e_std:
        logger.debug(f"Standard ee.Initialize failed: {e_std}. Trying explicit credentials fallback...")

    # 2. Try loading persistent OAuth credentials from ~/.config/earthengine/credentials
    cred_file = os.path.expanduser("~/.config/earthengine/credentials")
    if os.path.exists(cred_file):
        try:
            from google.oauth2.credentials import Credentials

            with open(cred_file, "r", encoding="utf-8") as f:
                cred_data = json.load(f)

            creds = Credentials(
                None,
                refresh_token=cred_data["refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=cred_data["client_id"],
                client_secret=cred_data["client_secret"],
            )
            ee.Initialize(credentials=creds, project=gee_project_id)
            logger.info(f"Earth Engine initialized successfully using explicit credentials for project '{gee_project_id}'.")
            return
        except Exception as e_cred:
            logger.debug(f"Explicit credentials init failed: {e_cred}. Trying ee.Authenticate()...")

    # 3. Fallback to Authenticate
    try:
        ee.Authenticate()
        ee.Initialize(project=gee_project_id)
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine with project ID '{gee_project_id}': {e}")
        raise
