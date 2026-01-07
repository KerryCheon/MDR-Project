#!/usr/bin/env python3
# Jakob Balkovec & Kerry Cheon
# Main Pipeline Execution Script

"""
Standalone script to run the MDR Temporal Pipeline.
This can be executed directly from the command line.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run the main module
from Temporal.Pipeline import main

if __name__ == "__main__":
    # The main module's __main__ block will execute
    pass
