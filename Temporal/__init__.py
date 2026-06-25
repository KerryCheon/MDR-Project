import sys

# Add src to sys.path so we can import pipeline
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pipeline

# Alias Temporal and Temporal.Pipeline in sys.modules to the pipeline package
sys.modules['Temporal'] = pipeline
sys.modules['Temporal.Pipeline'] = pipeline
sys.modules['pipeline.Pipeline'] = pipeline
