import sys

# Register aliases in sys.modules to support legacy imports from the old project structure.
# This transparently maps 'pipeline.Pipeline' and 'Temporal.Pipeline' to the 'pipeline' package.
sys.modules['pipeline.Pipeline'] = sys.modules[__name__]
sys.modules['Temporal'] = sys.modules[__name__]
sys.modules['Temporal.Pipeline'] = sys.modules[__name__]
