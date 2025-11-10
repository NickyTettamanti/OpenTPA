import sys, os
print('PYTHONPATH head:', sys.path[:3])

import pi_core, pi_core.ingest as ingest
print('ingest file:', ingest.__file__)
print('has standardize_columns?:', hasattr(ingest, 'standardize_columns'))
