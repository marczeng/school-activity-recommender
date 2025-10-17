import sys
print("Python:", sys.version)

import numpy as np
import pandas as pd

from sklearn import __version__ as skver
print("NumPy:", np.__version__, "| Pandas:", pd.__version__, "| scikit-learn:", skver)


