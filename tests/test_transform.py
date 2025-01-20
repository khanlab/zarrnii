import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from test_io import cleandir, nifti_nib

from zarrnii import Transform, ZarrNii
