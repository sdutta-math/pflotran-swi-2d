from pathlib import Path 

ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src" 
WORK = ROOT / "work"

import sys
sys.path.append(str(SRC))

from model.norfolk_model import NorfolkModel
import pandas as pd 
import h5py 
import pint
import numpy as np  
import os
from model.units import ureg
import glob
import pytest
import model.pfwrite as pfw

def test_pfwrite(tmp_path):
    test_model = NorfolkModel()
    tmp_path_string = str(tmp_path)
    pfw.write(test_model, dir=tmp_path_string)