from pflotran_swi.norfolk_model import NorfolkModel
import pandas as pd 
import h5py 
import pint
import numpy as np  
import os
from pflotran_swi.units import ureg
import glob
import pytest
import pflotran_swi.pfwrite as pfw

def test_pfwrite(tmp_path):
    test_model = NorfolkModel()
    tmp_path_string = str(tmp_path)
    pfw.write(test_model, dir=tmp_path_string)