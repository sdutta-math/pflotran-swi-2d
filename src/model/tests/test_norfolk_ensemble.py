from pathlib import Path 

ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src" 
WORK = ROOT / "work"

import sys
sys.path.append(str(SRC))

import gstools as gs 
from gstools.random import MasterRNG
import scipy.stats as stats
import numpy as np
import datetime as date 
import copy 
import pytest 

from model.norfolk_model import NorfolkModel
from model.norfolk_ensemble import NorfolkEnsemble

def test_llnl_random_walk(): 
    model = NorfolkModel(name="Test Model")
    ensemble = NorfolkEnsemble(NorfolkModel=model)

    ensemble.draw(1)
    assert True