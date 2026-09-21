import gstools as gs
from gstools.random import MasterRNG
import scipy.stats as stats
import numpy as np
import datetime as date 
import copy 
import pytest 

from pflotran_swi.norfolk_model import NorfolkModel
from pflotran_swi.norfolk_ensemble import NorfolkEnsemble

def test_llnl_random_walk(): 
    model = NorfolkModel(name="Test Model")
    ensemble = NorfolkEnsemble(NorfolkModel=model)

    ensemble.draw(1)
    assert True