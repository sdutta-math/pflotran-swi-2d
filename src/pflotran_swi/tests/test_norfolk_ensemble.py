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

def test_draw_records_sampling_seeds_without_changing_draws():
    """Seed bookkeeping must not perturb master_rng's stream: same draws as an unrecorded stream."""
    template = NorfolkModel()
    a = NorfolkEnsemble(NorfolkModel=template, seed=7).draw(2)
    b = NorfolkEnsemble(NorfolkModel=template, seed=7).draw(2)
    for ma, mb in zip(a, b):
        assert ma.sampling_seeds == mb.sampling_seeds
        np.testing.assert_array_equal(ma.field, mb.field)
    assert a[0].sampling_seeds != a[1].sampling_seeds
    expected = {"field", "land_profile", "shelf_profile", "dh_sea", "recharge"}
    expected |= {f"{n}_{m:02d}" for n in ("salinity", "air_pressure") for m in range(1, 13)}
    assert set(a[0].sampling_seeds) == expected


def test_parameter_table_matches_models():
    models = NorfolkEnsemble(NorfolkModel=NorfolkModel(), seed=3).draw(3)
    table = models.parameter_table()
    assert len(table) == 3
    for row, model in zip(table.itertuples(), models):
        assert row.recharge_m_per_year == pytest.approx(model.recharge.to("meter/year").magnitude)
        assert row.seed_field == model.sampling_seeds["field"]
        assert row.annual_salinity_g_per_kg_01 == pytest.approx(model.annual_salinity[0].magnitude)
