import os
import h5py
import numpy as np
import pytest
import pathlib
import pint

from pflotran_swi.norfolk_model import NorfolkModel, ureg
import signal
import threading
import time

@pytest.fixture(scope="module")
def test_data(): 
    test_dir = pathlib.Path(__file__).parent
    data_dir = test_dir / "data"

    # Parameter data     
    h_inland_data = np.load(data_dir / "h_inland_all.npy")
    pressure_sea_data = np.load(data_dir / "pressure_sea_all.npy")

    # Reference output 
    
    with h5py.File(data_dir / "creek_pressure.h5", 'r') as hf:
        creek_pressure_data = hf['/creek_pressure/Data'][:]
    
    with h5py.File(data_dir / "sea_pressure.h5", 'r') as hf:
        sea_pressure_data = hf['/sea_pressure/Data'][:]

    return {
        "creek_pressure": creek_pressure_data,
        "h_inland": h_inland_data,
        "pressure_sea": pressure_sea_data,
        "sea_data": sea_pressure_data
    }


def test_id_to_nz():
    model = NorfolkModel()
    test_ids = [0, 439, 440, 441]

    expected_nz = [0, 0, 1 ,1]

    for cell_id, expected in zip(test_ids, expected_nz):
        assert model.id_to_nz(cell_id) == expected, f"Failed for cell_id {cell_id}"

    big_set = np.ones(1000000, dtype=int)

    def run_test():
        model.id_to_nz(big_set)

    thread = threading.Thread(target=run_test)
    thread.start()
    thread.join(3)  # 3-second timeout

    if thread.is_alive():
        pytest.fail("id_to_nz timed out after 3 seconds")

def test_elevation_to_nz(monkeypatch):

    monkeypatch.setattr("pflotran_swi.norfolk_model.NorfolkModel.__setattr__", object.__setattr__)

    model = NorfolkModel()
    test_elevations = [5.0, -15.0, -15 + 0.01, -15 + 0.1, -15 + 0.11, -1] * ureg.meter
    expected_nz = [199, 0, 0, 0, 1, 139]

    for elevation, expected in zip(test_elevations, expected_nz):
        assert model.elevation_to_nz(elevation) == expected, f"Failed for elevation {elevation}"

    model = NorfolkModel(nz = 5)
    expected_nz = [4, 0, 0, 0, 0, 3]
    for elevation, expected in zip(test_elevations, expected_nz):
        assert model.elevation_to_nz(elevation) == expected, f"Failed for elevation {elevation} with nz=5"

def test_creek_pressure(test_data):
    """
    Test that pressure calculations match expected reference data from files.
    
    This test:
    1. Loads reference pressure data from HDF5 file
    2. Loads model parameter data from NumPy files
    3. Instantiates a NorfolkModel with these parameters
    4. Compares calculated pressure profiles with reference data
    """

    #Test against the first case    
    h_inland = test_data["h_inland"][0] * ureg.meter
    pressure_sea = test_data["pressure_sea"][0] * ureg.pascal
 
    # Create NorfolkModel with parameters from reference data
    model = NorfolkModel(
        dh_sea=h_inland,
        nz = 200,
        origin = (0.0 * ureg.meter, -15.0 * ureg.meter),
        ELEVATION_LEFT = 5.0 * ureg.meter,
    )
    
    #Old calculation for expected creek pressure
    expected_creek_pressure = np.array([101325 + 1000 * 9.81 * ((idx - model.ELEVATION_LEFT_NZ + model.mean_sea_level_nz) * model.dz + model.dh_sea).magnitude for idx in range(model.nz)])[::-1]

    # Calculate pressure profile at left boundary (creek)
    calculated_creek_pressure = model.pressure_profile_at_left
    
    # Sort 'fixture' to decouple from order conventions (which is writer-dependent)

    # Convert calculated pressures to match units in reference data (assuming Pascal)
    calculated_pressure_pa = calculated_creek_pressure.to(ureg.pascal).magnitude
    
    calculated_pressure_pa = np.squeeze(calculated_pressure_pa)
    # Compare calculated values with expected values from reference data
    np.testing.assert_allclose(
        calculated_pressure_pa,
        expected_creek_pressure,
        atol=981,  # Allowable tolerance (1 meter of water column)
        err_msg="Calculated creek pressure doesn't match legacy formula"
    )

    expected_creek_pressure = test_data["creek_pressure"]

    np.testing.assert_allclose(
        calculated_pressure_pa,
        expected_creek_pressure,
        atol=981,  # Allowable tolerance (1 meter of water column)
        err_msg="Calculated creek pressure doesn't match reference data from HDF5"
    )

def test_sea_pressure(test_data):

    model = NorfolkModel(
        annual_air_pressure_at_sea_level = test_data["pressure_sea"][0] * ureg.pascal,
        SPINUP_DURATION = 10 * ureg.year
    )

    expected_sea_pressure = test_data["sea_data"]
    calc_sea_pressure = np.array([r.magnitude for r in model.spinup_sea_pressure_profile_record]).T

    np.testing.assert_allclose(
        calc_sea_pressure,
        expected_sea_pressure,
        atol=981,  # Allowable tolerance (1 meter of water column)
        err_msg="Calculated sea pressure doesn't match reference data from HDF5"
    )
 
def test_region_sea(monkeypatch):

    monkeypatch.setattr("pflotran_swi.norfolk_model.NorfolkModel.__setattr__", object.__setattr__)
    
    top_profile = np.linspace(5, 0.0, 8) * ureg.meter
    ocean_profile = np.linspace(-5.0, -6.0, 2) * ureg.meter
    model = NorfolkModel(nx = 10, nz = 5, shelf_break_elevation = -6.0 * ureg.meter, land_profile = top_profile, shelf_profile = ocean_profile)

    expected_region = [9, 19, 29]
    calc_region = model.region_sea.flatten()

    np.testing.assert_equal(calc_region, expected_region)

    assert np.squeeze(calc_region).shape == (3,), "Region sea shape mismatch"

def test_subsurface_mask(monkeypatch):

    monkeypatch.setattr("pflotran_swi.norfolk_model.NorfolkModel.__setattr__", object.__setattr__)

    land_profile = np.linspace(5, -10, 2) * ureg.meter
    ocean_profile = np.linspace(-10, -5, 2) * ureg.meter
    model = NorfolkModel(nx = 4, nz = 4, land_profile = land_profile, shelf_profile = ocean_profile)
    expected_mask =         np.array([[True, True, True, True],
                                        [True, False, False, False],
                                        [True, False, False, False],
                                        [True, True, False, False]])
    calc_mask = model.subsurface_mask[:,0,:]
    np.testing.assert_equal(calc_mask, expected_mask)

def test_boundary_ids(monkeypatch):
    model = NorfolkModel()
    top_ids = model.boundary_ids[2]
    assert len(top_ids) == model.nx, "Top boundary IDs length should match number of x cells"
    assert top_ids.shape == (model.nx,), "Top boundary IDs should be a 1D array"

    left_ids = model.boundary_ids[3]
    assert left_ids.shape == (1,200), "Left boundary IDs should be a 1D array"

    right_ids = model.boundary_ids[1]
    assert right_ids.shape == (1,model.shelf_break_nz+1), "Right boundary IDs should be a 1D array"

    bot_ids = model.boundary_ids[0]
    assert bot_ids.shape == (model.nx,1), "Bottom boundary IDs should be a 1D array"

def test_dynamic_region(): 
    model=NorfolkModel()
    
    top_ids = model.boundary_ids[2]
    surf = model.region_post_spinup_surf
    wetted = model.region_post_spinup_wetted

    num_surf = np.array([len(frame) for frame in model.region_post_spinup_surf])
    num_wetted = np.array([len(frame) for frame in model.region_post_spinup_wetted])
    num_coastal = num_surf + num_wetted
    for i,nct in enumerate(num_coastal):
        assert nct == num_coastal[0], f"Number of coastal cells should remain constant, but changed at step {i}"

    surf_p = model.post_spinup_surf_pressure_profile_record
    wetted_p = model.post_spinup_wetted_pressure_profile_record
    num_surf_p = np.array([len(frame) for frame in surf_p])
    num_wetted_p = np.array([len(frame) for frame in wetted_p])
    num_coastal_p = num_surf_p + num_wetted_p
    for i,nct in enumerate(num_coastal_p):
        assert nct == num_coastal_p[0], f"Number of coastal pressure cells should remain constant, but changed at step {i}"

    for nt in range(360):
        current_surf_p = surf_p[nt]
        current_wetted_p = wetted_p[nt]
        current_coast_p = np.concatenate((current_surf_p, current_wetted_p))
        assert len(current_coast_p) == num_coastal[0], f"Number of coastal pressure cells should remain constant, but changed at step {nt}" 

def test_post_spinup_coastal_and_recharge_partition_top():
    model = NorfolkModel()
    terminal_nz = model.elevation_to_nz(model.mean_sea_level_elevation + model.sea_level_anomaly_rate * model.POST_SPINUP_DURATION)
    assert model.anomaly_nz == terminal_nz - model.mean_sea_level_nz
    assert 0 < model.anomaly_nz < 10, "Sea level rise should be a few cells, not an absolute index"
    coastal = np.squeeze(model.region_post_spinup_coastal)
    recharge = np.squeeze(model.region_post_spinup_recharge)
    assert len(recharge) > 0, "Land above the terminal sea level must still receive recharge"
    assert len(coastal) < model.nx
    assert len(coastal) + len(recharge) == model.nx
    assert len(np.intersect1d(coastal, recharge)) == 0
    assert len(coastal) > len(np.squeeze(model.region_spinup_wetted)), "Coastal extent must exceed the spinup wetted extent"
