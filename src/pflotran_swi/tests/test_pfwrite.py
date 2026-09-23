from pflotran_swi.norfolk_model import NorfolkModel
import pandas as pd 
import h5py 
import pint
import numpy as np  
import os
from pflotran_swi.units import ureg
import glob
import re
import pytest
import pflotran_swi.pfwrite as pfw

def test_pfwrite(tmp_path):
    test_model = NorfolkModel()
    tmp_path_string = str(tmp_path)
    pfw.write(test_model, dir=tmp_path_string)

def test_post_templates_match_written_files(tmp_path):
    """Every REGION / dataset file the post-spinup PFLOTRAN templates reference must be written."""
    pfw.write(NorfolkModel(), dir=str(tmp_path))
    post_dir = tmp_path / "postrun"
    with h5py.File(post_dir / "regions.h5", "r") as hf:
        written_regions = set(hf["Regions"].keys())
    templates = "".join((post_dir / name).read_text() for name in
                        ("regions.txt", "flow_conditions.txt", "condition_coupler.txt"))
    h5_regions = {name for name in re.findall(r"^REGION (\w+)", (post_dir / "regions.txt").read_text(), re.M)
                  if name != "All"}
    assert h5_regions <= written_regions, f"Templates declare regions missing from regions.h5: {h5_regions - written_regions}"
    for filename in re.findall(r"FILENAME (\S+\.h5)", templates):
        assert (post_dir / filename).exists(), f"{filename} referenced by post templates but not written"
        with h5py.File(post_dir / filename, "r") as hf:
            group = re.search(rf"FILENAME {re.escape(filename)}\s+HDF5_DATASET_NAME (\w+)", templates).group(1)
            assert group in hf, f"{group} missing from {filename}"


def test_post_coastal_dataset_origin(tmp_path):
    model = NorfolkModel()
    pfw.write(model, dir=str(tmp_path))
    n_coastal = len(np.squeeze(model.region_post_spinup_coastal))
    with h5py.File(tmp_path / "postrun" / "coastal_pressure.h5", "r") as hf:
        assert hf["coastal_pressure/Data"].shape[0] == n_coastal
        assert hf["coastal_pressure"].attrs["Origin"] == pytest.approx((model.nx - n_coastal) * model.dx.magnitude)


@pytest.mark.parametrize("phase", ["spinup", "postrun"])
def test_coupler_regions_are_declared_and_written(tmp_path, phase):
    """Every REGION a boundary/initial condition uses must be declared in regions.txt and exist in regions.h5."""
    pfw.write(NorfolkModel(), dir=str(tmp_path))
    run_dir = tmp_path / phase
    used = set(re.findall(r"^\s*REGION (\w+)", (run_dir / "condition_coupler.txt").read_text(), re.M))
    declared = set(re.findall(r"^REGION (\w+)", (run_dir / "regions.txt").read_text(), re.M))
    with h5py.File(run_dir / "regions.h5", "r") as hf:
        written = set(hf["Regions"].keys())
    assert used <= declared, f"{phase}: coupler uses undeclared regions {used - declared}"
    assert declared - {"All"} <= written, f"{phase}: regions.txt declares regions missing from regions.h5 {declared - {'All'} - written}"
