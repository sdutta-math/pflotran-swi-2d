from pflotran_swi.norfolk_model import NorfolkModel
import pandas as pd
import h5py
import pint
import numpy as np
import os
from pflotran_swi.units import ureg
import glob
import warnings
import shutil
import importlib.resources

def _magnitude(data):
    """Strip pint units for numpy/h5py/pandas interop, which don't understand Quantity."""
    if isinstance(data, pint.Quantity):
        return data.magnitude
    if isinstance(data, list):
        return np.array([_magnitude(d) for d in data])
    return data

def _default_ctrl_dir(name: str) -> str:
    """Bundled Norfolk-example PFLOTRAN control deck templates (spinup/post).

    Not tied to any particular simulation - callers running their own SWI
    scenario should pass their own ctrl_dir to write_spinup_run/write_post_run.
    """
    return str(importlib.resources.files("pflotran_swi") / "pfctrl" / name)

def _create_gridded_dataset(data, outfilename, group_name, dimension, discretization,
                           origin = [0.,0.], max_buffer_size = [4],
                           interpolation_method = 'LINEAR',
                           cell_centered = True, time_data = None, time_units = None):

    h5file = h5py.File(outfilename, mode='w')
    h5grp = h5file.create_group(group_name)
    # Add the following HDF5 Attributes to the HDF5 Group
    h5grp.attrs['Dimension'] = np.bytes_(dimension)
    h5grp.attrs['Discretization'] = discretization
    h5grp.attrs['Origin'] = origin
    h5grp.attrs['Max Buffer Size'] = max_buffer_size
    h5grp.attrs['Space Interpolation Method'] = np.bytes_(interpolation_method)
    h5grp.attrs['Cell Centered'] = [cell_centered]

    if time_units is not None:
        h5grp.attrs['Time Units'] = np.bytes_(time_units)
        h5grp.attrs['Transient'] = [True]
    h5dset = h5grp.create_dataset('Data', data = data)

    if time_data is not None:
        time = time_data
        h5dset = h5grp.create_dataset('Times', data = time)
    h5file.close()

# Internal PFLOTRAN specifiers 
def _write_pf_strata(region, material_id, outfilename = "/strata.h5"):
    with h5py.File(outfilename, mode='w') as h5file:
        h5file.create_dataset("/Materials/Cell Ids", data=region) # PFLOTRAN uses 1-based indexing
        h5file.create_dataset("/Materials/Material Ids", data=material_id)
    
    return None

def _write_pf_surfaces(region_names, regions, face_ids, outfilename = "/surfaces.h5"):
    with h5py.File(outfilename, mode='w') as h5file:
        for name, region, face_id in zip(region_names, regions, face_ids):
            h5file.create_dataset(f"/Regions/{name}/Cell Ids", data=region)
            h5file.create_dataset(f"/Regions/{name}/Face Ids", data=face_id)
    return None

def _write_pf_cell_indexed_dataset(cell_ids, ds_names, data_list, outfilename = "/cell_indexed.h5"):
    with h5py.File(outfilename, mode='w') as h5file:
        h5file['Cell Ids'] = cell_ids
        for name, data in zip(ds_names, data_list):
            h5file[name] = data
    return None

# Bridge from NorfolkModels to PFLOTRAN input files

def write_spinup_run(model: NorfolkModel, spinup_dir: str = "./spinup/", ctrl_dir: str = None):
    if not os.path.exists(spinup_dir):
        os.makedirs(spinup_dir)
    elif os.listdir(spinup_dir):
        raise OSError(f"Directory {spinup_dir} is not empty.  Please remove contents or specify a new directory.")
    
    _write_pf_strata(   region = np.reshape(model.cell_ids, (model.nx * 1 * model.nz,), order='F'),
                        material_id = np.reshape(model.subsurface_mask, (model.nx * 1 * model.nz,), order='F').astype(np.int8),
                        outfilename = spinup_dir + "strata.h5")
    
    region_names = ["Recharge", "Creek", "Sea", "Wetted"]
    regions = [np.squeeze(model.region_spinup_recharge), 
               np.squeeze(model.region_creek), 
               np.squeeze(model.region_sea), 
               np.squeeze(model.region_spinup_wetted)]
    face_ids = [[6]*len(regions[0]), 
                [1]*len(regions[1]), 
                [2]*len(regions[2]), 
                [6]*len(regions[3])]
    _write_pf_surfaces(region_names, regions, face_ids, outfilename = spinup_dir + "regions.h5")

    perm_names = ['perm_x', 'perm_z']
    perm_data_flat = [model.perm_x.flatten(), model.perm_z.flatten()]
    mask_1d = model.subsurface_mask.flatten(order='F')
    masked_perm_data = np.where(mask_1d, perm_data_flat, 0)
    _write_pf_cell_indexed_dataset( cell_ids = model.cell_ids.flatten(order='F'),
                                    ds_names = perm_names,
                                    data_list = masked_perm_data,
                                    outfilename = spinup_dir + "perm.h5")

    poro_names = ['poro']
    poro_data_flat = [model.poro.flatten()]
    masked_poro_data = np.where(mask_1d, poro_data_flat, 0)
    _write_pf_cell_indexed_dataset( cell_ids = model.cell_ids.flatten(order='F'),
                                    ds_names = poro_names,
                                    data_list = masked_poro_data,
                                    outfilename = spinup_dir + "poro.h5")

    _create_gridded_dataset(data = _magnitude(np.squeeze(model.pressure_profile_at_left)),
                            outfilename = spinup_dir + "creek_pressure.h5",
                            group_name='creek_pressure',
                            discretization= model.dz,
                            origin = model.origin[1],
                            max_buffer_size = [4], 
                            interpolation_method = 'STEP', 
                            cell_centered= True, 
                            time_data = None,
                            time_units = None,
                            dimension="Z")

    _create_gridded_dataset(data = _magnitude(model.spinup_sea_pressure_profile_record).T,
                            outfilename = spinup_dir + "sea_pressure.h5",
                            group_name='sea_pressure',
                            dimension="Z", 
                            discretization= model.dz,
                            origin = model.origin[1],
                            max_buffer_size = [4],
                            interpolation_method= 'STEP',
                            cell_centered= True,
                            time_data = _magnitude(model.spinup_month_record_hourly), 
                            time_units = 'h'
    )

    _create_gridded_dataset(data = _magnitude(model.spinup_wetted_pressure_profile_record).T,
                            outfilename = spinup_dir + "wetted_pressure.h5",
                            group_name='wetted_pressure',
                            dimension="X",
                            discretization= model.dx,
                            origin = (model.nx - len(model.region_spinup_wetted)) * model.dx,    
                            max_buffer_size = [4],
                            interpolation_method= 'STEP',
                            cell_centered= True,
                            time_data = _magnitude(model.spinup_month_record_hourly),
                            time_units = 'h'
    )

    gwr_annual = np.ones((12,)) * model.recharge
    df_gwr_month = {'TIME_UNITS': _magnitude(model.spinup_month_record_hourly[:12]), 'hour': _magnitude(gwr_annual)}
    df_gwr_month = pd.DataFrame(df_gwr_month)
    df_gwr_month.to_csv(spinup_dir + '/GWR_monthly_mod_30Percent.dat', index=False, sep = ' ', float_format='%.6e')
    
    with open(spinup_dir + '/constraints_list_1yr_1monthFrequency_20times_mod_resample.txt', 'w') as output_file:
        for tstamp, step in zip(model.spinup_month_record_hourly, range(len(model.spinup_month_record_hourly))):
            output_file.write(f"{tstamp.magnitude} t{step+1}\n")

    with open(spinup_dir + '/constraints_1yr_1monthFrequency_20times_mod_resample.txt', 'w') as f:
        for step, salinity in zip(range(len(model.spinup_month_record_hourly)), model.spinup_salinity_record):
            f.write(f"CONSTRAINT t{step+1}\n")
            f.write("  CONCENTRATIONS\n")
            f.write(f"    Sal {salinity:.2f} T\n")
            f.write("  /\n")
            f.write("END\n")

    if ctrl_dir is None:
        ctrl_dir = _default_ctrl_dir("spinup")
    files_to_copy = glob.glob(os.path.join(ctrl_dir, '*'))
    for file_path in files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy(file_path, spinup_dir)

def write_post_run(model: NorfolkModel, post_dir: str = "./postrun/", spinup_dir: str = "./spinup/", ctrl_dir: str = None):
    # Create a symbolic link to the spinup strata file in the postrun directory
    if not os.path.exists(post_dir):
        os.makedirs(post_dir)
    elif os.listdir(post_dir):
        raise OSError(f"Directory {spinup_dir} is not empty.  Please remove contents or specify a new directory.")
    

    def link_spinup(src, dst, strict=True):
        spinup_file_path = os.path.join(spinup_dir, src)
        postrun_file_path = os.path.join(post_dir, dst)
        if os.path.exists(spinup_file_path) and not os.path.exists(postrun_file_path):
            os.symlink(os.path.abspath(spinup_file_path), postrun_file_path)
        elif os.path.exists(postrun_file_path):
            os.remove(postrun_file_path)
            os.symlink(os.path.abspath(spinup_file_path), postrun_file_path)
        elif not strict:
            warnings.warn(f"{spinup_file_path} not found.  Strict mode is off, so preempting post-spinup with dangling symlink.")
            os.symlink(os.path.abspath(spinup_file_path), postrun_file_path)
        else: 
            raise AssertionError(f"Spinup file not found at {spinup_file_path}.  No symlink created.")
        

    static_files = ['strata.h5', 'perm.h5', 'poro.h5']

    for static_file in static_files:
        link_spinup(static_file, static_file)
    
    # PFLOTRAN writes the final-state checkpoint as <input_prefix>-restart.h5 at the end of the spinup run.
    link_spinup('pflotran-restart.h5', 'pflotran_spinup_restart.h5', strict=False)

    # Combine into one PFLOTRAN region, since PFLOTRAN doesn't support dynamic regions. 
    
    surf_pressures = model.post_spinup_surf_pressure_profile_record
    wetted_pressures = model.post_spinup_wetted_pressure_profile_record
    coastal_pressures = []
    region_post_spinup_wetted = model.region_post_spinup_wetted
    region_post_spinup_surf = model.region_post_spinup_surf
    for nt in range(int(model.POST_SPINUP_DURATION.to(ureg.month).magnitude)):
        coastal_pressures.append(np.concatenate((surf_pressures[nt], wetted_pressures[nt])))
    
    region_names = ["Recharge", "Creek", "Sea", "Coastal"]
    regions = [np.squeeze(model.region_post_spinup_recharge), 
               np.squeeze(model.region_creek),
               np.squeeze(model.region_sea),
               np.squeeze(model.region_post_spinup_coastal)]
    
    face_ids = [[6]*len(regions[0]),
                [1]*len(regions[1]),
                [2]*len(regions[2]),
                [6]*len(regions[3])]
    _write_pf_surfaces(region_names, regions, face_ids, outfilename = post_dir + "regions.h5")

    _create_gridded_dataset(data = _magnitude(np.squeeze(model.pressure_profile_at_left)),
                            outfilename = post_dir + "creek_pressure.h5",
                            group_name='creek_pressure',
                            discretization= model.dz,
                            origin = model.origin[1],
                            max_buffer_size = [4],
                            interpolation_method = 'STEP',
                            cell_centered= True,
                            time_data = None,
                            time_units = None,
                            dimension="Z")
    
    _create_gridded_dataset(data = _magnitude(model.post_spinup_sea_pressure_profile_record).T,
                            outfilename = post_dir + "sea_pressure.h5",
                            group_name='sea_pressure',
                            dimension="Z",
                            discretization= model.dz,
                            origin = model.origin[1],
                            max_buffer_size = [4],
                            interpolation_method= 'STEP',
                            cell_centered= True,
                            time_data = _magnitude(model.hour_monthly_record),
                            time_units = 'h'
    )

    _create_gridded_dataset(data = _magnitude(coastal_pressures).T,
                            outfilename = post_dir + "coastal_pressure.h5",
                            group_name='coastal_pressure',
                            dimension="X",
                            discretization= model.dx,
                            origin = (model.nx - len(np.squeeze(model.region_post_spinup_coastal))) * model.dx,
                            max_buffer_size = [4],
                            interpolation_method= 'STEP',
                            cell_centered= True,
                            time_data = _magnitude(model.hour_monthly_record),
                            time_units = 'h'
    )

    gwr_annual = np.ones((12,)) * model.recharge
    df_gwr_month = {'TIME_UNITS': _magnitude(model.hour_monthly_record[:12]), 'hour': _magnitude(gwr_annual)}
    df_gwr_month = pd.DataFrame(df_gwr_month)
    df_gwr_month.to_csv(post_dir + '/GWR_monthly_mod_30Percent.dat', index=False, sep = ' ', float_format='%.6e')

    with open(post_dir + '/constraints_list_1yr_1monthFrequency_20times_mod_resample.txt', 'w') as output_file:
        for tstamp, step in zip(model.hour_monthly_record, range(len(model.hour_monthly_record))):
            output_file.write(f"{tstamp.magnitude} t{step+1}\n")

    with open(post_dir + '/constraints_1yr_1monthFrequency_20times_mod_resample.txt', 'w') as f:
        for step, salinity in zip(range(len(model.hour_monthly_record)), model.post_salinity_record):
            f.write(f"CONSTRAINT t{step+1}\n")
            f.write("  CONCENTRATIONS\n")
            f.write(f"    Sal {salinity:.2f} T\n")
            f.write("  /\n")
            f.write("END\n")

    if ctrl_dir is None:
        ctrl_dir = _default_ctrl_dir("post")
    files_to_copy = glob.glob(os.path.join(ctrl_dir, '*'))
    for file_path in files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy(file_path, post_dir)


    # Well stuff here

def write(model: NorfolkModel, dir, spinup_ctrl_dir: str = None, post_ctrl_dir: str = None):
    write_spinup_run(model, spinup_dir = dir + "/spinup/", ctrl_dir = spinup_ctrl_dir)
    write_post_run(model, post_dir = dir + "/postrun/", spinup_dir = dir + "/spinup/", ctrl_dir = post_ctrl_dir)

