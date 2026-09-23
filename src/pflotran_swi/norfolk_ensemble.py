import gstools as gs 
from gstools.random import MasterRNG
import scipy.stats as stats
import numpy as np
import datetime as date 
import copy 
from pflotran_swi.units import ureg

from pflotran_swi.norfolk_model import NorfolkModel

class NorfolkModelList(list):
    """A list of NorfolkModel realizations that can carry a descriptive name."""

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

class NorfolkEnsemble:

    def __init__(self, NorfolkModel: NorfolkModel, **kwargs): 
        self.name   =                           kwargs.get('name', 'Norfolk Ensemble')
        self.template =                         NorfolkModel

        self.covariance_model =                 kwargs.get('covariance_model', gs.Exponential)
        # [25, 20] m was the original (pre-abstractify) design value, carried
        # over unchanged -- but with correct meter-space positions (see
        # _draw_subsurface_field), 20 m of vertical correlation is ~the
        # entire lz=20 m domain height, i.e. almost no vertical structure.
        # [25, 2] keeps the horizontal scale and sizes the vertical scale to
        # this domain's actual thickness instead.
        self.len_scales =                       kwargs.get('len_scales', [25.0, 2.0])
        self.angles =                           kwargs.get('angles', 0)
        # Single sequential seed stream for every random draw in the ensemble (subsurface
        # field, land/ocean profiles, salinity/pressure/water-table/recharge distributions).
        # Reproducibility depends on the seed AND on draw order: since master_rng() hands
        # out the next int from one RandomState stream, adding, removing, or reordering a
        # draw call shifts every seed drawn after it, changing the resulting realizations
        # even with the same seed value.
        self.master_rng =       MasterRNG(kwargs.get('seed', 20201007))
    
        self.water_table_gain_from_msl_dist =   kwargs.get("water_table_gain_from_msl_dist", stats.uniform(loc=0.5, scale=2.0))
        self.recharge_dist =                    kwargs.get("recharge_dist", stats.uniform(loc=1e-9, scale= 5e-8 - 1e-9))
        self.air_pressure_at_msl_dist =         kwargs.get("air_pressure_at_msl_dist", stats.uniform(loc=101325-1000*9.8*0.2, scale=2000*9.8*0.2))
        
        llnl_salinity_dist_maxes = np.array([33, 32, 35, 38, 37, 42, 40, 46, 40, 37, 39, 38]) / 58.442469
        llnl_salinity_dist_mins = llnl_salinity_dist_maxes - 6 / 58.442469
        llnl_salinity_ranges = zip(llnl_salinity_dist_mins, llnl_salinity_dist_maxes)
        default_salinity_rvs = [stats.uniform(loc=min, scale=max - min) for min, max in llnl_salinity_ranges]
        
        self.salinity_dists =                   kwargs.get("salinity_dists", default_salinity_rvs)
        self.land_profile_dist =                kwargs.get("land_profile_sde", "LLNL")
        self.ocean_profile_dist =               kwargs.get("ocean_profile_sde", "LLNL")


    def _llnl_random_walk(self, left_elv_range: tuple, right_elv_range: tuple, nx, nz, dhz = 50) -> int:
        """
        This is the way LLNL sampled their random walk.
        They sample elevations in cells, so to keep it consistent, it is converted back into physical units
        at the end of the function.

        The walk increment is uniform with the increment bounded by an envelope, endpoints, and a walk term.

        FIXME: dhz (default 50) and the +-3 per-step bound below are absolute cell counts,
        not scaled by dx/dz. Changing grid resolution (nx/nz) without touching these changes
        the taper proportion and the max realistic terrain slope, i.e. the statistical
        character of generated profiles is resolution-coupled, not resolution-invariant.
        Also assumes nz > dhz -- coarsening nz below ~50 breaks the envelope.
        """

        master_envelope_max = np.log(range(nx, 0, -1)) / np.log(nx) * (nz - dhz) + dhz
        master_envelope_min = (np.log(nx+1) - np.log(range(1, nx+1))) / (np.log(nx+1) - np.log(1)) * (nz - dhz) + dhz

        nzs = np.ones((nx,), dtype = np.float32)

        # Local RNG seeded from master_rng() instead of the numpy global RNG, so this walk
        # is reproducible from the ensemble's seed alone.
        rs = np.random.RandomState(self.master_rng())

        if left_elv_range[0] == left_elv_range[1]:
            nzs[0] = left_elv_range[0]
        else:
            nzs[0] = rs.randint(left_elv_range[0], left_elv_range[1])

        if right_elv_range[0] == right_elv_range[1]:
            nzs[-1] = right_elv_range[0]
        else:
            nzs[-1] = rs.randint(right_elv_range[0], right_elv_range[1])

        max_scale = (master_envelope_max - dhz) / (nz - dhz) * (nzs[0] - nzs[-1]) + nzs[-1]
        min_scale = (master_envelope_min - dhz) / (nz - dhz) * (nzs[0] - nzs[-1]) + nzs[-1]

        for i in range(1, nx-1):
            # +-3 is a hard cell-count bound, not derived from dz -- see FIXME on this method.
            min_val = np.max(np.array([nzs[i-1] - 3, min_scale[i], nzs[-1]]))
            max_val = np.min(np.array([nzs[i-1] + 3, max_scale[i]])) + 1
            if max_val - min_val < 1:
                nzs[i] = int(min_val)
            else:
                nzs[i] = rs.randint(min_val, max_val)

        return nzs

    @staticmethod
    def _llnl_smooth_profile(profile, d_smooth = 10) -> int:
        # Simple moving average smoothing
        # FIXME: d_smooth is a cell-count window (~2*d_smooth+1 x-cells), not a physical
        # length. Refining nx shrinks the real smoothing distance, so profile roughness
        # changes with grid resolution alone -- same coupling issue as _llnl_random_walk's
        # dhz/+-3 constants.
        smoothed_elevations = profile.copy()
        nx, = profile.shape
        for j in range(1, nx-1):
            start_idx = np.maximum(0, j-d_smooth)
            end_idx = np.minimum(j + d_smooth + 1, nx)
            if j < d_smooth: 
                end_idx = j * 2 + 1
            if j > nx - d_smooth: 
                start_idx = 2 * j - nx
            smoothed_elevations[j] = int(np.mean(profile[start_idx:end_idx]))
        
        return smoothed_elevations
    
    def _draw_subsurface_field(self):
        model = self.covariance_model(dim=2, var=1.0, len_scale=self.len_scales, angles=self.angles)
        srf = gs.SRF(model, seed = 1)
        # Position arrays must be in the same units as len_scales (meters) --
        # gstools has no notion of grid spacing, so passing bare cell indices
        # here previously made len_scale=[25,20] mean 25/20 *cells*, not
        # meters. With dx=5 m, dz=0.1 m (a 50:1 grid-spacing ratio) that
        # silently produced a ~36:1 physical anisotropy (~90 m x ~2.5 m,
        # empirically measured) instead of the intended near-isotropic
        # 25 m x 20 m field.
        idxs = np.arange(self.template.nx) * self.template.dx.magnitude
        idzs = np.arange(self.template.nz) * self.template.dz.magnitude
        srf.set_pos([idxs, idzs], "structured")
        date_time = date.datetime.now()
        seed = date_time.microsecond
        return srf(seed=self.master_rng())
    
    def _draw_land_profile(self):
        if self.land_profile_dist == "LLNL":
            left_nz = self.template.ELEVATION_LEFT_NZ
            right_nz = self.template.mean_sea_level_nz
            # Walk spans the land region only (land_nx columns), not the
            # full domain width -- NorfolkModel.land_profile requires
            # len(value) == land_nx (see __setattr__).
            sample = self._llnl_random_walk((left_nz, left_nz), (right_nz, right_nz), self.template.land_nx, self.template.nz)
            smoothed = self._llnl_smooth_profile(sample)
            return self.template.nz_to_elevation(smoothed)

    def _draw_ocean_profile(self):
        if self.ocean_profile_dist == "LLNL":
            left_nz = self.template.mean_sea_level_nz
            right_nz = self.template.shelf_break_nz
            # Walk spans the shelf region only (shelf_nx columns) -- see
            # note in _draw_land_profile above.
            sample = self._llnl_random_walk((left_nz,left_nz), (right_nz,right_nz), self.template.shelf_nx, self.template.nz)
            smoothed = self._llnl_smooth_profile(sample)
            return self.template.nz_to_elevation(smoothed)

    def _draw_salinity_record(self):
        return [self.salinity_dists[month].rvs(random_state=self.master_rng()) for month in range(12)] * ureg.gram/ureg.kg

    def _draw_air_pressure_at_msl(self):
        return [self.air_pressure_at_msl_dist.rvs(random_state=self.master_rng()) for _ in range(12)] * ureg.pascal

    def _draw_water_table_gain_from_msl(self):
        return self.water_table_gain_from_msl_dist.rvs(random_state=self.master_rng())* ureg.meter

    def _draw_recharge(self):
        return self.recharge_dist.rvs(random_state=self.master_rng()) *ureg.meter/ureg.year

    def _draw(self, name): 
        new_model = copy.copy(self.template)
        
        new_model.model_name = name
        new_model.field = self._draw_subsurface_field()
        # NorfolkModel reads land_profile/shelf_profile/annual_salinity;
        # assigning under any other name (as this did previously) is a
        # silent no-op -- the draw runs, the result is discarded, and every
        # realization keeps the template's default deterministic profile.
        new_model.land_profile = self._draw_land_profile()
        new_model.shelf_profile = self._draw_ocean_profile()
        new_model.annual_salinity = self._draw_salinity_record()
        new_model.annual_air_pressure_at_sea_level = self._draw_air_pressure_at_msl()
        new_model.dh_sea = self._draw_water_table_gain_from_msl()
        new_model.recharge = self._draw_recharge()

        return new_model

    def draw(self, n_models=1):
        return NorfolkModelList((self._draw('case' + str(i)) for i in range(n_models)), name=self.name)