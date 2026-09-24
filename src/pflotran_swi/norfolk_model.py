import pint
import numpy as np  
from pflotran_swi.units import ureg

class NorfolkModel:
    """
    Variable definitions are assumed to follow CF conventions.

    A class to represent a PFLOTRAN model specification for Norfolk, VA. 

    This class defines the parameters and methods necessary to outline a PFLOTRAN model simulating 
    groundwater flow and transport of salinity in a transect of a coastal aquifer system. The class 
    encapsulates the specification of the idealized problem as well as representation in a finite volume
    grid, but does not include an interface to PFLOTRAN itself.

    The model features: 
    
    - Rectangular specifications for the super-grid
    - Specification for subsurface domain represented by a 2D mask 
    - Ocean/land profile specified via bisected profile specification for left/right profiles  
    - Fixed specifications for the ocean-land boundary and shelf-break elevation 
    - Specification of permeability and porosity for the subsurface domain
    - Specification of a freshwater water table at the left boundary via a hydrostatic reference head (pressure coordinates)
    - Specification of a rainfall source at the top boundary via a neumann condition 
    - Specification of a wetted region at the top boundary via a hydrostatic pressure condition integrated from reference head @ mean_sea_level
    - Specification of a far field (Sea) condition at the right boundary via a pressure dirichlet condition, spanning the domain from the bottom up to the shelf break 
    - Specification of an (implicit) no flow condition at the bottom boundary and vertical profile boundaries
    - Specification of a spinup 
    - Specification of a post-spinup period with sea level rise 
    - Specification of isothermal/isohaline/isobaric fresh and saltwater densities external to the model domain 
    
    Spinup: all model factors are held steady
    - Mean_sea_level specified via wetted region 
    - Salinity
    Post-spinup: 
    - Sea level rise specified via: 
        - An increased and annual reference head @ mean_sea_level
        - A statically increased wetted region extent, including unsaturated pressure regions
        - Specification of annual salinity variations at the right boundary
    """

    def __init__(self, **kwargs):
        # region Domain extents        
        self.model_name =                       kwargs.get("model_name", "Norfolk_example")
        self.lx =                               kwargs.get("lx", 1100.0 * ureg.meter)
        self.lz =                               kwargs.get("lz", 20.0 * ureg.meter)
        self.nx =                               kwargs.get("nx", 440)
        self.nz =                               kwargs.get("nz", 200)
        self.origin =                           kwargs.get("origin", (0.0 * ureg.meter, -15.0 * ureg.meter))
        self.l_land =                           kwargs.get("l_land", 1000.0 * ureg.meter)
        self.l_shelf =                          kwargs.get("l_shelf", 100.0 * ureg.meter)
        self.ELEVATION_LEFT =                   kwargs.get("ELEVATION_LEFT", 5.0 * ureg.meter)
        self.mean_sea_level_elevation =         kwargs.get("mean_sea_level_elevation", 0.0 * ureg.meter)
        self.shelf_break_elevation =                  kwargs.get("shelf_break_elevation", -8.0 * ureg.meter)
        # endregion

        # region Subsurface properties 
        self.perm_mean =                        kwargs.get("perm_mean", 4.5)
        self.perm_std =                         kwargs.get("perm_std", 0.5)
        self.poro_mean =                        kwargs.get("poro_mean", 0.5)
        self.poro_std =                         kwargs.get("poro_std", 0.05)
        # endregion

        # region Physical specifications 
        self.freshwater_density =               kwargs.get("freshwater_density", 1000 * ureg.kilogram / ureg.meter**3)
        self.saltwater_density =                kwargs.get("saltwater_density", 1025 * ureg.kilogram / ureg.meter**3)
        self.SPINUP_DURATION =                  kwargs.get("SPINUP_DURATION", 10 * ureg.year)
        self.POST_SPINUP_DURATION =             kwargs.get("POST_SPINUP_DURATION", 30 * ureg.year)
        self.dh_sea =                           kwargs.get("dh_sea", 1.0 * ureg.meter)
        self.recharge =                         kwargs.get("recharge", 1.0 * ureg.meter / ureg.year)
        self.SPINUP_AIR_PRESSURE_AT_SEA_LEVEL = kwargs.get("SPINUP_AIR_PRESSURE_AT_SEA_LEVEL", 101325 * ureg.pascal)
        self.SPINUP_SALINITY =                  kwargs.get("SPINUP_SALINITY", 35 / 58.442469 * ureg.gram / ureg.kilogram)
        self.sea_level_anomaly_rate =           kwargs.get("sea_level_anomaly_rate", 1.0/100 * ureg.meter / ureg.year)
        self.annual_air_pressure_at_sea_level = kwargs.get("annual_air_pressure_at_sea_level", np.full(12, 101325) * ureg.pascal)
        self.annual_salinity =                  kwargs.get("annual_salinity", np.full(12, 35 / 58.442469) * ureg.gram / ureg.kilogram)
        
        self.land_profile = kwargs.get(
            "land_profile",
            np.linspace(self.ELEVATION_LEFT.magnitude, self.mean_sea_level_elevation.magnitude, 400) * ureg.meter
        )
        self.shelf_profile = kwargs.get(
            "shelf_profile",
            np.linspace(self.mean_sea_level_elevation.magnitude, self.shelf_break_elevation.magnitude, 40) * ureg.meter
        )
        self.field = kwargs.get("field", np.ones((self.nx, self.nz)))
        # endregion

        # RNG seeds that produced this realization (empty for hand-built models); filled in by
        # NorfolkEnsemble._draw so parameters() can record them.
        self.sampling_seeds = kwargs.get("sampling_seeds", {})


    def __str__(self):
        return f"PFLOTRAN Model: {self.model_name}"

    def __setattr__(self, name, value):
        if name == "land_profile":
            assert len(value) == self.land_nx, "Land profile length must match land_nx"
        elif name == "shelf_profile":
            assert len(value) == self.shelf_nx, "Shelf profile length must match shelf_nx"
        super().__setattr__(name, value)

    def parameters(self) -> dict:
        """JSON-serializable record of this realization's parameters (unit-stripped, units in key names).

        Holds the sampled scalars/arrays, the full land and shelf profiles, summary statistics of the
        subsurface fields (mean over subsurface_mask cells), and the RNG seeds if the model came from a
        NorfolkEnsemble.
        """
        mask = np.squeeze(self.subsurface_mask)
        perm_x = self.perm_x[mask]
        poro = self.poro[mask]
        poro = poro[poro > 0]
        return {
            "model_name": self.model_name,
            "nx": int(self.nx),
            "nz": int(self.nz),
            "lx_m": float(self.lx.to(ureg.meter).magnitude),
            "lz_m": float(self.lz.to(ureg.meter).magnitude),
            "dh_sea_m": float(self.dh_sea.to(ureg.meter).magnitude),
            "recharge_m_per_year": float(self.recharge.to(ureg.meter / ureg.year).magnitude),
            "sea_level_anomaly_rate_m_per_year": float(self.sea_level_anomaly_rate.to(ureg.meter / ureg.year).magnitude),
            "annual_salinity_g_per_kg": np.asarray(self.annual_salinity.to(ureg.gram / ureg.kilogram).magnitude).tolist(),
            "annual_air_pressure_at_sea_level_Pa": np.asarray(self.annual_air_pressure_at_sea_level.to(ureg.pascal).magnitude).tolist(),
            "perm_mean": float(self.perm_mean),
            "perm_std": float(self.perm_std),
            "poro_mean": float(self.poro_mean),
            "poro_std": float(self.poro_std),
            "log10_perm_x_mean": float(np.log10(perm_x).mean()) if perm_x.size else None,
            "poro_field_mean": float(poro.mean()) if poro.size else None,
            "land_profile_m": np.asarray(self.land_profile.to(ureg.meter).magnitude).tolist(),
            "shelf_profile_m": np.asarray(self.shelf_profile.to(ureg.meter).magnitude).tolist(),
            "surface_elevation_mean_m": float(np.mean(self.profile.to(ureg.meter).magnitude)),
            "sampling_seeds": {k: int(v) for k, v in self.sampling_seeds.items()},
        }

    # region Derived grid properties
    @property
    def dx(self):
        return self.lx / self.nx

    @property
    def dz(self):
        return self.lz / self.nz

    @property
    def xs(self): 
        return self.origin[0] + np.arange(self.nx) * self.dx    
    
    @property 
    def zs(self): 
        return self.origin[1] + np.arange(self.nz) * self.dz
    
    @property 
    def profile(self): 
        return np.concatenate((self.land_profile, self.shelf_profile))
    
    @property
    def cell_ids(self):
        cell_id = np.reshape(np.arange((self.nx*1*self.nz)),(self.nx,1,self.nz),order="F")
        cell_id = cell_id.astype('int')
        return cell_id

    def elevation_to_nz(self, elevation):
            # Round up but adjust for zero indexing 
            ceiling = np.ceil(((elevation - self.origin[1]) / self.dz).magnitude).astype(int) - 1 
            return np.clip(ceiling, 0, self.nz - 1)
    
    def nz_to_elevation(self, nz):
        return nz * self.dz + self.origin[1]
    
    @property 
    def profile_nz(self): 
        profile_nz = self.elevation_to_nz(self.profile)
        return profile_nz

    @property
    def subsurface_mask(self):
        mask = np.ones((self.nx, 1, self.nz), dtype=bool)
        for i in range(self.nx):
            if self.profile_nz[i] < self.nz - 1:
                mask[i, 0, self.profile_nz[i]+1:] = False    
        return mask
    
    @property 
    def boundary_ids(self): 
        ix = np.arange(self.nx)
        top_ids =   self.cell_ids[ix,0,self.profile_nz]
        left_ids =  self.cell_ids[0,:,:]
        right_ids = self.cell_ids[-1,:,:self.shelf_break_nz+1]
        bot_ids =   self.cell_ids[:,:,0]
        return bot_ids, right_ids, top_ids, left_ids

    
    @property
    def mean_sea_level_nz(self):
        return self.elevation_to_nz(self.mean_sea_level_elevation)
    
    @property
    def shelf_break_nz(self):
        # Also the top of the Sea (right) boundary.
        return self.elevation_to_nz(self.shelf_break_elevation)
    
    @property
    def ELEVATION_LEFT_NZ(self):
        value = self.elevation_to_nz(self.ELEVATION_LEFT)
        if value != self.nz - 1:
            raise ValueError(f"ELEVATION_LEFT_NZ ({value}) must be equal to nz ({self.nz})")
        return value
    
    @property
    def land_nx(self):
        return int(self.l_land / self.dx)  # Grid index of the land region

    @property
    def shelf_nx(self):
        return int(self.l_shelf / self.dx)
    
    @property 
    def spinup_month_record_hourly(self): 
        month_index = np.arange(int(self.SPINUP_DURATION.to(ureg.month).magnitude)) * ureg.month 
        return month_index.to(ureg.hour)
    
    @property
    def hour_monthly_record(self): 
        month_index = np.arange(self.POST_SPINUP_DURATION.to(ureg.month).magnitude) * ureg.month 
        return month_index.to(ureg.hour)  

    @property
    def anomaly_nz(self) -> int:  
        """ Total post-spinup sea level rise, in cells (not an absolute elevation index) """
        total_sea_level_anomaly = self.sea_level_anomaly_rate * self.POST_SPINUP_DURATION
        terminal_nz = self.elevation_to_nz(self.mean_sea_level_elevation + total_sea_level_anomaly)
        return terminal_nz - self.mean_sea_level_nz

    @property 
    def sea_level_record(self, freq=ureg.month): 
        num_months = self.POST_SPINUP_DURATION.to(freq).magnitude

        anomaly_record = self.sea_level_anomaly_rate.to(ureg.meter/freq) * np.arange(num_months) * freq
        return self.mean_sea_level_elevation + anomaly_record
    
    @property
    def air_pressure_record(self):
        num_years = self.POST_SPINUP_DURATION.to(ureg.year).magnitude 
        return np.tile(self.annual_air_pressure_at_sea_level, num_years)
    
    # endregion    

    # region Derived subsurface properties 
    @property
    def perm(self): 
        return np.exp(self.field*self.perm_std + self.perm_mean) * 1e-11 
    
    @property
    def perm_x(self): 
        return self.perm
    
    @property
    def perm_z(self): 
        return 0.1 * self.perm
    
    @property
    def poro(self):
        return np.maximum(self.field*self.poro_std + self.poro_mean, 0)
    # endregion

    # region Spinup Regions 
    @property
    def region_creek(self):
        return self.boundary_ids[3]
    
    @property 
    def region_sea(self): 
        return self.boundary_ids[1]

    @property 
    def region_spinup_wetted(self): 
        top_ids = self.boundary_ids[2]
        return top_ids[self.profile_nz < self.mean_sea_level_nz]

    @property
    def region_spinup_recharge(self): 
        top_ids = self.boundary_ids[2]
        return top_ids[self.profile_nz >= self.mean_sea_level_nz]
    # endregion 

    # region Post-spinup Regions

    @property 
    def region_post_spinup_coastal(self): 
        top_ids = self.boundary_ids[2]
        return top_ids[self.profile_nz < (self.mean_sea_level_nz + self.anomaly_nz)]
    
    @property 
    def region_post_spinup_surf(self): 
        top_ids = self.boundary_ids[2]
        surf_id_record = []
        region_post_spinup_wetted = self.region_post_spinup_wetted
        region_post_spinup_coastal = self.region_post_spinup_coastal
        for nt in range(int(self.POST_SPINUP_DURATION.to(ureg.month).magnitude)):
            current_wetted = region_post_spinup_wetted[nt]
            current_surf = np.setdiff1d(region_post_spinup_coastal, current_wetted)
            surf_id_record.append(current_surf)
        return surf_id_record
    
    @property 
    def region_post_spinup_wetted(self):
        """ This region includes the saturated and unsaturated regions of the top boundary """  
        top_ids = self.boundary_ids[2]
        coastal = self.region_post_spinup_coastal
        sea_level_record = self.sea_level_record
        wetted_id_record = []
        for nt in range(int(self.POST_SPINUP_DURATION.to(ureg.month).magnitude)):
            current_sea_level = sea_level_record[nt]
            current_sea_level_nz = self.elevation_to_nz(current_sea_level)
            is_below_sea = self.profile_nz < current_sea_level_nz
            current_wetted = top_ids[is_below_sea]
            wetted_id_record.append(current_wetted)
        return wetted_id_record
    
    @property
    def region_post_spinup_recharge(self): 
        top_ids = self.boundary_ids[2]
        return top_ids[self.profile_nz >= (self.mean_sea_level_nz + self.anomaly_nz)]
    # endregion

    # region Pressure and helper methods
    def pressure_coordinates_at(self, elevation, reference_head, reference_elevation, density):
        """
        Calculate the hydrostatic pressure at a given elevation.
        """
        g = 9.81 * ureg.meter / ureg.second**2  
        return reference_head + density * g * (reference_elevation-elevation)
    
    @property
    def pressure_profile_at_left(self): 
        #pressure taken at cell bottoms 
        region_elevations = self.id_to_nz(self.region_creek) * self.dz + self.origin[1]
        hydrostatic_from_mean_sea_level = self.pressure_coordinates_at(
            elevation=region_elevations,
            reference_head=self.SPINUP_AIR_PRESSURE_AT_SEA_LEVEL,
            reference_elevation=self.mean_sea_level_elevation,
            density= self.freshwater_density
        )
        return hydrostatic_from_mean_sea_level + self.freshwater_density * 9.81 * ureg.meter / ureg.second**2 * self.dh_sea
    
    @property
    def spinup_sea_pressure_profile_record(self):
        region_elevations = self.id_to_nz(self.region_sea) * self.dz + self.origin[1]
        hydrostatic_from_mean_sea_level = self.pressure_coordinates_at(
            elevation=region_elevations,
            reference_head=self.SPINUP_AIR_PRESSURE_AT_SEA_LEVEL,
            reference_elevation=self.mean_sea_level_elevation,
            density= self.saltwater_density
        )
        num_months = self.SPINUP_DURATION.to(ureg.month).magnitude
        return [np.squeeze(hydrostatic_from_mean_sea_level) for _ in range(int(num_months))]
    
    def id_to_nz(self, cell_ids) -> np.ndarray:
        cell_ids = np.asarray(cell_ids)
        if np.any(cell_ids < 0) or np.any(cell_ids > self.nx * self.nz - 1):
            raise ValueError("Cell IDs out of bounds")
        return cell_ids // self.nx
    
    @property
    def spinup_wetted_pressure_profile_record(self): 
        region_elevations = self.id_to_nz(self.region_spinup_wetted) * self.dz + self.origin[1]
        hydrostatic_from_mean_sea_level = self.pressure_coordinates_at(
            elevation=region_elevations,
            reference_head=self.SPINUP_AIR_PRESSURE_AT_SEA_LEVEL,
            reference_elevation=self.mean_sea_level_elevation,
            density= self.saltwater_density
        )
        num_months = self.SPINUP_DURATION.to(ureg.month).magnitude
        return [np.squeeze(hydrostatic_from_mean_sea_level) for _ in range(int(num_months))]

    @property
    def post_spinup_wetted_pressure_profile_record(self):
        num_months = int(self.POST_SPINUP_DURATION.to(ureg.month).magnitude)
        region_elevations = [self.id_to_nz(region_frame) * self.dz + self.origin[1] for region_frame in self.region_post_spinup_wetted]
        wetted_pressure_record = []
        for elevation_frame, month in zip(region_elevations, range(num_months)):
            hydrostatic_from_rising_sea = self.pressure_coordinates_at(
                elevation=elevation_frame,
                reference_head=self.air_pressure_record[month],
                reference_elevation=self.sea_level_record[month],
                density= self.saltwater_density
            )
            wetted_pressure_record.append(hydrostatic_from_rising_sea)
        return wetted_pressure_record
    
    @property
    def post_spinup_surf_pressure_profile_record(self):
        num_months = int(self.POST_SPINUP_DURATION.to(ureg.month).magnitude)
        region_elevations = [self.id_to_nz(region_frame) * self.dz + self.origin[1] for region_frame in self.region_post_spinup_surf]
        surf_pressure_record = []
        air_pressure_record = self.air_pressure_record
        sea_level_record = self.sea_level_record
        for elevation_frame, month in zip(region_elevations, range(int(num_months))):
            hydrostatic_from_rising_sea = self.pressure_coordinates_at(
                elevation=elevation_frame,
                reference_head=air_pressure_record[month],
                reference_elevation=sea_level_record[month],
                density= self.freshwater_density
            )
            surf_pressure_record.append(hydrostatic_from_rising_sea)
        return surf_pressure_record
    # endregion

    @property
    def post_spinup_sea_pressure_profile_record(self):
        num_months = int(self.POST_SPINUP_DURATION.to(ureg.month).magnitude)
        region_elevations = self.nz_to_elevation(self.id_to_nz(self.region_sea))
        hydrostatic_from_rising_sea = self.pressure_coordinates_at(
            elevation=region_elevations,
            reference_head=np.expand_dims(self.air_pressure_record, axis =1),
            reference_elevation=np.expand_dims(self.sea_level_record, axis=1),
            density= self.saltwater_density
        )
        return hydrostatic_from_rising_sea


    # region Recharge assignment
    def assign_steady(self, value, region, duration): 
        inner_length = len(region[0])
        assert all(len(inner) == inner_length for inner in region), \
            "Region must be a fixed (non-time varying) region"
        if isinstance(value, list):
            num_months = duration.to(ureg.month).magnitude
            return np.tile(value, (num_months, 1))
        elif isinstance(value, (int, float)):
            return np.tile(value, (num_months, len(region)))

    @property
    def spinup_recharge_profile_record(self): 
        return self.assign_steady(self.region_spinup_recharge, self.recharge, self.SPINUP_DURATION)
    
    @property
    def post_spinup_recharge_profile_record(self): 
        return self.assign_steady(self.region_post_spinup_recharge, self.recharge, self.POST_SPINUP_DURATION)
    # endregion

    # region Transport conditions
    @property
    def post_salinity_record(self): 
        num_years = int(self.POST_SPINUP_DURATION.to(ureg.year).magnitude)
        return np.tile(self.annual_salinity, num_years)

    @property
    def spinup_salinity_record(self): 
        num_months = int(self.SPINUP_DURATION.to(ureg.month).magnitude)
        return np.full(num_months, self.SPINUP_SALINITY)