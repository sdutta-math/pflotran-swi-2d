List of features that I don't have time to work on right now. 


<font color="red">Great to have</font> 
<font color="green">Helps to Have</font> 
<font color="lightgrey">Want to Have</font> 
<font color="yellow" >Just a thought</font>

# Stale Branches 

## Packaging 
Structure the repo as a package so that external notebooks may use it. Done — pyproject.toml (hatchling), editable install, importlib.resources-bundled pfctrl templates.

# Growing Branches 

## Input deck data structure 
Add a data structure to represent a pflotran input deck so it naturally interacts with pfwrite. Probably some sort of nested dictionary or tree class, with each node having a name, field, children. 
## Self-Descriptive Data 
Refactor NorfolkModel to export entirely DataArrays to further reinforce Model-Writer independence (such as being indexed by elevation). Add metadat to indicate normal directions for boundary definitions
## Electric Fence 
Improve error catches 
## Point Extraction Wells
Add point extraction wells to the domain (NorfolkModel has no well/extraction/pumping attributes, NorfolkEnsemble has no draw method for one). A scaffold exists but is disabled: `pfctrl/spinup/pflotran.in` and `pfctrl/post/pflotran.in` both have a commented-out `#EXTERNAL_FILE source_sink.txt` under `extraction wells`, no `source_sink.txt` present, and `pfwrite.py` has a bare `# Well stuff here` stub in `write_post_run`. Checked SWINet_dev's abstractify, extended, hotfix, and metadata branches for a prior implementation - same disabled scaffold and stub on all of them, no working feature found anywhere.
## Elevation Profile Resolution Coupling
`NorfolkEnsemble._llnl_random_walk` and `_llnl_smooth_profile` bound the land/shelf elevation walk with absolute cell counts (`dhz=50`, the `+-3` per-step bound, `d_smooth=10`), none scaled by dx/dz. Changing grid resolution (nx/nz) alone changes the taper proportion, the max realistic terrain slope, and the smoothing length in physical units -- so the generated profile ensemble's statistical character is resolution-coupled, not resolution-invariant. Also assumes nz > dhz. FIXMEs left in code at each constant; not fixed yet, would need dhz/+-3/d_smooth expressed in meters and converted via dx/dz instead of raw cell counts.
## Dry Surf Cells in the Coastal Region
Post-spinup, `Coastal` is a static PFLOTRAN region covering every top cell below the *terminal* sea level (MSL + `sea_level_anomaly_rate` * `POST_SPINUP_DURATION`, 0.3 m by default), because PFLOTRAN cannot change a region over time. Cells between MSL and the terminal level ("surf", `region_post_spinup_surf`) are still land at the start. They are dry until the rising sea reaches them, but they get (1) the Coastal Dirichlet liquid-pressure condition (freshwater-density hydrostatic below the current sea level, i.e. mild suction, e.g. -981 Pa for a cell 0.1 m above the sea) instead of the `Recharge` flux, and (2) the `Saltwater` transport condition of the whole `Coastal` boundary instead of `Freshwater`. Consequences: no recharge on that land for the entire post-spinup run, even before it floods (up to ~51 of ~92 coastal cells in draws with flat low-lying coast; 0 to 51 across 60 default-ensemble draws), and seawater salinity imposed on cells that are not under seawater. Inherited from SWINet_dev (`Top_Sea` region, static at the terminal level, same transport condition), so the archived ensemble has the same approximation. SWINet_dev's `Top` region additionally put the recharge flux on *all* top cells including the sea, which this package no longer does.

Not fixed because it cannot be validated without running PFLOTRAN. To analyze: (a) how much recharge is lost (surf cell count * dx * recharge rate) relative to total recharge, per realization; (b) sensitivity of the intrusion metrics to giving the surf cells recharge, i.e. re-running a subset with a separate `Surf` region (flux + `Freshwater`) that is fixed to the cells above the *current* sea level at some coarse time step; (c) whether the salinity imposed on dry surf cells matters, since they would be unsaturated and mostly outside the wedge. Candidate fixes: run post-spinup in restarted segments (e.g. yearly, `RESTART` from the previous checkpoint) with `Surf`/`Coastal` regions redefined per segment; or a PFLOTRAN flow-condition type that switches between flux and pressure by saturation (seepage-type; needs checking against the PFLOTRAN docs and a test run). Any change breaks comparability with the archived SWINet_dev cases, so decide that alongside re-generating the training data.

# Future Branches

## MasterRNG
Modify the ensemble class to universally use the same seed for all the random draws, so it can be reproducible. Done — `_llnl_random_walk` now uses a local `RandomState(self.master_rng())` instead of numpy's global RNG; the scipy `.rvs()` draws (salinity, air pressure, water table, recharge) now pass `random_state=self.master_rng()`. Caveat documented in code: reproducibility depends on draw order, since `master_rng()` hands out sequential values from one stream.
## Subsurface Field Units and Length Scale
`_draw_subsurface_field` passed bare cell-index arrays (`np.arange(nx)`, `np.arange(nz)`) into `gstools.SRF.set_pos`, so `len_scale=[25,20]` was interpreted in *cells*, not meters. With `dx=5 m, dz=0.1 m` that produced a ~36:1 physical anisotropy (~90 m x ~2.5 m, empirically measured) instead of the intended near-isotropic 25x20 m field -- the root cause of the permeability field looking needle-thin/horizontally stratified rather than the mottled, patchy texture a 2D GRF should give. Verified the same bug (bare index arrays, no dx/dz scaling) exists in the original `notebooks/pflotran/generate_reals.ipynb` on SWINet_dev's `extended` branch -- present since inception, not a refactor regression. Done — `idxs`/`idzs` now scaled by `dx.magnitude`/`dz.magnitude`. Also revisited the length-scale *value*: with correct units, `len_scale_z=20 m` is ~the entire `lz=20 m` domain height (no vertical structure at all), so the default was changed to `[25, 2]` m, sized to this domain's actual thickness.
## Ensemble Draw Attribute Mismatch
`NorfolkEnsemble._draw()` assigned `new_model.land_elevation_profile`, `new_model.ocean_elevation_profile`, `new_model.annual_salinity_record` — but `NorfolkModel` only ever reads `land_profile`, `shelf_profile`, `annual_salinity`. The elevation random walk and salinity draw ran, computed real values, and were silently discarded on every realization; every case fell back to the template's deterministic default terrain (a straight `linspace`) and constant non-seasonal salinity. Found because freshly-drawn realizations for Figure 2b looked suspiciously identical in terrain. Done — renamed to the correct attributes; `_draw_land_profile`/`_draw_ocean_profile` also had to be fixed to walk `land_nx`/`shelf_nx` columns (400/40) instead of the full domain `nx` (440), since `land_profile`/`shelf_profile`'s length assertion would otherwise reject the (previously-unreachable) result. Verified across 5 draws: `land_profile` and `annual_salinity` now differ per realization.
## Consistency of Model
Make sure elevation_to_nz and inverse method are used everywhere applicable
## Unit Layer
Use an enforcement layer instead of ureg everywhere to improve code brevity
## Default Subclass
Remove declaration of default instance from init, makes it kind of awkward when testing. Maybe favor a subclass or setdefault() method.
## Broadcasting Rules
Clarify how these are interpreted internally 
## Live Property Optimizations
Use a setattr layer to recalculate derived properties on demand, since invoking a property over and over may be unintentionally slow. 
## Minor Refactoring
* Shorten 'post_spinup' to 'post' 
* Change 'nz' suffix to more correct 'iz' or 'idx' suffix 
* Double check salinity units