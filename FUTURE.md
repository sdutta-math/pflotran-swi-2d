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

# Future Branches

## MasterRNG
Modify the ensemble class to universally use the same seed for all the random draws, so it can be reproducible. Done — `_llnl_random_walk` now uses a local `RandomState(self.master_rng())` instead of numpy's global RNG; the scipy `.rvs()` draws (salinity, air pressure, water table, recharge) now pass `random_state=self.master_rng()`. Caveat documented in code: reproducibility depends on draw order, since `master_rng()` hands out sequential values from one stream.
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