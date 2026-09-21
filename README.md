# pflotran-swi-2d

Property-based PFLOTRAN model generator for 2D seawater intrusion (SWI)
simulation in coastal aquifers, parameterized for a Norfolk, VA transect.

## Overview

`pflotran-swi-2d` separates the *specification* of a coastal aquifer model
from its *representation* as a [PFLOTRAN](https://www.pflotran.org/) input
deck. A `NorfolkModel` describes an idealized 2D transect — grid geometry,
land/ocean elevation profiles, permeability/porosity fields, hydrostatic
boundary conditions, salinity, and a spinup + post-spinup (sea-level-rise)
period — as plain, unit-aware Python properties. A separate writer module
(`pfwrite`) translates a `NorfolkModel` instance into the PFLOTRAN files
needed to actually run it (region/strata HDF5, permeability/porosity
datasets, pressure and salinity boundary time series, and control decks).

This repo does not build, link against, or bundle PFLOTRAN itself — it only
produces the files PFLOTRAN reads. The user runs PFLOTRAN separately against 
the generated deck.

A `NorfolkEnsemble` draws batches of randomized `NorfolkModel` realizations
(swapping in different subsurface random fields, elevation profiles,
salinity, recharge, etc.) for generating training data across many
plausible aquifer configurations.

## Features

- Rectangular super-grid with a 2D subsurface domain mask
- Land/ocean elevation profiles specified independently, joined at a
  shelf-slope break
- Freshwater water table (left/creek boundary), wetted top-boundary
  hydrostatic condition, far-field right-boundary pressure condition
- Correlated random permeability/porosity fields (via
  [`gstools`](https://github.com/GeoStat-Framework/GeoStat-Framework))
- Spinup period (steady mean sea level and salinity) followed by a
  post-spinup period with sea-level rise and annual salinity variation
- Units enforced throughout via [`pint`](https://pint.readthedocs.io/)
- Ensemble sampling for generating batches of randomized models

## Installation

```bash
conda env create -f norfolkenv.yml
conda activate norfolk
pip install -e .
```

## Quick start

```python
from pflotran_swi import norfolk_model as nm
from pflotran_swi import norfolk_ensemble as nensemble
from pflotran_swi import pfwrite as pfw
from pflotran_swi.units import ureg

# Define a template model
template_model = nm.NorfolkModel(
    model_name="template",
    lx=1100.0 * ureg.meter,
    lz=20.0 * ureg.meter,
    nx=440,
    nz=200,
)

# Draw a batch of randomized realizations
ensemble = nensemble.NorfolkEnsemble(NorfolkModel=template_model)
samples = ensemble.draw(3)

# Write out PFLOTRAN input decks for each realization
for realization in samples:
    pfw.write(realization, f"./work/{realization.model_name}")
```

See `notebooks/sampling_training_data.ipynb` for a fuller walkthrough,
including unit enforcement, model validation, and functional-style
elevation/pressure transforms.

## Project structure

```
src/pflotran_swi/
    norfolk_model.py     # NorfolkModel: model specification
    norfolk_ensemble.py  # NorfolkEnsemble: randomized sampling
    pfwrite.py            # NorfolkModel -> PFLOTRAN input deck
    pfctrl/               # Bundled example PFLOTRAN control decks
    tests/                 # pytest suite + reference fixtures
```

## Testing

```bash
pytest src/pflotran_swi/tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Citing

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff),
or use GitHub's "Cite this repository" button. Plain-text form:

> Wu, K., Dutta, S., Dwivedi, D., Farthing, M., & Dawson, C. (2026). 
> *pflotran-swi-2d* (Version 0.1.0) [Computer software].
> https://github.com/sdutta-math/pflotran-swi-2d

## License

BSD 3-Clause — see [LICENSE](LICENSE).

This repo only generates PFLOTRAN input decks (text/HDF5 files); it does not
link against or redistribute PFLOTRAN itself, so it isn't bound by PFLOTRAN's
own LGPLv3 license. PFLOTRAN is a separate, independently-installed
dependency you run against the generated decks.
