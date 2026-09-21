## Generating a training dataset
### Setup
From the root: 
```bash
conda env create -f norfolkenv.yml
conda activate norfolk
pip install -e .
```
### Examples 
See `notebooks/sampling_training_data.ipynb` for an example workflow

### Running tests
```bash
pytest src/pflotran_swi/tests
```

## License
BSD 3-Clause — see [LICENSE](LICENSE).

This repo only generates PFLOTRAN input decks (text/HDF5 files); it does not
link against or redistribute PFLOTRAN itself, so it isn't bound by PFLOTRAN's
own LGPLv3 license. PFLOTRAN is a separate, independently-installed
dependency you run against the generated decks.