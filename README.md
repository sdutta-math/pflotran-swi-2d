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
TBD — pending a compatibility check against PFLOTRAN's own license. No LICENSE file yet; do not treat this repo as licensed for reuse until one is added.