# Time-series classification on Turso cluster

## Modules
Conda is installed as a user module in ```/proj/hajaalin/Miniconda3/4.11.0/```.
See https://github.com/UH-LMU/Ukko2-settings. The module is loaded in sbatch scripts.

## Python environments

```bash
# install Mamba
conda install -n base conda-forge::mamba
```

This is the main environment that is activated in sbatch scripts.
```bash
mamba create -n tsc
conda activate tsc
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit==11.2 keras numpy pandas pip scikit-learn scipy==1.4.1 sktime==0.10.1 pyyaml -c conda-forge
pip install scikeras
pip install click
```

Jinja2 templating is used for creating sbatch scripts.
```
mamba create -n jinja2
conda activate jinja2
conda install -c anaconda jinja2
```

Jupyter environments can be used with interactive notebooks.
```bash
mamba create -n tsc_jupyter
conda activate tsc_jupyter
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit>=11.2 jupyterlab keras matplotlib numpy pandas pip scikit-learn sktime==0.10.0 -c conda-forge -vvv
pip install scikeras
```
```bash
mamba create -n tsc_jupyter2
source activate tsc_jupyter2
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit==11.2 keras numpy pandas pip scikit-learn scipy==1.4.1 sktime==0.10.1 pyyaml jupyterlab matplotlib -c conda-forge
pip install scikeras
pip install click
python -m ipykernel install --user --name tsc_jupyter2 --display-name "Python (tsc_jupyter2)"
```

## Workflow

### Create sbatch scripts
```bash
source ~/.bashrc
conda activate jinja2

python create_sbatch.py --job_name le_norm2_k20_f1 --loop_epochs 2 50 2 --options "--fset f_mot --kernel_size=20 --repeats=30" --sbatch_dir sbatch/le_norm2_k20_step2b --paths /proj/hajaalin/Projects/n_track_ML/scripts/tsc/paths.yml

```

### Submit sbatch scripts
```bash
# repeat until no python environment is active
# (an active environment will mess up environment variables sent with sbatch)
conda deactivate 

for s in $(ls sbatch/le_norm2_k20_step2b/*.sh); do sbatch $s; done;

```

