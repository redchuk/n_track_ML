# Python environment for the time-series classification part

```bash
# install Mamba
conda install -n base conda-forge::mamba

mamba create -n tsc_jupyter
conda activate tsc_jupyter
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit>=11.2 jupyterlab keras matplotlib numpy pandas pip scikit-learn sktime==0.10.0 -c conda-forge -vvv
pip install scikeras
```

```
mamba create -n tsc
conda activate tsc
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit==11.2 keras numpy pandas pip scikit-learn scipy==1.4.1 sktime==0.10.1 pyyaml -c conda-forge
pip install scikeras
pip install click
```

```
mamba create -n tsc_jupyter2
source activate tsc_jupyter2
CONDA_CUDA_OVERRIDE="11.2" mamba install tensorflow==2.7.0 cudatoolkit==11.2 keras numpy pandas pip scikit-learn scipy==1.4.1 sktime==0.10.1 pyyaml jupyterlab matplotlib -c conda-forge
pip install scikeras
pip install click
python -m ipykernel install --user --name tsc_jupyter2 --display-name "Python (tsc_jupyter2)"
```

```
mamba create -n jinja2
conda activate jinja2
conda install -c anaconda jinja2
```
