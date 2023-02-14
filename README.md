# n_track_ML


Python scripts for microscopy image analysis. 

Takes 4D images of chromosome loci in live cells as input. Next, images are segmented, chromosome loci are detected and tracked to generate morphology-based and motility-based features. 
Collected data is used to train the machine learning models to distinguish different experimental conditions (whether the cells are in serum or starved). Finally, the explanation of ML models output is used to extract biological meaning from imaging-derived data.

* [tracking.py](https://github.com/redchuk/n_track_ML/blob/master/to_become_public/tracking.py)    image segmentation, registration and particle tracking
* [feature_engineering.py](https://github.com/hajaalin/n_track_ML/blob/master/to_become_public/feature_engineering.py)    generate aggregated features
* [GBC.py](https://github.com/hajaalin/n_track_ML/blob/master/to_become_public/GBC.py)    baseline performance, GBC, SHAP explanation
* [MLP.py](https://github.com/hajaalin/n_track_ML/blob/master/to_become_public/MLP.py)    NN trained with aggregated features, SHAP explanation
* [InceptionTime](https://github.com/hajaalin/n_track_ML/tree/code_review/scripts/tsc)    InceptionTime trained with raw data, SHAP explanation
* [data_47091baa.csv](https://github.com/hajaalin/n_track_ML/blob/master/to_become_public/tracking_output/data_47091baa.csv)    master dataset 

## Requirements

- numpy~=1.19.5
- tensorflow~=2.6.0
- pandas~=1.2.4
- scikit-learn~=1.0.2
- keras~=2.6.0
- scipy~=1.7.0
- PyYAML~=6.0
- Jinja2~=3.0.2
- shap~=0.40.0
- dabest~=0.3.1
- seaborn~=0.11.2
- matplotlib~=3.4.2
- trackpy~=0.5.0
- scikit-image~=0.19.3
- pystackreg~=0.2.6.post1
- apeer_ometiff_library~=1.10.1
