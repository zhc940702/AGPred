# AGPred

# Requirements
* python == 3.7.3
* pytorch == 1.11.0
* Numpy == 1.21.5
* torch_geometric == 2.1.0
* scikit-learn == 0.24.1
* pandas == 1.2.3
* transformers == 4.27.3
* deepchem == 2.6.1
* rdkit == 2022.9.5


# Files:
1.Data

In the `dataset/raw` folder, there are data files named in the format `benchmark_train_fold_<number>.csv` and `benchmark_test_fold_<number>.csv`, which constitute the 10-fold data for the benchmark dataset.

2.Code
### `model.py`
This code file contains various neural network models used in the study.

### `main_with_prop.py`
This code file includes data processing, model training, and prediction, as well as the evaluation of model performance.

### `cal_property.py`
This code file is used to calculate the 53-dimensional descriptors of molecules.

### `prop_transformer.py`
This code file contains the transformer encoder for embedding and feature extraction of the 53-dimensional descriptors.

### `config.py`
This code file contains the parameter configurations for the models.


# Train and test folds
You can directly run the sample code to perform the 10-fold cross-validation experiment on the benchmark dataset.

Example:
```bash
python main_with_prop.py
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn
