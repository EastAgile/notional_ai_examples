# Notional AI - Exogenous Variables Feature Selection

This repository contains the necessary files for the exogenous variables feature selection on various examples. The goal of this project is to identify the most relevant features from the exogenous variables in the dataset and evaluate the performance across various time series forecasting models, using data from [Notional API](https://notional.ai/).

## Files

The following files are included in this repository:

1. **utils.py**: This file contains utility functions that are used in other scripts for data preprocessing and feature engineering.

2. **requirement.txt**: This file lists the dependencies required to run the project. Make sure to install these dependencies before executing the code.

3. **bike_sharing/demo_bike_dataset.ipynb**: This Jupyter Notebook provides a demonstration of the feature selection process on the bike dataset. It loads the processed data file (`bike_sharing/data/bike_sharing_day.csv`), applies feature selection techniques, and evaluates the selected features. You will also learn how to use the bulk API from Notional API to get the exogenous features.

4. **calculate_feature_score.py**: This script is designed to calculate scores for a batch of features. It is placed in a separate file to support multiprocessing, allowing for efficient computation.

5. **feature_selection.py**: This script is responsible for executing the feature selection process. Within the main training loop, it calls the `calculate_feature_score` function from `calculate_feature_score.py`. The feature selection technique employed is metric-based, wherein it evaluates the features based on a specific metric (scoring function). It will return a ranked list of selected features subset that has been deemed most relevant for the dataset.

## Usage

To use this project, follow these steps:

Install the dependencies listed in the `requirement.txt` file using different package managers, you can follow the instructions below:

### Installing with pip

```shell
pip install -r requirement.txt
```

### Installing with conda

```shell
conda install --file requirement.txt
```

### Bike sharing dataset example
You can checkout the example on the bike sharing dataset:

1. Open the `bike_sharing/demo_bike_dataset.ipynb` notebook in Jupyter Notebook.

2. Execute the cells in the notebook sequentially to see the step-by-step process of feature selection on the bike dataset.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


<!-- ## License

This project is licensed under the [LICENSE](LICENSE). You are free to use, modify, and distribute the code in this repository for personal or commercial purposes. -->
