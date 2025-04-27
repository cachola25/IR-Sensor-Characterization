# IR Sensor Characterization Capstone Project

## Table of Contents
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Directories Explained](#directories-explained)
  - [data](#data)
  - [models](#models)
  - [overallModels](#overallmodels)
  - [Regression and Single Label Models](#regression-and-single-label-models)
  - [utils](#utils)

## Installation

Follow these steps to set up the project environment:

### 1. Clone the Repository

```bash
git clone https://github.com/cachola25/IR-Sensor-Characterization.git
cd IR-Sensor-Characterization
```

### 2. Install Python 3.12

Ensure you have **Python 3.12.x** installed.

You can check your installed Python version with:

```bash
python3 --version
```

If Python 3.12 is not installed, download it from [python.org](https://www.python.org/downloads/).

> **Important:**  
> This project was developed and tested on Python 3.12.  
> Python 3.13+ is newer and **may not be fully supported** by all dependencies yet.
> 
> - If `python3 --version` shows Python 3.13 or higher, try checking if Python 3.12 is also installed by running:
> 
>   ```bash
>   python3.12 --version
>   ```
> 
> - If Python 3.12 is available, **edit the `Makefile`** and replace all instances of `python3` with `python3.12` **before** running any setup commands.
> 
> - If Python 3.12 is not installed, and your system supports it, please install Python 3.12 to proceed.
### 3. Ensure `venv` is Installed

Most systems already have the `venv` module. 
Check `python3 -m venv --help` to see if the `venv` module is found

If not:

- **Ubuntu/Debian**:

  ```bash
  sudo apt install python3-venv
  ```

- **macOS/Windows**: `venv` is usually included automatically.

### 4. Set Up the Virtual Environment and Install Dependencies

Use the provided `Makefile` to automate setup:

```bash
make setup
```

This will:
- Create a new virtual environment named `venv/`
- Activate the environment temporarily
- Install all project dependencies listed in `requirements.txt`

### 5. Activate the Virtual Environment (for new terminal sessions)

Every time you open a new terminal, activate the environment:

- **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

- **Windows (PowerShell)**:

  ```bash
  .\venv\Scripts\activate
  ```

You should see `(venv)` appear at the start of your terminal prompt.

### 6. Run the Project

With the virtual environment activated, you can now run any project scripts:

```bash
python3 path/to/your_script.py
```

Example:

```bash
python3 overallModels/testModel.py
```

### 7. (Optional) Clean Up the Environment

To delete the virtual environment and start fresh:

```bash
make clean
```

## Directory Structure
<sub>Additional files in tuning_dir and data directories are omitted for clarity</sub>
```
IR-Sensor-Characterization
├── data
│   ├── differentSizesData
│   ├── newData
│   ├── multi_object_data.csv
│   ├── pca_combinationOfAllData.csv
│   └── pca_multi_object_data.csv
│
├── models
│   ├── rnn.keras
│   ├── rnn_pca_model.joblib
│   ├── single_label.keras
│   └── single_label_pca_model.joblib
│
├── overallModels
│   ├── overallModel.py
│   └── testModel.py
│
├── regressionneuralnetwork
│   ├── tuning_dir
│   └── createModel.py
│
├── singleLabelObjectDetection
│   ├── tuning_dir
│   └── createModel.py
│
├── utils
│   ├── finalVisualization
│   ├── create_heatmap.py
│   ├── getData.py
│   └── pca.py
│
├── LICENSE
├── makefile
├── readme.md
└── requirements.txt
```
## Directories Explained

### data
- Contains `differentSizesData` and `newData`.
- `newData` holds the primary datasets used for model training, separated by object type.
- `combinationOfAllData.csv` combines all individual datasets.
- `pca_combinationOfAllData.csv` is the PCA-transformed version used for training the Regression NN.
- `multi_object_data.csv` and its PCA variant serve the Single Label NN.
- More details about PCA can be found in the [utils](#utils) section.

### models
- Stores the final, trained models: the Regression NN, Single Label NN, and the PCA transformers.
- Ensures consistent data transformations during real-world predictions.

### overallModels
- Contains:
  - **overallModel.py**: Defines the `overallModel` class which automatically loads the Regression NN and Single Label NN.
    - Contains a single `predict()` method that accepts raw IR values.
  - **testModel.py**: A testing script for experimenting with the `overallModel` class.

### Regression and Single Label Models
- Combines `regressionneuralnetwork` and `singleLabelObjectDetection` directories.
- Each contains:

  - **createModel.py**
    - Builds and trains either the Regression or Single Label NN.
    - Saves the trained model (`rnn.keras` for regression, `single_label.keras` for single label) to the `models/` directory.

  - **tuning_dir**
    - Stores hyperparameter tuning results using [Keras Tuner](https://keras.io/keras_tuner/).
    - Contains the best model architectures.
    - To retune, delete `tuning_dir` or set `overwrite=True` in the tuning script.

- The training pipeline is identical for both model types.

### utils
- Contains utility scripts for data processing and visualization:
  - **create_heatmap.py**: Creates a heatmap of the RNN's prediction errors across the polar grid.
  - **getData.py**:
    - Prompts to collect data for either the Regression NN (0) or Single Label NN (1).
    - Saves output CSVs to the `data/` directory.
    - You must manually set the output file name inside the script.
  - **pca.py**:
    - Applies PCA to either the Regression NN or Single Label NN datasets.
    - You must manually set the input file in the script.
  - **finalVisualization/visualizationFinal.py**:
    - Visualizes predictions:
      - No cone = 0 objects.
      - Red cone = 1 object.
      - Red + Blue cones = 2 objects.
    - WARNING: May not be fully compatible with Windows systems.
