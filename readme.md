# IR Sensor Characterization Capstone Project

## Table of Contents
- [Directory Structure](#directory-structure)
- [Directories Explained](#directories-explained)
  - [data](#data)
  - [models](#models)
  - [overallModels](#overallmodels)
  - [singleLabelObjectDetection](#singlelabelobjectdetection)
  - [utils](#utils)

## Directory Structure 
<sub>Additional files in tuning_dir directories are omitted for clarity</sub>
```
IR-Sensor-Characterization
├── data
│   ├── differentSizesData
│   │   ├── differentSizesData.csv
│   │   ├── differentSizesData2.csv
│   │   ├── testDifferentSizesData.csv
│   │   └── testDifferentSizesDataShuffle.csv
│   ├── newData
│   │   ├── blackCube.csv
│   │   ├── blackShoe.csv
│   │   ├── blackWaterBottle.csv
│   │   ├── combinationOfAllData.csv
│   │   ├── paperTowelRoll.csv
│   │   ├── starCup.csv
│   │   ├── whiteCube.csv
│   │   ├── whiteCubeEdge.csv
│   │   ├── whiteShoe.csv
│   │   ├── woodenCube.csv
│   │   └── woodenCubeEdge.csv
│   ├── multi_object_data.csv
│   └── pca_combinationOfAllData.csv
│
├── models
│   ├── rnn.keras
│   ├── rnn_pca_model.joblib
│   └── single_label.keras
│
├── overallModels
│   ├── overallModel.py
│   └── testModel.py
│
├── regressionneuralnetwork
│   ├── tuning_dir
│   ├── createModel.py
│   └── validateModel.py
│
├── singleLabelObjectDetection
│   ├── tuning_dir
│   └── createModel.py
│
├── utils
│   ├── finalVisualization
│   │   ├── iRobot.png
│   │   └── visualizationFinal.py
│   ├── create_heatmap.py
│   ├── getData.py
│   └── pca.py
│
├── LICENSE
└── readme.md
```

## Directories Explained

### data
- This folder contains two subfolders—differentSizesData and newData. newData is the main dataset that the team used to
  train the regression neural network (not exactly, see next paragraph for more details). Each object included in the dataset
  has their own separate files (blackCube, blackShoe, etc...). combinationOfAllData.csv is the combination of all of the data
  from the individual objects put into one CSV file.

  You will also see pca_combinationOfAllData.csv. This is the dataset after applying a PCA transform on the dataset and it's the
  data that was acutally used to train the Regression Neural Network. They are separeted because training the model with
  the PCA transformed data isn't exactly necessary, but the team did see improvements in model performance after pca was
  applied.

- multi_object_data.csv contains the data used to train the Single Label Neural Network. Like, the Regression Neural Network,
  pca_multi_object_data.csv contains the data that's actually used to train the Single Label Neural Network.

- More detail on the pca transform script can be found in [utils](#utils)
  

### models
- Description of the models folder...

### overallModels
- Description of the overallModels folder...

### singleLabelObjectDetection
- Description of the singleLabelObjectDetection folder...

### utils
- Description of the utils folder...
