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
- Description of what data folder contains...

### models
- Description of the models folder...

### overallModels
- Description of the overallModels folder...

### singleLabelObjectDetection
- Description of the singleLabelObjectDetection folder...

### utils
- Description of the utils folder...
