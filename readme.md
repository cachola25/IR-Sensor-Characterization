# IR Sensor Characterization Capstone Project
## Directory Structure <sub>Additional files in tuning_dir directories are omitted for clarity</sub>

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