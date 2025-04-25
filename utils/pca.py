import os
import pandas as pd
from sklearn.decomposition import PCA

REGRESSION_MODEL = "0"
SINGLE_LABEL_MODEL = "1"

def get_name(file_path):
    name, ext = os.path.splitext("pca_" + os.path.basename(file_path))
    base = name
    i = 1
    filename = base + ext
    while os.path.isfile(os.path.join(data_dir, filename)):
        filename = f"{base} ({i}){ext}"
        i += 1
    return filename

prompt = "Which model is this data for?\n" \
         "(0) Regression Neural Network\n" \
         "(1) Single-Label Object Model\n> "
model = input(prompt).strip()
while model not in {"0", "1"}:
    print("Invalid input. Please enter 0 or 1.")
    model = input(prompt).strip()
    
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")

if model == REGRESSION_MODEL:
    file_path = os.path.join(data_dir, "newData/combinationOfAllData.csv")
    df = pd.read_csv(file_path, header=None)
    df.columns = ['L3','L2','L1','M','R1','R2','R3','distance','start_angle','end_angle']
    sensor_cols = ['L3','L2','L1','M','R1','R2','R3']
    X = df[sensor_cols].values
    components = 7
    pca = PCA(n_components=components, random_state=42)
    pca_model = pca.fit(X)
    
    print(f"\n% of variance retained by {components} components: {pca_model.explained_variance_ratio_.cumsum()[-1].__round__(4)*100}%\n")

    X_pca = pca_model.transform(X)
    pc_df = pd.DataFrame(pca_model.components_, columns=sensor_cols, 
                         index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])])
    print(pc_df)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(components)])
    combined_df = pd.concat([pca_df, df[['distance','start_angle','end_angle']]], axis=1)
    filename = get_name(file_path)
    output_path = os.path.join(data_dir, filename)
    combined_df.to_csv(output_path, index=False, header=None)
else:
    file_path = os.path.join(data_dir, "multi_object_data.csv")
    df = pd.read_csv(file_path, header=None)
    df.columns = ['L3','L2','L1','M','R1','R2','R3','num_objects']
    sensor_cols = ['L3','L2','L1','M','R1','R2','R3']
    X = df[sensor_cols].values
    components = 7
    pca = PCA(n_components=components, random_state=42)
    pca_model = pca.fit(X)
    print(f"\n% of variance retaiend by {components} components: {pca_model.explained_variance_ratio_.cumsum()[-1].__round__(4)*100}%\n")
    X_pca = pca_model.transform(X)

    pc_df = pd.DataFrame(pca_model.components_, columns=sensor_cols, 
                        index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])])
    print(pc_df)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(components)])
    combined_df = pd.concat([pca_df, df[['num_objects']]], axis=1)
    filename = get_name(file_path)
    output_path = os.path.join(data_dir, filename)
    output_path = os.path.join(data_dir, filename)
    combined_df.to_csv(output_path, index=False, header=None)

