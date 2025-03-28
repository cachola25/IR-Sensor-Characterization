import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv("./test3.csv", header=None)
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
combined_df.to_csv("./pca_test_data.csv", index=False, header=None)