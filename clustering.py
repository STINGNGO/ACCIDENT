import pandas as pd

data = pd.read_csv("Accident_Dataset_1_prepared.csv")
from sklearn.cluster import _feature_agglomeration

ag_cluster = _feature_agglomeration(n_clusters=3)

# Train the algorithm = calculate attribute on the basis of dataset
print(ag_cluster.fit(data))
