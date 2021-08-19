# read the dataset
import pandas as pd

data = pd.read_csv("Accident_Dataset_1_prepared.csv")
print(data.head(3))

independent_variables = data.columns
independent_variables = independent_variables.delete(5)
data1 = data[independent_variables]

# perform clustering to prepare a clustered dataset
from sklearn.cluster import AgglomerativeClustering

agg_cluster = AgglomerativeClustering(n_clusters=3)

# train the agg model
agg_cluster.fit(data1)

# predict the cluster label
data["Cluster Labels"] = agg_cluster.fit_predict(data1)

# select dependent and independent variables

Y = data["Casualty Severity"]
independent_variables = data.columns
independent_variables = independent_variables.delete(5)
X = data[independent_variables]

# classify using Gradient Boosted Tree
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

# train the model
gbc.fit(X, Y)

# predict using GBC Classifier
data["Predicted Casualty Severity"] = gbc.predict(X)

# Take a user input and predict casualty severity


independent_variables = independent_variables.delete(8)
user_input = {}
for var in independent_variables:
    temp = input("Enter" + var + ": ")
    user_input[var] = temp

# Calculate the cluster model
index = data1.shape[0]
user_df = pd.DataFrame(data=user_input, index=[index], columns=independent_variables)
data1 = pd.concat([data1, user_df], axis=0)  # add a new row in data1

data1.reset_index
# perform agglomerative clustering on dataset
# data1["Cluster Labels"] = agg_cluster.fit_predict(data1)
data1["Cluster Labels"] = agg_cluster.fit_predict(data1)

user_request = data1.tail(1)
severity = gbc.predict(user_request)
severity = severity[0]
if severity == 1:
    print("Casualty Severity is Slight (%d)" % severity)
elif severity == 2:
    print("Casualty is Severe (%d)" % severity)
elif severity == 3:
    print("Casualty Severity is Fatal (%d)" % severity)
