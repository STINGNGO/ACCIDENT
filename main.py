# read the  dataset
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Accident_Dataset_1_prepared.csv")
data.head(3)
print(data.head(3))

# check shape of data and print rows and columns
print("Data has %d rows and %d columns" % data.shape)

# check the content of data table
print(data)
print(data.tail(5))

# select column from data
print(data.columns)
print(data["Number of Vehicles"])

# extract multiple columns fro dataset
print(data[["Number of Vehicles", "Road Surface", "Lighting Conditions", "Type of Vehicle"]])

# extract unique values from columns
print(data["Number of Vehicles"].unique())

# create graphs in python
print(data.plot("Number of Vehicles", kind="bar"))

print(plt.scatter(data["Number of Vehicles"], data["Casualty Severity"]))
print(plt.title("Compare number of vehicles with casualty severity"))
print(plt.xlabel("Number of Vehicles(Independent)"))
print(plt.ylabel("Casualty Severity(Dependent)"))

# mathematical operations on Number of vehicles
print("Max number of vehicles", data["Number of Vehicles"].max())
print("Min number of vehicles", data["Number of Vehicles"].min())
print("Unique number of vehicles", sorted(data["Number of Vehicles"].unique()))
print("Mean number of vehicles", round(data["Number of Vehicles"].mean()))
print("Count the rows in number of vehicles", data["Number of Vehicles"].count())
print(data["Number of Vehicles"].describe())
print(data.corr())
