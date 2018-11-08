# This is a k-nearest neighbour classifier for the iris classification
# dataset available at https://archive.ics.uci.edu/ml/datasets/iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# The first step is to import the .csv dataset.
csv_data = pd.read_csv("data.csv")

# It should now be split into data and labels.
data = csv_data.iloc[:, 0:-1].values
labels = csv_data.iloc[:, 4].values

# Normalising the data.
normaliser = StandardScaler()
data = normaliser.fit_transform(data)

# Now encode the categorical data inside labels so it can be used by the model.
labelsencoder = LabelEncoder()
labels = labelsencoder.fit_transform(labels)


# Compute the accuracies of k values from 1 to 30 on 100 different datasets
# to determine the optimal k value.
best_k = 0
k_accuracies = np.zeros((100,30),dtype=float)
k_accuracies_avg = np.zeros((1,30),dtype=float)
for dataset in range (100):
    Xtr, Xte, Ytr, Yte = train_test_split(data, labels, train_size=0.8, test_size=0.2)
    for k in range (1, 31):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(Xtr, Ytr)
        predictions = classifier.predict(Xte)
        correct = 0
        for row,prediction in np.ndenumerate(predictions):
            if prediction == Yte[row]:
                correct += 1
        accuracy = correct/30
        k_accuracies[dataset,k-1] = accuracy
del dataset,k,accuracy,row,prediction,correct

k_accuracies_avg[0,:] = np.average(k_accuracies, axis=0)
best_k = np.argmax(k_accuracies_avg) + 1


# Test the model with the optimal k.
# To mesure the accuracy of the model we perform 15 such splits of the data
# and average their respective accuracies.
accuracy_array = np.zeros((50,1),dtype=float)
for iteration in range (1, 51):
    Xtr, Xte, Ytr, Yte = train_test_split(data, labels, train_size=0.8, test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=best_k)
    classifier.fit(Xtr, Ytr)
    predictions = classifier.predict(Xte)
    correct = 0
    for row,prediction in np.ndenumerate(predictions):
        if prediction == Yte[row]:
            correct += 1
    accuracy_array[iteration-1] = correct/30
    print('Dataset ', iteration)
    print(classification_report(Yte, predictions))
del row,prediction,correct

model_accuracy = np.average(accuracy_array)
print('Best k (tested on 100 partitions of data): ', best_k)
print('Model accuracy (tested on 50 partitions of data): ', model_accuracy)
