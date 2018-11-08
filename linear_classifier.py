# This is a linear classifier for the iris classification
# dataset available at https://archive.ics.uci.edu/ml/datasets/iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Train and test the model using 50 different partitions of the data to get
# an accurate estimate of its accuracy. The data will be split into an
# 80% training - 20% testing ratio.
model_accuracy = 0
accuracies = np.zeros([50,1])
for dataset in range (0, 50):
    Xtr, Xte, Ytr, Yte = train_test_split(data, labels, train_size=0.8, test_size=0.2)

    # Train and test the model. To do this we have to get the probability of
    # the testing data belonging to class 0 against the probability of it
    # belonging to any other class and repeat for the other two classes. The
    # highest individual probability will decide the label assigned to the
    # prediction. To represent this scenario we change the labels so 1 means
    # it belongs to the class and -1 means it doesn't (therefore belongs to
    # any of the other two).
    class_probability = np.zeros([30,3])
    weights = np.zeros([4,1])
    predictions = np.zeros([30,1])
    for class_label in range (0,3):
        Ytr_mod = np.where(Ytr!=class_label,-1,1)

        # Calculate the optimal weights by using the normal function derived
        # from minimising the least squares error function. To avoid
        # unnecessary calculations, this will be computed using the
        # Moore-Penrose inverse or pseudoinverse.
        weights = np.matmul(np.linalg.pinv(Xtr), Ytr_mod)

        # Test the weights with the testing data.
        class_probability[:,class_label] = np.matmul(Xte, weights)

    # Finally, assign a label depending on the maximum probability among the
    # three each sample has.
    correct_predictions = 0
    predictions = np.argmax(class_probability, axis=1)
    for sample,value in np.ndenumerate(predictions):
        if predictions[sample] == Yte[sample]:
            correct_predictions += 1
    accuracies[dataset,0] = correct_predictions / 30

del sample, value, class_label, dataset, correct_predictions, class_probability

# Compute model accuracy.
model_accuracy = np.mean(accuracies)
print('Model accuracy (tested on 50 partitions of data): ', model_accuracy)
