# Python iris classifiers
Two different classifiers for the Iris dataset written in Python.

The dataset can be found in [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris) and has 150 
samples of iris with 4 features: 
  1. Sepal length (in cm)
  2. Sepal width (in cm)
  3. Petal length (in cm)
  4. Petal width (in cm)
  
Additionally, it provides a label for each entry to indicate it belongs to one of this classes:
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica

There are 50 samples for each class, with a line of data looking like this:
```
5.1,3.5,1.4,0.2,Iris-setosa
```

## Data conversion to .csv
The first step was to transform the raw data in the *iris.data* file into a .csv file which is can be imported
by pandas in Python. Fortunately, this can be done quite easily by importing the csv library. Only stripping and splitting have
to be done, since the features are conveniently separated by commas.

## k-nn classifier
I first implemented a k-nearest neighbour classifier. The way it operates is that for every testing sample, it calculates the 
distances between it and the rest of training elements. Then this distances are sorted and the most dominant label among the
first k elements (which will be, consequently, the closest to the testing element) is also assigned to the testing sample.

In order to train the model and test it afterwards we have to make a division of 'data' and 'labels' into training and testing. 
To do this, I have chosen to use the holdout method to randomly split the set. The split will be 80% data for training, 20% for 
testing. In the future it might be interesting to experiment with different values.

I though normalising the data would be a benefitial step for the model. Since k-nn computes the Euclidean distances to calculate 
the squared error, if we do not normalise the data and one of the features has a broad range of values, the distance will be governed 
by this particular feature.

After normalising I encoded the categorical data in the labels:
```
iris-setosa -> 0
iris-versicolor -> 1
iris-virginica -> 2
```
The final step before seeing how the model performs was to find the optimal value for k. I took a very basic approach for this,
which was comparing the performance of different possible k values in 100 splits of the data (using the same proportion that
would be use for the real testing, 80% - 20%) and averaging their performance.

And after this I could go straight to seeing how the model performed, in this case by using 50 new random splits of the data.

## Linear classifier
The same preprocessing that happened for the k-nn classifier was also applied to the data this time. However, the main concern 
in this situation is that I was facing non-linearly separable data. Because of this, I had to build a probabilistic model
that assigns a lable based on the probability of the testing sample belonging to each class against the other two:
```
->P(setosa|testing_sample) = p**
P(versicolor_or_virginica|testing_sample) = 100 - p

->P(versicolor|testing_sample) = p**
P(setosa_or_virginica|testing_sample) = 100 - p

->P(virginica|testing_sample) = p**
P(setosa_or_versicolor|testing_sample) = 100 - p
```
Where the probabilities marked with arrows indicate the ones we are interested in. The highest value among them will determine
the label we assign to the element we are testing.

The product of the model would be the weights, which I calculated using the normal equation that can be derived from optimising
the least square errors function:
```
weights = pseudoinverse(training_data) * training_labels
```
where training_labels have been replaced for each of the above cases by 1 if it's the class we are testing or -1 otherwise.

## Results
Surprisingly, the k-nn model performed quite a bit better than the linear one. It probably is due to the nature of the data itself that makes k-nn more suited for the job. It would be interesting to investigate the reason this is the case in the future, though. Perhaps the data is not linearly separable or can't be easily separated in a linear manner? 

On the other hand, I don't discard errors on my part, since this is a summer project during my ML learning and it is 
the first time I used Python for such a task (also the first time I do such a task at all). If there is anything I missed or any mistakes I have made, let me know!
