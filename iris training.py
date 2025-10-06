import numpy as np  # For numerical operations and arrays

# Function to calculate Euclidean distance between two points
# This is how we measure "closeness" in our KNN model
def euclidean_distance(x1, x2):
    # √((x1_1 - x2_1)^2 + (x1_2 - x2_2)^2 + ... )
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Our kawaii KNN class!
class KNN:
    def __init__(self, k=3):
        # k is the number of neighbors to consider
        # A smaller k (like 1) is more sensitive to noise
        # A larger k (like 5) is more stable but might miss local patterns
        self.k = k

    def fit(self, X_train, y_train):
        # Just store the training data
        # KNN is a "lazy learner" - it doesn't learn until you ask it to predict!
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # For each sample in X_test, call _predict
        # This is the main function you call to get predictions for a whole batch
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        # This is the heart of KNN - finding the closest neighbors!
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        # argsort returns the indices that would sort the array
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        # These are the "votes" from the neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        # bincount counts the number of occurrences of each label
        # argmax returns the label with the highest count
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# (⌒‿⌒)っ✎ Loading the Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
# This dataset has 150 samples, 4 features, and 3 classes
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (0: setosa, 1: versicolor, 2: virginica)

# Split the dataset into training and testing sets
# 80% for training, 20% for testing
# random_state=42 ensures you get the same split every time (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
# k=3 means we look at the 3 closest neighbors
knn = KNN(k=3)

# Train the classifier on the training set
# This just stores the data - KNN doesn't "learn" until you predict
knn.fit(X_train, y_train)

# Make predictions on the test set
# This is where the magic happens - the model finds the k nearest neighbors and votes!
predictions = knn.predict(X_test)

# Calculate the accuracy of the classifier
# Accuracy = (number of correct predictions) / (total number of predictions)
accuracy = accuracy_score(y_test, predictions)

# (｡•̀ᴗ-)✧ Let's try predicting a new flower!
# This is how you'd use the model in real life
new_flower = np.array([[5.0, 3.6, 1.4, 0.2]])  # Example measurements
pred = knn.predict(new_flower)
print("Predicted species:", iris.target_names[pred][0])

# Print the accuracy
print(f"Accuracy: {accuracy}")

# (๑˃ᴗ˂)ﻭ✧ Let's visualize the data!
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame for seaborn
# This makes it easy to plot and explore the data
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# Plot a pairplot - this shows all features against each other
# hue='species' colors the points by species
# diag_kind='hist' shows histograms on the diagonal
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# (｡♥‿♥｡) Let's see how well our model did with a confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# A confusion matrix shows:
# - True Positives (correctly predicted)
# - False Positives (predicted as positive but actually negative)
# - False Negatives (predicted as negative but actually positive)
# - True Negatives (correctly predicted as negative)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



