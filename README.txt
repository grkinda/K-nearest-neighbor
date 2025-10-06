Iris KNN Training and Visualization

Overview
- Trains a simple K-Nearest Neighbors (KNN) classifier on the Iris dataset using a custom implementation in `iris training.py`.
- Evaluates accuracy, predicts a sample flower, and visualizes data and performance.

Requirements
- Python 3.8+
- numpy, scikit-learn, matplotlib, seaborn, pandas

Install
1) Open PowerShell in this folder
2) Install dependencies:
   pip install numpy scikit-learn matplotlib seaborn pandas

Run
- Execute:
  python "iris training.py"

What it does
- Loads Iris dataset, splits into train/test (80/20, reproducible split).
- Trains custom KNN (Euclidean distance) with k=3.
- Prints predicted species for a sample input and overall accuracy.
- Displays a seaborn pairplot and a confusion matrix.

Customize
- Change neighbors: edit `knn = KNN(k=3)`.
- Change sample prediction: edit `new_flower` array.

Notes
- KNN stores the training data and computes distances at prediction time.
- Visualization windows must be closed to let the script finish.
