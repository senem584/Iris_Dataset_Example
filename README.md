# Iris Dataset Example
This project demonstrates classification using the Iris dataset with machine learning algorithms. The main focus is on implementing a Decision Tree Classifier. The Iris dataset is a classic dataset in machine learning that contains 150 samples of iris flowers, with four features (sepal length, sepal width, petal length, petal width) and three classes (Setosa, Versicolor, and Virginica). It is a classic example of a supervised learning problem used to explore classification, data visualization, and model evaluation techniques.

The goal of this project is to showcase fundamental techniques in data preprocessing, visualization, and machine learning classification, highlighting how models can learn from real-world numeric data to distinguish between different categories with high accuracy. The code is implemented in two styles: a regular procedural version and an object-oriented (OOP) version for better modularity and reusability. Both are shown in the [source](#src) folder. 

# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [Error Handling](#error-handling)
- [References](#references)

# Implementation
The implementation begins by loading the Iris dataset from the scikit-learn library. The dataset contains:
- 150 samples
- 4 numerical features (sepal length, sepal width, petal length, petal width)
- 3 species labels (setosa, versicolor, virginica)

Data visualization is performed using Seaborn and Matplotlib, including:
- Pairplots showing relationships between features
- Scatter plots comparing sepal and petal dimensions
- Violin and histogram plots to display data distribution across species

The dataset is then split into training and testing sets (70/30 split) using train_test_split(). A Decision Tree Classifier is trained on the training set, and predictions are made on the test set.

To evaluate model performance:
- Accuracy score is calculated using metrics.accuracy_score()
- Confusion Matrix and Heatmap visualize the model’s prediction performance

This dataset is widely used for educational purposes in machine learning and data visualization because it provides a simple yet powerful introduction to classification concepts.
# Requirments 
This project requires tensorflow, keras, and scikit-learn. It was developed using a Python environment through VSCode.

Use 'pip install -r requirements.txt' to install the following dependencies:

```
contourpy==1.3.3
cycler==0.12.1
fonttools==4.60.1
joblib==1.5.2
kiwisolver==1.4.9
matplotlib==3.10.6
numpy==2.3.2
packaging==25.0
pandas==2.3.3
pillow==11.3.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
```
# How to Use
Clone the repository
- On GitHub, click the Code button and copy the HTTPS URL.
- In VS Code, choose Clone Repository, then paste the URL.

Run the file
- Locate the file named iris_decision_tree.py.

Run using:
- Make sure your enviornment is active. 
- python iris_decision_tree.py
# References 
- [1] Scikit-learn Documentation: Iris Dataset. https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
- [2] GeeksforGeeks, “Decision Tree Classifier in Python using Scikit-learn.” https://www.geeksforgeeks.org/machine-learning/decision-tree-implementation-python/
- [3] suneelpatel, “Learn ML from Scratch with IRIS Dataset,” Kaggle.com, Sep. 04, 2019. https://www.kaggle.com/code/suneelpatel/learn-ml-from-scratch-with-iris-dataset
‌
‌
