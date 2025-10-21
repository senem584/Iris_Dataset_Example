# This code was written by: Senem Keceli
# This code was written for: Learning the basics of ML using the iris dataset using a decision tree classifier

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
# you can also use a .csv file and upload the dataset using pandas 
# because i used load_iris the data is already organized for me 
# if it is an external dataset, manually define X (features) and y (targets/labels) 
# Example for iris set: 
# X = iris.drop(['Id','Species']) 
# y = iris['Species]

#defining my features and labels
X = iris.data #features (sepal length, sepal width, petal length, petal width)
y = iris.target #labels / targets variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

#doing this allows us to make a pairplot by converting to pandas dataframe
#also replaces numerical place holders with the species name
df = pd.DataFrame(X, columns=iris.feature_names)
df["Species"] = pd.Series(y).map(dict(enumerate(iris.target_names)))

print(X.shape)  # (150, 4): 150 rows (flower samples) and 4 columns (measurements: sepal length, sepal width, petal length, and petal width).
print(y.shape)  # (150,)

#visualization

# pairplot
sns.pairplot(df, hue='Species', markers='+')
plt.show()

# sepal length vs width
ax = df[df.Species=='setosa'].plot(kind='scatter',
                                   x='sepal length (cm)', y='sepal width (cm)',
                                   color='orange', label='setosa')
df[df.Species=='versicolor'].plot(kind='scatter',
                                  x='sepal length (cm)', y='sepal width (cm)',
                                  color='blue', label='versicolor', ax=ax)
df[df.Species=='virginica'].plot(kind='scatter',
                                 x='sepal length (cm)', y='sepal width (cm)',
                                 color='green', label='virginica', ax=ax)
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_title("Sepal Length vs Width")
ax.figure.set_size_inches(12, 8)       
plt.show()

# histograms
df.hist(edgecolor='black')
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
plt.show()

# violin plots
sns.violinplot(y='Species', x='sepal length (cm)', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='sepal width (cm)', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='petal length (cm)', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='petal width (cm)', data=df, inner='quartile')
plt.show()

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# train decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# metrics (y_true, y_pred)
acc_dt = metrics.accuracy_score(y_test, y_pred)
print("accuracy:", acc_dt)

cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Iris Decision Tree Classifier")
plt.show()
