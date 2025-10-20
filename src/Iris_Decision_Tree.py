#This code was written by: Senem Keceli 
#This code was written for: Learning the basics of ML using the iris dataset using a decison tree classifier

from sklearn.datasets import load_iris
#importing all of these to show the difference between different types of classification algorithms
#during this program, i will use the decision tree classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

iris = load_iris() #loading the iris dataset into python
#you can also use a .csv file and upload the dataset using pandas

#because i used load_iris the data is already organized for me
#if it is an external dataset, manually define X (features) and y (targets/labels)
#Example for iris set: 
#X = iris.drop(['Id','Species'])
#y = iris['Species]

#defining my features and labels 
X = iris.data  #features (sepal length, sepal width, petal length, petal width)
y = iris.target  #labels / targets variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

#doing this allows us to make a pairplot by converting to pandas dataframe
#also replaces numerical place holders with the species name
df = pd.DataFrame(X, columns=iris.feature_names)
df["Species"] = pd.Series(y).map(dict(enumerate(iris.target_names)))

#data preview
print(X.shape) #(150,4): 150 rows (flower samples) and 4 columns (measurements: sepal length, sepal width, petal length, and petal width). 
print(y.shape) #(150,)

#data visualization
#pairplot
sns.pairplot(df, hue='Species', markers='+')
plt.show()

#sepal length vs width 
fig = df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()

df.hist(edgecolor='black')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

g = sns.violinplot(y='Species', x='SepalLengthCm', data=df, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='SepalWidthCm', data=df, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalLengthCm', data=df, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalWidthCm', data=df, inner='quartile')
plt.show()


#split into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
#test size is the proportion (0-1) that the data is in the test split. here, 30% of data is going into the test split
#random state allows for reproductible results, but taking away random state could tell you how stable your model is. 
#random state = 10 ensures that it is the same split each time it runs

#training the decison tree
dt = DecisionTreeClassifier() #creating a model
dt.fit(X_train, y_train) #training on the training set X and y
y_pred = dt.predict(X_test) #predict the test set
#ensure model accuracy 
acc_dt = metrics.accuracy_score(y_pred,y_test)
print("accuracy:", acc_dt)
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel('True')
plt.title("Iris Decision Tree Classifier")
plt.show()