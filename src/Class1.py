from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Iris:
    def __init__(self, model=None, test_size=0.3, random_state=10):

        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.model = model if model else DecisionTreeClassifier()
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4
        self.y_pred = None
        self.accuracy = None

    def data(self):
        print("Features shape:", self.X.shape)  
        print("Labels shape:", self.y.shape)    
        print("Target names:", self.iris.target_names)
        print("Feature names:", self.iris.feature_names)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print("Model accuracy:", self.accuracy)
        return self.accuracy