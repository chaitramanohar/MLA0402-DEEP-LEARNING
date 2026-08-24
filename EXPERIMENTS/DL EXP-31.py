from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3, 3),
    activation='relu',
    learning_rate_init=0.03,
    max_iter=1000
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
