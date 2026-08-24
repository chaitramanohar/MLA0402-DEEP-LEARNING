from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3, 3),
    activation='tanh',
    learning_rate_init=0.1,
    max_iter=1000
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
