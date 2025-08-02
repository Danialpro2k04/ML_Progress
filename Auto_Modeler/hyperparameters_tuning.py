import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

MODEL_CLASSES = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Support Vector Machine": SVC
}

HYPERPARAM_GRIDS = {
    "Logistic Regression": [
        {"C": 0.01}, {"C": 0.1}, {"C": 1}, {"C": 10}
    ],
    "Decision Tree": [
        {"criterion": "gini", "max_depth": None},
        {"criterion": "gini", "max_depth": 5},
        {"criterion": "entropy", "max_depth": None}
    ],
    "Random Forest": [
        {"n_estimators": 50},
        {"n_estimators": 100},
        {"n_estimators": 200}
    ],
    "Support Vector Machine": [
        {"C": 0.1}, {"C": 1}, {"C": 10}
    ]
}


def tune(best_model_name, X_train, X_test, y_train, y_test):
    model_class = MODEL_CLASSES.get(best_model_name)
    if model_class is None:
        raise ValueError(f"Unsupported model: {best_model_name}")

    params_list = HYPERPARAM_GRIDS.get(best_model_name, [])
    if not params_list:
        raise ValueError(f"No hyperparameters defined for {best_model_name}")

    scores = []
    labels = []

    for params in params_list:
        model = model_class(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores.append(acc)
        labels.append(str(params))
        print(f"Params: {params} -> Accuracy: {acc:.4f}")

    best_index = scores.index(max(scores))
    best_params = params_list[best_index]
    best_score = scores[best_index]

    plt.figure(figsize=(8, 4))
    plt.plot(labels, scores, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title(f"Hyperparameter Tuning for {best_model_name}")
    plt.tight_layout()
    plt.show()

    return best_params, best_score
