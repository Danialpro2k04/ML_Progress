from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC()
    }

    results = {}

    for name, model in models.items():
        # Listing algos which require scaling
        # StandardScaler is used for Logistic Regression and SVM in this case
        if name in ['Logistic Regression', 'Support Vector Machine']:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used, X_test_used = X_train, X_test

        # Train & evaluate
        model.fit(X_train_used, y_train)
        y_pred = model.predict(X_test_used)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    # Results dataframe
    results_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    })

    plt.figure(figsize=(8, 5))
    sns.pointplot(data=results_df, x="Model", y="Accuracy",
                  color="dodgerblue", markers="o", linestyles="--")
    plt.title("Model Accuracy Comparison", fontsize=10)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_accuracy = results[best_model_name]
    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

    return best_model_name, best_model, best_accuracy, results
