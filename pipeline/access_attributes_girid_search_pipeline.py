from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # load and split the data
    cancer = load_breast_cancer()

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=4
    )
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"Best estimator:\n{grid.best_estimator_}")
    print(
        f"Logistic regression step:\n{grid.best_estimator_.named_steps['logisticregression']}"
    )
    print(
        f"Logistic regression coefficients:\n{grid.best_estimator_.named_steps['logisticregression'].coef_}"
    )


if __name__ == "__main__":
    main()
