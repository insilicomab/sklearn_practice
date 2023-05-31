from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # load and split the data
    cancer = load_breast_cancer()

    pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])

    param_grid = [
        {
            "classifier": [SVC()],
            "preprocessing": [StandardScaler(), None],
            "classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        {
            "classifier": [RandomForestClassifier(n_estimators=100)],
            "preprocessing": [None],
            "classifier__max_features": [1, 2, 3],
        },
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0
    )

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best params:\n{}\n".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))


if __name__ == "__main__":
    main()
