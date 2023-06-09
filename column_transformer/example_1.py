from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def main():
    X, y = load_iris(as_frame=True, return_X_y=True)
    sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
    petal_cols = ["petal length (cm)", "petal width (cm)"]

    preprocessor = ColumnTransformer(
        [
            ("scaler", StandardScaler(), sepal_cols),
            ("kbin", KBinsDiscretizer(encode="ordinal"), petal_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_out = preprocessor.fit_transform(X)
    print(X_out.sample(n=5, random_state=0))


if __name__ == "__main__":
    main()
