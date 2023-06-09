import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    X, y = fetch_openml(
        "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
    )
    numeric_features = ["age", "fare"]
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    categorical_features = ["embarked", "pclass"]
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )
    log_reg = make_pipeline(preprocessor, SelectKBest(k=7), LogisticRegression())
    log_reg.fit(X, y)

    log_reg_input_features = log_reg[:-1].get_feature_names_out()

    pd.Series(log_reg[-1].coef_.ravel(), index=log_reg_input_features).plot.bar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
