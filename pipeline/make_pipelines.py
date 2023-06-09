from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


def main():
    pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
    print(f"Pipeline steps:\n{pipe_short.steps}")

    pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
    print(f"Pipeline steps:\n{pipe.steps}")


if __name__ == "__main__":
    main()
