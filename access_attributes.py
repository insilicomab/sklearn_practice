from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # load and split the data
    cancer = load_breast_cancer()

    pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())

    # fit the pipeline defined before to the cancer dataset
    pipe.fit(cancer.data)

    # extract the first two principal components from the "pca" step
    components = pipe.named_steps["pca"].components_
    print(f"components.shape: {components.shape}")


if __name__ == "__main__":
    main()
