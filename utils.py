import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def create_model(clf, categorical_features):
    categorical_features_onehot_transformer = ColumnTransformer(
        transformers=[
            (
                "one-hot-encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        [
            ("one-hot-encoder", categorical_features_onehot_transformer),
            ("clf", clf),
        ]
    )
    return model


def split_data(data, target, drop_na=False):
    _df = data.copy()
    if drop_na:
        _df = _df.dropna()

    X = _df.drop(columns=[target])
    y = _df[target]

    return X, y


def train_and_evaluate(
    model, train_data, test_data, target, drop_na=False, sample_weight=None
):
    X_train, y_train = split_data(train_data, target, drop_na)
    X_test, y_test = split_data(test_data, target, drop_na)

    if sample_weight is not None:
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


def describe_model(y_test, y_pred, verbose=False):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
    }

    if verbose:
        print(f"Metric{' ':8} Value{' ':15}")
        for k, v in metrics.items():
            print(f"{k:15}{v:.3f}")
            
    return metrics


def plot_metrics(data, xlabel, ylabel, title):
    data.plot(kind="line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

def prepare_data_fair_learning(df_train, df_test, nominal_features, target):
    def custom_one_hot_encoding(X_train, X_test, nominal_features):
        X_train = pd.get_dummies(X_train, columns=nominal_features)
        X_test = pd.get_dummies(X_test, columns=nominal_features)

        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

        X_train.index = X_train.index
        X_test.index = X_test.index
        X_train[X_train.columns] = X_train[X_train.columns].astype(int)
        X_test[X_test.columns] = X_test[X_test.columns].astype(int)
        return X_train, X_test

    _df_train = df_train.copy()
    _df_test = df_test.copy()
    X_train, y_train = split_data(_df_train, target, drop_na=True)
    X_test, y_test = split_data(_df_test, target, drop_na=True)

    if isinstance(_df_train.index, pd.MultiIndex):
        X_train.index = y_train.index = pd.MultiIndex.from_arrays(X_train.index.codes, names=X_train.index.names)
        X_test.index = y_test.index = pd.MultiIndex.from_arrays(X_test.index.codes, names=X_test.index.names)
    y_train = pd.Series(y_train.factorize(sort=True)[0], index=y_train.index, name=y_train.name)
    y_test = pd.Series(y_test.factorize(sort=True)[0], index=y_test.index, name=y_test.name)
    
    X_train, X_test = custom_one_hot_encoding(X_train, X_test, nominal_features)

    return X_train, y_train, X_test, y_test


def plot_fairlearning_results(results):
    _, axes = plt.subplots(1, 3, squeeze=True, figsize=(24, 6))
    by_fair = results.set_index("param_fairness_weight")

    for ax, r in zip(axes, by_fair.index.unique()):
        pivot_table = by_fair.xs(r).pivot(
            index="param_reconstruct_weight",
            columns="param_target_weight",
            values="mean_test_score",
        )
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".1%",
            vmin=results["mean_test_score"].min(),
            vmax=results["mean_test_score"].max(),
            square=True,
            cbar=False,
            ax=ax,
        )
        ax.set_title("param_fairness_weight={}".format(r))

    plt.show()