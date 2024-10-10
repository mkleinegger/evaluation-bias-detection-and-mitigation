import numpy as np
import pandas as pd
import fairlens as fl
from matplotlib import pyplot as plt
from aif360.detectors.mdss_detector import bias_scan
from aif360.datasets import StandardDataset
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import (
    equal_opportunity_difference,
    average_odds_difference,
    statistical_parity_difference,
    disparate_impact_ratio,
    theil_index,
)
from aif360.algorithms.preprocessing import Reweighing
from utils import create_model, describe_model, split_data, train_and_evaluate
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from aif360.sklearn.metrics import (
    make_scorer,
    statistical_parity_difference,
)


def search_bias(
    data,
    observations,
    expectations,
    favorable_value,
    scoring="Bernoulli",
    penalty=1,
    alpha=0.24,
):
    privileged_subset = bias_scan(
        data=data,
        observations=observations,
        scoring=scoring,
        expectations=expectations,
        overpredicted=True,
        penalty=penalty,
        alpha=alpha,
        favorable_value=favorable_value,
    )

    unprivileged_subset = bias_scan(
        data=data,
        observations=observations,
        scoring=scoring,
        expectations=expectations,
        overpredicted=False,
        penalty=penalty,
        alpha=alpha,
        favorable_value=favorable_value,
    )
    return privileged_subset, unprivileged_subset


def calc_fairness_score(data, protected_attributes, target, verbose=False):
    _df = data.copy()
    _df.dropna(inplace=True)
    _df.reset_index(inplace=True)

    fscorer = fl.FairnessScorer(
        _df, target_attr=target, sensitive_attrs=protected_attributes
    )

    if verbose:
        fscorer.demographic_report()

    return fscorer


def explain_detected_bias(data, expectations, target, subset, subset_type="privileged"):
    _df = data.copy()
    _df["expectations"] = expectations.copy()

    _to_choose = _df[subset.keys()].isin(subset).all(axis=1)
    _to_choose = _df.loc[_to_choose]

    print(
        "Our detected {} group has a size of {}, we observe {} as the average probability of earning >50k, but our model predicts {}".format(
            subset_type,
            len(_to_choose),
            np.round(_to_choose[target].mean(), 4),
            np.round(_to_choose["expectations"].mean(), 4),
        )
    )


def transform_to_bias_dataset(
    data, protected_attributes, priveleged_groups, increase_bias=False, verbose=False
):
    column_renaming = {}
    _df = (
        data.copy()
    )  # Make a copy of the DataFrame to avoid modifying the original DataFrame
    _df = _df.dropna()
    if verbose and len(data) != len(_df):
        print(f"{len(data)-len(_df)} Na rows removed!")

    for i, col in enumerate(protected_attributes):
        _df[f"{col}_index"] = _df[col].copy()
        if increase_bias:
            _df[f"{col}_index"] = _df[f"{col}_index"].map(
                lambda x: 1 if x in priveleged_groups[i] else 0
            )
        _df[f"{col}"] = _df[col].map(lambda x: 1 if x in priveleged_groups[i] else 0)
        column_renaming[f"{col}_index"] = f"{col}"

    if len(protected_attributes) > 0:
        _df.set_index(protected_attributes, inplace=True)
        _df = _df.rename(columns=column_renaming)

    return _df


def describe_fairness(observations, expectations, protected_attributes, verbose=False):
    privileged_group = (1,) * len(protected_attributes)
    fairness_metrics = {}

    fairness_metrics["statistical_parity_difference"] = statistical_parity_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["average_odds_difference"] = average_odds_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["equal_opportunity_difference"] = equal_opportunity_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["disparate_impact"] = disparate_impact_ratio(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["theil_index"] = theil_index(1 + expectations - observations)

    if verbose:
        print(f"Metric{' ':25} Value{' ':15}")
        for k, v in fairness_metrics.items():
            print(f"{k:32}{v:.3f}")

    return fairness_metrics


def scan_and_calculate_fairness(model, data, target, penalty):
    _df = data.copy()

    X_train, y_train = split_data(_df, target, drop_na=True)
    _, y_pred = train_and_evaluate(model, _df, _df, target, drop_na=True)
    y_probs = pd.Series(model.predict_proba(X_train)[:, 1])

    privileged_subset = bias_scan(
        data=X_train,
        observations=y_train,
        scoring="Bernoulli",
        expectations=y_probs,
        overpredicted=True,
        penalty=penalty,
        alpha=0.24,
        favorable_value=1,
    )

    df_bias = transform_to_bias_dataset(
        _df,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )

    if len(privileged_subset[0].keys()) == 0:
        return {
            "statistical_parity_difference": 0,
            "average_abs_odds_difference": 0,
            "equal_opportunity_difference": 0,
            "disparate_impact": 1,
            "theil_index": 0,
        }, privileged_subset
    else:
        metrics = describe_fairness(
            df_bias[target], y_pred, list(privileged_subset[0].keys())
        )
        return metrics, privileged_subset


def compute_metrics(
    dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp=True
):
    """Compute the key metrics"""
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    metrics = OrderedDict()
    metrics["Statistical parity difference"] = (
        classified_metric_pred.statistical_parity_difference()
    )
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = (
        classified_metric_pred.average_abs_odds_difference()
    )
    metrics["Equal opportunity difference"] = (
        classified_metric_pred.equal_opportunity_difference()
    )
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def create_aif360_standardDataset(
    data,
    categorical_features,
    target,
    favorable_classes,
    protected_attributes,
    privileged_groups,
):
    _df = data.copy()

    if isinstance(favorable_classes, int):
        favorable_classes = [favorable_classes]

    privileged_classes = []
    for i, group in enumerate(privileged_groups):
        if protected_attributes[i] not in categorical_features:
            privileged_classes.append(lambda x, group=group: x in group)
        else:
            privileged_classes.append(group)

    dataset = StandardDataset(
        df=_df,
        label_name=target,
        favorable_classes=favorable_classes,
        scores_name="",
        protected_attribute_names=protected_attributes,
        privileged_classes=privileged_classes,
        categorical_features=[
            col for col in categorical_features if col not in protected_attributes
        ],
    )

    dataset.scores = dataset.labels.copy()
    return dataset


def plot_fairness_metrics(data):
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plotting df_bias
    data[filter(lambda x: x != "disparate_impact", data.columns)].plot(
        kind="line", ax=axes[1]
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Bias Metrics vs. Iteration")
    axes[1].grid(True)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plotting df_bias
    data["disparate_impact"].plot(kind="line", ax=axes[0])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Bias Metrics vs. Iteration")
    axes[0].grid(True)
    axes[0].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def reweight_mitigation(
    clf,
    nominal_features,
    target,
    X_train,
    y_train,
    X_test,
    y_test,
    penalty=5,
    sample_weights=None,
):
    _df_train = pd.concat([X_train.copy(), y_train.copy()], axis=1)
    model = create_model(clf, nominal_features)
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    probs = pd.Series(model.predict_proba(X_train)[:, 1])
    y_pred = model.predict(X_train)

    privileged_subset, _ = search_bias(X_train, y_train, probs, 1, penalty=penalty)
    # if bias-free, just return
    if len(privileged_subset[0].keys()) <= 0:
        return None, None, None

    model_metrics = describe_model(y_test, model.predict(X_test), verbose=False)

    # do bias mitigation
    train_standard_dataset = create_aif360_standardDataset(
        _df_train,
        nominal_features,
        target,
        1,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )

    if sample_weights is not None:
        train_standard_dataset.instance_weights = sample_weights

    # create (un)privileged groups
    privileged_groups = [{key: 1 for key in list(privileged_subset[0].keys())}]
    unprivileged_groups = [{key: 0 for key in list(privileged_subset[0].keys())}]

    df_train_bias = transform_to_bias_dataset(
        _df_train,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )
    fair_metrics = describe_fairness(
        df_train_bias[target], y_pred, list(privileged_subset[0].keys()), verbose=False
    )

    RW = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    RW.fit(train_standard_dataset)

    dataset = RW.transform(train_standard_dataset)
    return dataset.instance_weights, model_metrics, fair_metrics


def get_fair_learning_scoring(protected_attributes):
    def discrimination(y_true, y_pred, protected_attributes):
        return abs(
            statistical_parity_difference(
                y_true,
                y_pred,
                prot_attr=protected_attributes,
                priv_group=(1,) * len(protected_attributes),
            )
        )

    def delta(y_true, y_pred, protected_attributes=None, use_bal_acc=False):
        if use_bal_acc:
            return balanced_accuracy_score(y_true, y_pred) - discrimination(
                y_true, y_pred, protected_attributes
            )
        else:
            return accuracy_score(y_true, y_pred) - discrimination(
                y_true, y_pred, protected_attributes
            )

    return make_scorer(
        delta, protected_attributes=protected_attributes, use_bal_acc=False
    )
