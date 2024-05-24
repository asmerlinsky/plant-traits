import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from src.constants import ID, SD, SPECIES, TARGETS


def drop_by_magnitude(df, magnitude_dict):

    for target, (min_magnitude, max_magnitude) in magnitude_dict.items():
        try:
            mask += (df[target] > min_magnitude) & (df[target] < max_magnitude)
        except NameError:
            mask = (df[target] > min_magnitude) & (df[target] < max_magnitude)


def drop_outliers(df, z_threhold, apply_cols=None):
    if apply_cols is None:
        drop_cols = [sd for sd in SD if sd in df.columns]

        if ID in df.columns:
            drop_cols.append(ID)

        zcored_data = df.apply(zscore, axis=0).drop(axis=1, labels=drop_cols)
    else:
        zcored_data = df[apply_cols].apply(zscore, axis=0)

    mask = np.abs(zcored_data) > z_threhold
    print(
        f"Dropping {mask.any(axis=1).sum()} rows ({mask.any(axis=1).sum()/len(zcored_data.columns):2.0f} outliers per column)"
    )
    return df[~mask.any(axis=1)], df[mask.any(axis=1)]


def scale_variables(df, test_df=None, log_scale_targets=[]):

    train_variables = df.drop(axis=1, labels=TARGETS + SD + [ID])

    if SPECIES in df.columns:
        train_variables = train_variables.drop(axis=1, labels=[SPECIES])

    targets_df = df[TARGETS]

    other_df = df[SD + [ID]]

    train_variables.mean()

    var_mean = train_variables.mean()
    var_std = train_variables.std()

    transformed_train_df = (train_variables - var_mean) / var_std

    var_mean.name = "mean"
    var_std.name = "std"

    scaler = pd.concat((var_mean, var_std), axis=1)

    transformed_train_df = transformed_train_df.merge(
        other_df, left_index=True, right_index=True
    )

    if SPECIES in df.columns:
        transformed_train_df[SPECIES] = df[SPECIES]

    if len(log_scale_targets) > 0:
        targets_df[log_scale_targets] = targets_df[log_scale_targets].apply(np.log)

    target_mean = targets_df.mean()
    target_std = targets_df.std()
    target_mean.name = "mean"
    target_std.name = "std"

    target_scaler = pd.concat((target_mean, target_std), axis=1)
    target_scaler.index.name = "targets"

    transformed_targets_df = (targets_df - target_mean) / target_std

    if test_df is not None:
        test_variables = test_df.drop(axis=1, labels=[ID])
        test_ids = test_df[[ID]]
        transformed_test_df = (test_variables - var_mean) / var_std

        transformed_test_df = transformed_test_df.merge(
            test_ids, left_index=True, right_index=True
        )
    else:
        transformed_test_df = None

    return (
        transformed_train_df.merge(
            transformed_targets_df, left_index=True, right_index=True
        ),
        transformed_test_df,
        (scaler, target_scaler),
    )


def get_scaler(path):
    return pd.read_csv("data/std_scaler.csv", index_col="targets")


def inverse_transform(
    pred_df: pd.DataFrame, standard_scaler: StandardScaler, log_targets: list
):
    inverted = standard_scaler.inverse_transform(pred_df)
    inverted = pd.DataFrame(inverted, index=pred_df.index, columns=pred_df.columns)
    inverted[log_targets] = np.exp(inverted[log_targets])
    return inverted
