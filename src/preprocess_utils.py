import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

from src.constants import ID, SD, TARGETS


def drop_outliers(df, z_threhold):
    zcored_data = df.apply(zscore).drop(axis=1, labels=TARGETS + SD + [ID])
    mask = np.abs(zcored_data) > z_threhold
    return df[~mask.any(axis=1)]


def scale_variables(df, test_df=None):
    sc = StandardScaler()
    target_scaler = StandardScaler()

    train_variables = df.drop(axis=1, labels=TARGETS + SD + [ID])
    column_variables = train_variables.columns

    targets_df = df[TARGETS]

    other_df = df[SD + [ID]]

    transformed_variables = sc.fit_transform(train_variables.values)
    transformed_train_df = pd.DataFrame(transformed_variables, columns=column_variables)

    transformed_targets = target_scaler.fit_transform(targets_df.values)
    transformed_targets_df = pd.DataFrame(
        transformed_targets, columns=targets_df.columns
    )

    if test_df is not None:
        test_variables = test_df.drop(axis=1, labels=[ID])
        test_ids = test_df[[ID]]
        transformed_test = sc.transform(test_variables.values)
        transformed_test_df = pd.DataFrame(
            transformed_test, columns=test_variables.columns
        )
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
        (sc, target_scaler),
    )
