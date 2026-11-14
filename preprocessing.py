from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

# High-level preprocessing overview
# Drop 'atemp' and 'dteday' - columns unnecessary for all models
# Split data, deterministically
# Provide transformer to one hot encode certain categorical columns for knn and ridge

# preprocessing ensures three things are identical for everyone:
# 1) rows they train on
# 2) way they are validated
# 3) the metric

# Harness shouldn't fit anything

# categorizing labels, 'dteday', 'atemp' dropped and 'instant' not present
CONTINUOUS = ["temp", "hum", "windspeed"]
# non-binary non-numerical features which need to be encoded for KNN and Ridge
CATEGORICAL_NONBINARY = ["season", "mnth", "hr", "weekday", "weathersit"]
CATEGORICAL_BINARY = ["yr", "holiday", "workingday"]

CV = TimeSeriesSplit()


def preprocess():

    # load dataset
    bike_sharing = fetch_ucirepo(id=275)
    X = bike_sharing.data.features
    y = bike_sharing.data.targets["cnt"]

    # drop columns useless for all 5 models
    X = X.drop(columns=["dteday", "atemp"])

    # split the encoded features. rows are ordered chronologically, thus temporal structure is respected.
    # Because shuffling is off this, deterministically, always splits teh data into the same 80-20 split.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False, test_size=0.2)

    return X_tr, X_te, y_tr, y_te


def load_train_data() -> tuple:
    """
    Returns the tuple (X_tr, y_tr)
    """
    X_tr, _, y_tr, _ = preprocess()
    return X_tr, y_tr


# strictly off limits until final benchmarking
def load_test_data() -> tuple:
    """
    Returns the tuple (X_te, y_te)
    """
    _, X_te, _, y_te = preprocess()
    return X_te, y_te


# to be used for knn and ridge
def encode_categorical():
    # initialize the transformer
    transformer = ColumnTransformer(
        # OneHotEncoder for each column. No need to drop first because only matters for ols.
        transformers=[
            (
                "onehot",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_NONBINARY,
            )
        ],
        # don't throw away non-transformed features
        remainder="passthrough",
        # prevent verbose renaming of columns
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return transformer


# Questions
# Do I need to drop any further columns for sake of redundancy? -> No.
# ColumnTransformer on entire X or just X_tr (i.e. do I transform then split or vice versa)?
# -> transforming should happen inside pipeline of individual models.
# Unknown feature handling? -> Ignore
