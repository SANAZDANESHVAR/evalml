import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from evalml.utils.gen_utils import _convert_to_woodwork_structure


def load_data(path, index, target, n_rows=None, drop=None, verbose=True, **kwargs):
    """Load features and target from file.

    Arguments:
        path (str): Path to file or a http/ftp/s3 URL
        index (str): Column for index
        target (str): Column for target
        n_rows (int): Number of rows to return
        drop (list): List of columns to drop
        verbose (bool): If True, prints information about features and target

    Returns:
        pd.DataFrame, pd.Series: features and target
    """

    feature_matrix = pd.read_csv(path, index_col=index, nrows=n_rows, **kwargs)

    targets = [target] + (drop or [])
    y = feature_matrix[target]
    X = feature_matrix.drop(columns=targets)

    if verbose:
        # number of features
        print(number_of_features(X.dtypes), end='\n\n')

        # number of total training examples
        info = 'Number of training examples: {}'
        print(info.format(len(X)), end='\n')

        # target distribution
        print(target_distribution(y))

    return X, y


def split_data(X, y, regression=False, test_size=.2, random_state=None):
    """Splits data into train and test sets.

    Arguments:
        X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
        y (ww.DataColumn, pd.Series, or np.ndarray): Target data of length [n_samples]
        regression (bool): If true, do not use stratified split
        test_size (float): Percent of train set to holdout for testing
        random_state (int, np.random.RandomState): Seed for the random number generator

    Returns:
        ww.DataTable, ww.DataTable, ww.DataColumn, ww.DataColumn: Feature and target data each split into train and test sets
    """
    X = _convert_to_woodwork_structure(X)
    y = _convert_to_woodwork_structure(y)

    if regression:
        CV_method = ShuffleSplit(n_splits=1,
                                 test_size=test_size,
                                 random_state=random_state)
    else:
        CV_method = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state)
    train, test = next(CV_method.split(X.to_dataframe(), y.to_series()))

    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]

    return X_train, X_test, y_train, y_test


def number_of_features(dtypes):
    """Get the number of features of each specific dtype in a DataFrame.

    Arguments:
        dtypes (pd.Series): DataFrame.dtypes to get the number of features for

    Returns:
        pd.Series: dtypes and the number of features for each input type
    """
    dtype_to_vtype = {
        'bool': 'Boolean',
        'int32': 'Numeric',
        'int64': 'Numeric',
        'float64': 'Numeric',
        'object': 'Categorical',
        'datetime64[ns]': 'Datetime',
    }

    vtypes = dtypes.astype(str).map(dtype_to_vtype).value_counts()
    return vtypes.sort_index().to_frame('Number of Features')


def target_distribution(targets):
    """Get the target distributions.

    Arguments:
        targets (pd.Series): Target data

    Returns:
        pd.Series: Target data and their frequency distribution as percentages.
    """
    distribution = targets.value_counts() / len(targets)
    return distribution.mul(100).apply('{:.2f}%'.format).rename_axis('Targets')


def drop_nan_target_rows(X, y):
    """Drops rows in X and y when row in the target y has a value of NaN.

    Arguments:
        X (pd.DataFrame): Data to transform
        y (pd.Series): Target data

    Returns:
        pd.DataFrame: Transformed X (and y, if passed in) with rows that had a NaN value removed.
    """
    X_t = X
    y_t = y

    if not isinstance(X_t, pd.DataFrame):
        X_t = pd.DataFrame(X_t)

    if not isinstance(y_t, pd.Series):
        y_t = pd.Series(y_t)

    # drop rows where corresponding y is NaN
    y_null_indices = y_t.index[y_t.isna()]
    X_t = X_t.drop(index=y_null_indices)
    y_t.dropna(inplace=True)
    return X_t, y_t
