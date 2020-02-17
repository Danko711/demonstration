import numpy as np
from scipy.stats import spearmanr
from utils.helper_functions import target_cols


def spearman_one_column(y_true, y_pred):
    return np.nan_to_num(
        spearmanr(y_true,y_pred).correlation
    )


def spearman_correlation(y_true, y_pred):
    rho_val = np.nanmean([
        spearmanr(y_true[:, ind], y_pred[:, ind]).correlation
        for ind in range(y_pred.shape[1])
    ])
    return rho_val


def spearman_correlation_columnwise(y_true, y_pred):
    spearman_dict = {}
    for ind in range(y_pred.shape[1]):
        spearman_column = np.nan_to_num(spearmanr(
            y_true[:, ind],
            y_pred[:, ind]
        ).correlation)
        spearman_dict[target_cols[ind]] = spearman_column
    return spearman_dict
