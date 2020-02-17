import numpy as np
import pandas as pd
from scipy.special import erfinv


def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x += efi_x.mean()
    return efi_x


def replace_one_col_with_nearest(pred_col, uniq_targets, thres):
    new_pred_col = np.zeros_like(pred_col)
    for i, pred in enumerate(pred_col):
        nearest = uniq_targets[np.argmin(np.abs(uniq_targets - pred))]
        min_dist = abs(pred - nearest)
        new_pred_col[i] = (nearest if min_dist < thres else pred)

    return new_pred_col


def replace_with_nearest(preds, targets, thres_list):

    new_preds = preds.copy()
    for ind in range(preds.shape[1]):
        pred_col = preds[:, ind]
        target_col = targets[:, ind]
        thres = thres_list[ind]
        if np.isnan(thres):
            continue
        new_pred_col = replace_one_col_with_nearest(pred_col, np.unique(target_col), thres)
        new_preds[:, ind] = new_pred_col

    return new_preds


def replace_with_nearest_old(preds, targets, thres):
    cols_to_replace = [2, 5, 11, 12, 14, 15, ]
    cols_to_replace = np.arange(preds.shape[1])
    cols_to_replace = [2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 24, 28]
    cols_to_replace = [2, 11, 12, 13, 14, 15, 16]

    new_preds = preds.copy()
    for ind in range(preds.shape[1]):
        pred_col = preds[:, ind]
        target_col = targets[:, ind]
        if ind in cols_to_replace:
            new_pred_col = replace_one_col_with_nearest(pred_col, np.unique(target_col), thres)
        elif ind == 19:  # quest_type_spelling
            new_pred_col = pred_col
            new_pred_col[np.argsort(new_pred_col)[:475]] = 0
        else:
            new_pred_col = pred_col
        new_preds[:, ind] = new_pred_col

    return new_preds


def blend_preds(preds_list):
    unique_values = set()
    for preds in preds_list:
        unique_values = unique_values.union(set(preds))
    mapping = dict([[val, rank] for rank, val in enumerate(sorted(unique_values))])
    encoded_preds = [pd.Series(preds).map(mapping) for preds in preds_list]
    blended_preds = [np.mean(i) for i in zip(*encoded_preds)]
    return blended_preds
