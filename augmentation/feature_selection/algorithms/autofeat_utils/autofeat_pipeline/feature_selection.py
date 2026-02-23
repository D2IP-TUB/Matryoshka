import numpy as np
from scipy.stats import rankdata


def pearson_correlation(x, y):
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    sum_dev = y_dev.T.dot(x_dev).reshape((x.shape[1],))
    denominators = np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i
         in range(len(denominators))])
    return results


def spearman_correlation(x, y):
    n = x.shape[0]
    if n < 2:
        raise ValueError("The input should contain more than 1 sample")

    x_ranks = np.apply_along_axis(rankdata, 0, x)
    y_ranks = rankdata(y)

    return pearson_correlation(x_ranks, y_ranks)
