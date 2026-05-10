def mean_squared_error(y_pred, y_true):
    return float(np.mean((np.array(y_pred) - np.array(y_true)) ** 2))
