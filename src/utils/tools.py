import numpy as np
from sklearn.linear_model import LinearRegression


def get_rho_lstsq(x, y, end_point=False, ydim=0):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        end_point (bool, optional): _description_. Defaults to False.
        ydim (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    path_length = x.shape[1]
    data_num = x.shape[0]
    dim = x.shape[2]
    rho = []

    if not end_point:
        X = []
        Y = []
        for i in range(path_length):
            Y.append(y[:, i, ydim][:, None])
        Y = np.vstack(Y)

        for i in range(1, path_length + 1):
            X.append(
                np.hstack(
                    [
                        np.zeros((data_num, path_length * dim - i * dim)),
                        x[:, :i, :].reshape(data_num, i * dim),
                    ]
                )
            )
        X = np.vstack(X)
    else:
        X = x.reshape(data_num, path_length * dim)
        Y = y[:, :, 0]

    # r = np.linalg.lstsq(X, Y)[0]
    reg = LinearRegression(n_jobs=6).fit(X, Y)
    rho = reg.coef_

    rho = rho.reshape(path_length, dim)
    return np.flip(rho)
