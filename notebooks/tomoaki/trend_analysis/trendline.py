import numpy as np
from sklearn.linear_model import LinearRegression


def smoothing(data, dim, sig_ob=1.0, sig_sys=1.0e-1):
    """Get smoothed data by trend models with dimension=dim

    matrix for state space model used in this model as follows
    x = Fx + Gv
    y  = Hx + w
    we use Q, R as variance matrix for v, w respectively

    Args:
        data (List(float)): timeseries data
        dim (int): the dimension of the trend model
        sig_ob (float): deviation of noise in observation
        sig_sys (float): deviation of nosie in hiddenvariable

    Return:
        np.array(float): smoothed time series data
    """

    # make data for analysis
    data_list = []
    data_def = np.array(data)
    data_list.append(data_def)
    for i in xrange(dim):
        data_def = data_def[1:] - data_def[:-1]
        data_list.append(data_def)

    # reshape data for available
    N_data = len(data_list[-1])
    for i in xrange(len(data_list)):
        data_list[i] = data_list[i][-N_data:]

    # define state space model parameters
    F = np.tri(dim + 1).T
    G = np.tri(dim + 1).T
    R = sig_ob**2 * np.identity(dim + 1)
    Q = sig_sys**2 * np.identity(dim + 1)
    x_data = np.array(data_list).T

    # filtering step
    V_pred_data = []
    V_filt_data = []
    x_pred_data = []
    x_filt_data = []
    x_filt = x_data[0]
    x_filt_data.append(x_filt)
    V_filt = R
    V_filt_data.append(V_filt)
    for x in iter(x_data[1:]):
        x_pred = np.dot(F, x_filt)
        V_pred = np.dot(F, np.dot(V_filt, F.T)) + np.dot(G, np.dot(Q, G.T))
        kal_gain = np.dot(V_pred, np.linalg.inv(V_pred + R))
        x_filt = x_pred + np.dot(kal_gain, x - x_pred)
        V_filt = np.dot(np.identity(dim + 1) - kal_gain, V_pred)
        # store data for smoothing
        x_pred_data.append(x_pred)
        V_pred_data.append(V_pred)
        x_filt_data.append(x_filt)
        V_filt_data.append(V_filt)

    # smoothing step
    N_pred_data = len(V_pred_data)
    x_sm_data = []
    x_sm = x_filt_data[-1]
    x_sm_data.append(x_sm[0])
    V_sm = V_filt_data[-1]
    for i in xrange(N_pred_data):
        idx = N_pred_data - i - 1
        smoother = np.dot(
            V_filt_data[idx],
            np.dot(F.T, np.linalg.inv(V_pred_data[idx]))
        )
        x_sm = x_filt_data[idx]\
            + np.dot(smoother, x_sm - x_pred_data[idx])
        V_sm = V_filt_data[idx]\
            + np.dot(smoother, np.dot(V_sm - V_pred_data[idx], smoother.T))
        x_sm_data.append(x_sm[0])

    x_sm_data.reverse()
    # insert raw data to value at t=0, 1, ..., dim-1
    x_sm_data = np.r_[data[range(dim)], x_sm_data]

    return x_sm_data


def regularized_data(x, y):
    """Get rid of trend of whole data
     and return modified value
    """
    x_tilde = np.atleast_2d(x).T
    y_tilde = np.atleast_2d(y).T
    regr = LinearRegression()
    regr.fit(x_tilde, y_tilde)
    y_tilde = y_tilde - regr.predict(x_tilde)
    return y_tilde.ravel()


def regularized_param(x, y):
    """Return coefficients and intercept of trend
    of whole data to get rid of it
    """
    x_tilde = np.atleast_2d(x).T
    y_tilde = np.atleast_2d(y).T
    regr = LinearRegression()
    regr.fit(x_tilde, y_tilde)
    return regr.coef_[0][0], regr.intercept_[0]


def get_locmax_idx(y):
    """Rreturn local maximal values' index
    don't take edge values into consideration
    """
    # data of first order derivative
    def_data = y[1:] - y[:-1]

    # find local optimum point
    T = len(def_data)
    idx_list = []

    # local maximum
    for i in xrange(1, T):
        if def_data[i] <= 0 and def_data[i - 1] >= 0:
            idx_list.append(i)

    return idx_list


def get_line_param(x1, x2, y1, y2, is_high=True):
    """Get parameters of trendlines of two subsets

    Args:
        x_1, x_2 (list): index of time series data
        y_1, y_2 (list): time series data
        is_high (boolean):
            flag to identify weather data is high_data or lowdata
        window (int): the size of forward and backward points
            when determining local maxmum

    Return:
        coef (float): coefficient of trendline
        incpt (float): intercept of trendline
    """
    if is_high is False:
        y1 = -y1
        y2 = -y2

    x = np.r_[x1, x2]
    y = np.r_[y1, y2]

    # get regularized data and localmax from each subset respectively
    coef, incpt = regularized_param(x, y)
    y1_tilde = y1 - (coef * x1 + incpt)
    y2_tilde = y2 - (coef * x2 + incpt)
    locmax_idx1 = get_locmax_idx(y1_tilde)
    locmax_idx2 = get_locmax_idx(y2_tilde)

    # select two points for trend line
    x1_line = x1[locmax_idx1]
    x2_line = x2[locmax_idx2]
    y1_line = y1[locmax_idx1]
    y2_line = y2[locmax_idx2]
    x = np.r_[x1_line, x2_line]
    y = np.r_[y1_line, y2_line]
    coef, incpt = regularized_param(x, y)
    y1_tilde = y1_line - (coef*x1_line + incpt)
    y2_tilde = y2_line - (coef*x2_line + incpt)
    max_idx1 = np.argsort(y1_tilde)[-1]
    max_idx2 = np.argsort(y2_tilde)[-1]
    idx1 = locmax_idx1[max_idx1]
    idx2 = locmax_idx2[max_idx2]
    x1_line = x1[idx1]
    x2_line = x2[idx2]
    y1_line = y1[idx1]
    y2_line = y2[idx2]

    # compute parameters
    coef = (y1_line - y2_line) / (x1_line - x2_line)
    incpt = y1_line - coef * x1_line

    if is_high is False:
        coef = -coef
        incpt = -incpt

    return coef, incpt


def trendline_backward(data, is_high=True, n_seperate=5, tolerance=0.1):
    """Get trendline data

    Args:
        data(numpy.array): stock data
        is_high(bool): when True, data is high data
                                when False, data is low data
        n_seprates(int): the number of subsets you split data into
        tolearance(float): the tolearance ratio of data going
                                       over trendlines

    Return
        line_data(List(np.array)): trendline data
    """

    if is_high is False:
        data = -data

    n_data = len(data)
    # index for data
    index = np.arange(n_data)

    # make a trendline on each seperated data
    split_data = np.array_split(data, n_seperate, axis=0)[::-1]
    split_idx = np.array_split(index, n_seperate, axis=0)[::-1]
    # use the latest grid and the other grid to draw lines
    sub_d0 = split_data[0]
    sub_idx0 = split_idx[0]

    # make line data
    x = np.r_[split_idx[1], sub_idx0]
    y = np.r_[split_data[1], sub_d0]
    coef, incpt = get_line_param(sub_idx0, split_idx[1],
                                 sub_d0, split_data[1], is_high=True)
    line_data = coef * index + incpt

    for idx in xrange(2, len(split_data)):
        sub_d = split_data[idx]
        sub_idx = split_idx[idx]
        x = np.r_[sub_idx, sub_idx0]
        y = np.r_[sub_d, sub_d0]

        # check weather line is broken
        sub_line = coef * x + incpt
        over_idx = np.where(sub_line < y)[0]
        # if data go over line too much, make a new line
        n_tolerance = int(n_data * tolerance / n_seperate)
        if len(over_idx) >= n_tolerance:
            # make line data
            coef, incpt = get_line_param(sub_idx0, sub_idx,
                                         sub_d0, sub_d, is_high=True)
            line_data = coef*index + incpt

    if is_high is False:
        line_data = -line_data

    return line_data


def get_line_data(data, n_seperate=5, tolerance=0.1):
    """
    data:
    dict: timeseries data
        {
            "Date": [...],
            "<indicator_id_1>": [...],
            "<indicator_id_2>": [...],
            "<indicator_id_3>": [...],
            "<indicator_id_4>": [...],
            ...
        }
    n_seperate (int): the number of subsets you split data into
    tolerance (float): criteria weather to change trendline

    Return:
    list:
        beg_end_list = [[begin_time, end_time], [begin_time, end_time], ...]
        high_line_list =
            [[begin_value, end_value], [begin_value, end_value], ...]
        low_line_list =
            [[begin_value, end_value], [begin_value, end_value], ...]

    they should satisfy following
    high(low)_line_data[begin(end)_time] == begin(end)_value
    """
    time_data = data['Date']
    high_data = np.array(data['High'])
    low_data = np.array(data['Low'])

    # get line data
    high_data = smoothing(high_data, dim=4)
    low_data = smoothing(low_data, dim=4)
    high_trendline = trendline_backward(
        high_data,
        is_high=True,
        n_seperate=n_seperate,
        tolerance=tolerance
    )
    low_trendline = trendline_backward(
        low_data,
        is_high=False,
        n_seperate=n_seperate,
        tolerance=tolerance
    )

    # they will be return value
    high_line_list = []
    low_line_list = []
    beg_end_list = []
    beg_end_list.append([time_data[0], time_data[-1]])
    high_line_list.append([high_trendline[0], high_trendline[-1]])
    low_line_list.append([low_trendline[0], low_trendline[-1]])

    return beg_end_list, high_line_list, low_line_list
