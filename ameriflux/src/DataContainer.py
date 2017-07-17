import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from statsmodels.sandbox.tools.tools_pca import pcasvd


import statsmodels.formula.api as sm


def linear_fit(df, y, x='num'):
    """ finds parameters of linear fit

    Args:
        df (pd.dataframe): dataframe with data
        y (string): which column analyze?

    Returns:
        sm.ols: statistical linear model
    """
    return sm.ols(formula=y + " ~ " + x, data=df).fit()


def biplot(plt, pca, labels=None, colors=None, xpc=1, ypc=2, scale=1):
    """Generate biplot from the result of pcasvd of statsmodels.

    Parameters
    ----------
    plt : object
        An existing pyplot module reference.

    pca : tuple
        The result from statsmodels.sandbox.tools.tools_pca.pcasvd.

    labels : array_like, optional
        Labels for each observation.

    colors : array_like, optional
        Colors for each observation.

    xpc, ypc : int, optional
        The principal component number for x- and y-axis. Defaults to
        (xpc, ypc) = (1, 2).

    scale : float
        The variables are scaled by lambda ** scale, where lambda =
        singular value = sqrt(eigenvalue), and the observations are
        scaled by lambda ** (1 - scale). Must be in [0, 1].

    Returns
    -------
    None.

    """
    xpc, ypc = (xpc - 1, ypc - 1)
    xreduced, factors, evals, evecs = pca
    singvals = np.sqrt(evals)

    # data
    xs = factors[:, xpc] * singvals[xpc]**(1. - scale)
    ys = factors[:, ypc] * singvals[ypc]**(1. - scale)

    if labels is not None:
        for i, (t, x, y) in enumerate(zip(labels, xs, ys)):
            c = 'k' if colors is None else colors[i]
            plt.text(x, y, t, color=c, ha='center', va='center')
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xpad = (xmax - xmin) * 0.1
        ypad = (ymax - ymin) * 0.1
        plt.xlim(xmin - xpad, xmax + xpad)
        plt.ylim(ymin - ypad, ymax + ypad)
    else:
        colors = 'k' if colors is None else colors
        plt.scatter(xs, ys, c=colors, marker='.')

    # variables
    tvars = np.dot(np.eye(factors.shape[0], factors.shape[1]),
                   evecs) * singvals**scale

    for i, col in enumerate(xreduced.columns.values):
        x, y = tvars[i][xpc], tvars[i][ypc]
        plt.arrow(0, 0, x, y, color='r',
                  width=0.002, head_width=0.05)
        plt.text(x * 1.4, y * 1.4, col, color='r', ha='center', va='center')

    plt.xlabel('PC{}'.format(xpc + 1))
    plt.ylabel('PC{}'.format(ypc + 1))


class DataContainer:
    """docstring for DataContainer"""

    def __init__(self):
        print('loading sites:')
        sites = self.find_unique_sites()
        self.df = pd.DataFrame()
        for s in sites:
            self.df = self.df.append(self.load_pair(s))
        # self.df = self.df.apply(pd.to_numeric, errors='ignore')
        self.df['LOCATION_LAT'] = pd.to_numeric(self.df['LOCATION_LAT'], errors='raise')
        self.df['LOCATION_LONG'] = pd.to_numeric(self.df['LOCATION_LONG'], errors='raise')
        self.df = self.df[np.isfinite(self.df['FC'])]
        self.df = self.df.reset_index()
        self.df['date'] = self.df['TIMESTAMP_START']
        self.df.set_index(['TIMESTAMP_START'], inplace=True)

    def read_measurements_from_file(self, f):
        df = pd.read_csv(f, skiprows=2)
        df = df.convert_objects(convert_numeric=True)
        df = df.replace(-9999, np.NaN)
        try:
            df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
        except:
            print(f)
        return df

    def read_site_description(self, f):
        df = pd.read_excel(f)
        df = df[['VARIABLE', 'DATAVALUE']].set_index('VARIABLE').transpose()
        df = df.drop('REFERENCE_PAPER', 1, errors='ignore')
        df = df.drop('REFERENCE_USAGE', 1, errors='ignore')
        df['site'] = f[9:15]
        return df[['LOCATION_LAT', 'LOCATION_LONG', 'site', 'IGBP', 'CLIMATE_KOEPPEN']]

    def merge_2_df(self, df1, df2):
        for c in df2.columns:
            try:
                df1[c] = df2[c][0]
            except:
                pass
        return df1

    def find_unique_sites(self):
        onlyfiles = [f[9:15] for f in glob.glob('data/*.{}'.format('csv'))]
        return set(onlyfiles)

    def load_pair(self, file_mask):
        print(file_mask)
        csv_file = glob.glob('data/*' + file_mask + '*.csv')[0]
        df1 = self.read_measurements_from_file(csv_file)
        xls_file = glob.glob('data/*' + file_mask + '*.xlsx')[0]
        df2 = self.read_site_description(xls_file)
        # return df2
        return self.merge_2_df(df1, df2)
