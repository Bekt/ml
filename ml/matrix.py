import arff
import copy
import numpy as np

from scipy import stats


class Matrix(object):

    def __init__(self, arff):
        """Creates a new matrix class."""
        self._arff = arff

    @property
    def attributes(self):
        return self._arff['attributes']

    @property
    def data(self):
        return self._arff['data']

    def row(self, ind):
        """Returns the specified row."""
        return self.data[ind]

    def rows(self):
        """Returns number of rows in the matrix."""
        return len(self.data)

    def cols(self):
        """Returns the number of columns in the matrix."""
        return len(self.attributes)

    def attr_name(self, col):
        """Returns the name of the specified attribute."""
        return self.attributes[col][0]

    def is_nominal(self, col):
        """Returns whether the specified column is nominal."""
        return _is_nominal(self.attributes[col])

    def is_numerical(self, col):
        """Returns whether the specified column is numerical."""
        return not self.is_nominal(col)

    def column_mean(self, col):
        """Returns the mean of known (not '?') values in
        the specified column."""
        return np.nanmean(self.data, axis=0)[col]

    def column_max(self, col):
        """Returns the max value in the specified column."""
        return np.nanmax(self.data, axis=0)[col]

    def column_min(self, col):
        """Returns the min valie in the specified column."""
        return np.nanmin(self.data, axis=0)[col]

    def column_common(self, col):
        """Returns the most common value in the specified column."""
        vals, counts = stats.mode(self.data, axis=0)
        return vals[0][col]

    def sub_cols(self, start, end):
        """Returns a new instance of matrix with columns sliced
        from the given start to end.
        All data is a shallow copy.
        TODO: This probably messes up ARFF attributes other than
        attributes and data. Look into this whenever necessary."""
        if start < 0:
            raise IndexError('start needs to be >= 0')
        m = Matrix(copy.copy(self._arff))
        m._arff['attributes'] = m.attributes[start:end]
        m._arff['data'] = m.data[:, start:end]
        return m

    def sub_rows(self, start, end):
        """Returns a new instance of matrix with rows sliced
        from the given start to end.
        All data is a shallow copy."""
        if start < 0:
            raise IndexError('start needs to be >= 0')
        m = Matrix(copy.copy(self._arff))
        m._arff['data'] = m.data[start:end]
        return m


def from_arff(filename):
    """Loads a matrix from an ARFF file and returns
    an instance of Matrix."""
    with open(filename) as f:
        arff_data = _from_arff(f)
        return Matrix(arff_data)


def _from_arff(f):
    res = arff.load(f)
    attrs, data = res['attributes'], res['data']
    if not attrs or not data:
        raise ValueError('ARFF data not valid.')
    # Here, we are:
    #   1. converting nominal values to their corresponding indices.
    #   2. converting the data array to a float64 numpy array.
    nominals = [ind for ind, attr in enumerate(attrs) if _is_nominal(attr)]
    if nominals:
        for row in data:
            for ind in nominals:
                if row[ind]:
                    row[ind] = attrs[ind][1].index(row[ind])
                else:
                    row[ind] = np.nan
    res['data'] = np.float_(res['data'])
    return res


def _is_nominal(attr):
    return type(attr[1]) == list