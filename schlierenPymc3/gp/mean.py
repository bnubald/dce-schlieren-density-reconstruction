import theano.tensor as tt

from pymc3.gp.mean import Mean

__all__ = ["SchlierenNorm"]

class SchlierenNorm(Mean):
    R"""
    Constant mean function for Gaussian process.
    Parameters
    ----------
    bx: variable, array or integer
        Constant mean value
    by: variable, array or integer
        Constant mean value
    c: variable, array or integer
        Constant mean value
    """

    def __init__(self, bx=0.0, by=0.0, c=0.0):
        Mean.__init__(self)
        self.bnucustom = True
        self.bx = -bx
        self.by = -by
        self.c  = c

    def __call__(self, X_f=None, X_df=None):
        if X_f is not None:
            m = X_f.shape[0]
        else:
            m = 0

        if X_df is not None:
            n = X_df.shape[0]
            d = X_df.shape[1]
        else:
            n, d = 0, 0

        X_shape = m + n*d
        mn = tt.alloc(0.0, X_shape) * self.c

        mn = tt.set_subtensor(mn[:n], self.bx )
        if d == 2:
            mn = tt.set_subtensor(mn[n:n*2], self.by )

        if X_f is not None:
            if d == 1:
                mn = tt.set_subtensor(mn[n:X_shape], self.c )
            elif d == 2:
                mn = tt.set_subtensor(mn[n*2:X_shape], self.c )

        return mn


class SchlierenNormPartial(Mean):
    R"""
    Constant mean function for Gaussian process.
    Parameters
    ----------
    bx: variable, array or integer
        Constant mean value
    by: variable, array or integer
        Constant mean value
    c: variable, array or integer
        Constant mean value
    """

    def __init__(self, bx=0.0, by=0.0, c=0.0):
        Mean.__init__(self)
        self.bnucustom = True
        self.bx = -bx
        self.by = -by
        self.c  = c

    def __call__(self, X_f=None, X_df=None):
        if X_f is not None:
            m = X_f.shape[0]
        else:
            m = 0

        if X_df is not None:
            n = X_df.shape[0]
            d = 1
        else:
            n, d = 0, 0

        X_shape = m + n*d
        mn = tt.alloc(0.0, X_shape) * self.c

        mn = tt.set_subtensor(mn[:n], self.bx )
        if d == 2:
            mn = tt.set_subtensor(mn[n:n*2], self.by )

        if X_f is not None:
            if d == 1:
                mn = tt.set_subtensor(mn[n:X_shape], self.c )
            elif d == 2:
                mn = tt.set_subtensor(mn[n*2:X_shape], self.c )

        return mn
