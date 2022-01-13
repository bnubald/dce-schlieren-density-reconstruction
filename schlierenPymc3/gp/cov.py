import numpy as np

import theano
from theano import tensor as tt

from pymc3.gp.cov import Covariance

class NormGEK(Covariance):
    r"""
    Gaussian Enhanced Kriging. Takes an existing covariance kernel
    and calculates a new kernel which takes in gradient values as
    input
    """
    def __init__(self, input_dim, ls, sigma_f, alphas, kappa=1.0, lsf=None, cov=None, active_dim=None):
        super().__init__(input_dim)
        self.ls = tt.as_tensor_variable(ls)
        if lsf is not None:
            self.lsf = tt.as_tensor_variable(lsf)
        else:
            self.lsf = None

        self.sigma_f = tt.as_tensor_variable(sigma_f)
        self.alphas = tt.as_tensor_variable(alphas)
        self.kappa = tt.as_tensor_variable(kappa)
        self.active_dim = active_dim

    def full(self, X=None, Xs=None, kappa=None, gp=False):
        X, X_df = X[0], X[1]
        X, Xs = self._slice(X, Xs)

        if kappa == None:
            kappa = self.kappa

        # Check if lengthscale specific to density field is provided,
        # if so, use lsf for just density block of covar matrix
        if (self.lsf is None) or (gp == True):
            kernel = self.kernel
        else:
            kernel = self.kernel_lsf

        if X is not None:
            K11 = kernel(X, Xs)*kappa
            if X_df is None:
                return K11

        if X is not None:
            K10 = self.kernel_derivative(X, X_df)
        else:
            K10 = self.kernel_derivative(Xs, X_df)
            return K10

        K01 = K10.T
        K00 = self.kernel_derivative_derivative(X_df, X_df)

        row1 = tt.concatenate([K00, K01], axis=1)
        row2 = tt.concatenate([K10, K11], axis=1)

        Kxx =  tt.concatenate([row1, row2], axis=0)

        return Kxx

    def dist(self, X1, X2):
        return np.sum(X1.reshape(-1, 1), axis=1).reshape(-1, 1) - \
                np.sum(X2.reshape(-1, 1), axis=1)

    def square_dist(self, X, Xs, ls=None):
        if ls is None:
            ls = self.ls

        X = tt.mul(X, ls)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, self.ls)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

    def kernel(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.square(self.sigma_f) * tt.exp(-0.5 * self.square_dist(X, Xs))

    def kernel_lsf(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.square(self.sigma_f) * tt.exp(-0.5 * self.square_dist(X, Xs, self.lsf))

    def kernel_derivative(self, X1, X2):
        m = len(X1)
        n = len(X2)
        d = X1.shape[1]

        R = self.kernel(X1, X2)

        # First derivatives
        dRdxi = np.zeros( ( m, n*d ) )
        dRdxi = theano.shared(dRdxi)
        for i in range(d):
            dxi = self.dist(X1[:,i], X2[:,i])
            s1 = - tt.square(self.ls[i]) * R * dxi
            s2 = s1 * self.alphas[i]
            dRdxi = tt.set_subtensor(dRdxi[:, n*i:n*(i+1)], s2 )


        return -dRdxi

    def kernel_derivative_derivative(self, X1, X2):
        m = len(X1)
        n = len(X2)
        d = X1.shape[1]

        R = self.kernel(X1, X2)
        dxi = []

        # Second derivatives
        d2Rdxi2 = np.zeros( ( m*d, n*d ) )
        d2Rdxi2 = theano.shared(d2Rdxi2)

        for i in range(d):
            dxi.append( self.dist(X1[:,i], X2[:,i]) )
            lscale = self.ls[i]
            s1 = lscale**2.0 * R * ( lscale**2.0 * dxi[i]**2.0 - 1 )
            s2 = s1 * tt.square( self.alphas[i] )
            d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ n*i:n*(i+1), n*i:n*(i+1) ], s2)


        # Cross derivatives
        if (d == 2):
            s1 = self.ls[0]**2.0 * self.ls[1]**2.0 * R * dxi[0] * dxi[1]
            s2 = s1 * self.alphas[0] * self.alphas[1]

            d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ n:n*d, :n ], s2)
            d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ :n, n:n*d ], s2)

        return -d2Rdxi2

    def _slice(self, X, Xs):
        if self.active_dim is None:
            if Xs is None:
                return X, Xs
            else:
                return X, Xs
        else:
            if Xs is None:
                return X[:,self.active_dim].reshape(len(X), 1), Xs
            else:
                return X[:,self.active_dim].reshape(len(X), 1), Xs[:,self.active_dim].reshape(len(Xs), 1)

    def diag(self, X):
        return tt.alloc(1.0, X.shape[0])


class NormGEKPartial(Covariance):
    r"""
    Partial Gaussian Enhanced Kriging. Takes an existing covariance kernel
    and calculates a new kernel which takes in either x or y gradient values
    as input
    """
    def __init__(self, input_dim, ls, sigma_f, alphas, cov=None, active_dim=None):
        super().__init__(input_dim)
        self.ls = tt.as_tensor_variable(ls)
        self.sigma_f = tt.as_tensor_variable(sigma_f)
        self.alphas = tt.as_tensor_variable(alphas)
        self.active_dim = active_dim

    def full(self, X=None, Xs=None):
        X, X_df = X[0], X[1]
        X, Xs = self._slice(X, Xs)

        if X is not None:
            K11 = self.kernel(X, Xs)
            if X_df is None:
                return K11

        if X is not None:
            K10 = self.kernel_derivative(X, X_df)
        else:
            K10 = self.kernel_derivative(Xs, X_df)
            return K10

        K01 = K10.T
        K00 = self.kernel_derivative_derivative(X_df, X_df)

        row1 = tt.concatenate([K00, K01], axis=1)
        row2 = tt.concatenate([K10, K11], axis=1)

        Kxx =  tt.concatenate([row1, row2], axis=0)

        return Kxx

    def dist(self, X1, X2):
        return np.sum(X1.reshape(-1, 1), axis=1).reshape(-1, 1) - \
                np.sum(X2.reshape(-1, 1), axis=1)

    def square_dist(self, X, Xs):
        X = tt.mul(X, self.ls)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, self.ls)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

    def kernel(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.square(self.sigma_f) * tt.exp(-0.5 * self.square_dist(X, Xs))

    def kernel_derivative(self, X1, X2):
        m = len(X1)
        n = len(X2)

        d = X1.shape[1]

        R = self.kernel(X1, X2)

        active_dim = self.active_dim
        if active_dim != None:
            d = 1

        # First derivatives
        dRdxi = np.zeros( ( m, n*d ) )
        dRdxi = theano.shared(dRdxi)
        for i in range(d):
            if active_dim != None:
                i = active_dim
                alpha = self.alphas
            else:
                alpha = self.alphas[i]
            dxi = self.dist(X1[:,i], X2[:,i])
            s1 = - tt.square(self.ls[i]) * R * dxi
            s2 = s1 * alpha
            if active_dim != None:
                dRdxi = s2
            else:
                dRdxi = tt.set_subtensor(dRdxi[:, n*i:n*(i+1)], s2 )

        return -dRdxi

    def kernel_derivative_derivative(self, X1, X2):
        m = len(X1)
        n = len(X2)
        d = X1.shape[1]

        R = self.kernel(X1, X2)
        dxi = []

        active_dim = self.active_dim
        if active_dim != None:
            d = 1

        # Second derivatives
        d2Rdxi2 = np.zeros( ( m*d, n*d ) )
        d2Rdxi2 = theano.shared(d2Rdxi2)

        for i in range(d):
            if active_dim != None:
                i = active_dim
                j = 0
                alpha = self.alphas
            else:
                j = i
                alpha = self.alphas[i]
            dxi.append( self.dist(X1[:,i], X2[:,i]) )
            lscale = self.ls[i]
            s1 = lscale**2.0 * R * ( lscale**2.0 * dxi[j]**2.0 - 1 )
            s2 = s1 * tt.square( alpha )
            if active_dim != None:
                d2Rdxi2 = s2
            else:
                d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ n*i:n*(i+1), n*i:n*(i+1) ], s2)

        # Cross derivatives
        if (d == 2):
            s1 = self.ls[0]**2.0 * self.ls[1]**2.0 * R * dxi[0] * dxi[1]
            s2 = s1 * self.alphas[0] * self.alphas[1]

            d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ n:n*d, :n ], s2)
            d2Rdxi2 = tt.set_subtensor(d2Rdxi2[ :n, n:n*d ], s2)

        return -d2Rdxi2

    def _slice(self, X, Xs):
        if Xs is None:
            return X, Xs
        else:
            return X, Xs

    def diag(self, X):
        return tt.alloc(1.0, X.shape[0])
