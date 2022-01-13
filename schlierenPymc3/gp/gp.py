import numpy as np
import pymc3 as pm

from theano import tensor as tt

from pymc3.gp.cov import Covariance
from pymc3.gp.util import (conditioned_vars, infer_shape,
                           stabilize, cholesky, solve_lower)
from pymc3.distributions import draw_values

__all__ = ["MarginalNormGEK", "MarginalNormGEKPartial"]

@conditioned_vars(["X", "y", "noise_y", "noise_dy"])
class MarginalNormGEK(pm.gp.gp.Base):
    R"""
    Marginal Gaussian process for GEK - for normalised Schlieren inputs.

    The `gp.MarginalNormGEK` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed with associated
    function and gradient values.

    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Marginal(cov_func=cov_func)

            # Place a GP prior over the function f.
            sigma_y = pm.HalfCauchy("sigma_y", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise_y=sigma)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def _build_marginal_likelihood(self, X=None, X_df=None, noise_y=None,
                                   noise_dy=None):
        X_shape = X.shape[0] + X_df.shape[0]*X_df.shape[1]
        if "bnucustom" in dir( self.mean_func ):
            mu = self.mean_func(X_f=X, X_df=X_df)
        else:
            mu = tt.alloc(0.0, X_shape)
        Kxx = self.cov_func(X=[X, X_df])
        if not isinstance(noise_y, Covariance):
            Knx_y = tt.alloc(tt.square(noise_y), X.shape[0])
        if not isinstance(noise_dy, Covariance):
            Knx_dy = tt.alloc(tt.square(noise_dy), X_df.shape[0]*X_df.shape[1])
        Knx = tt.diag( tt.concatenate( [Knx_dy, Knx_y] ) )
        cov = Kxx + Knx
        self.Kxx = cov

        return mu, cov

    def marginal_likelihood(self, name, X=None, X_df=None, y=None, noise_y=None,
                            noise_dy=None, is_observed=True, **kwargs):
        R"""
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.

        This is integral over the product of the GP prior and a normal likelihood.

        .. math::

           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        X_df: array-like
            Gradient input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise_y: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise for input function.
            Can also be a Covariance for non-white noise.
        noise_dy: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise for input gradient.
            Can also be a Covariance for non-white noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        mu, cov = self._build_marginal_likelihood(X, X_df, noise_y, noise_dy)
        self.X = X
        self.X_df = X_df
        self.y = y
        self.noise_y = noise_y
        self.noise_dy = noise_dy
        if is_observed:
            return pm.MvNormal(name, mu=mu, cov=cov, observed=y, **kwargs)
        else:
            shape = infer_shape(X, kwargs.pop("shape", None))
            return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def _get_given_vals(self, given):
        if given is None:
            given = {}

        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'y', 'noise_y', 'noise_dy']):
            X, y, noise_y, noise_dy = given['X'], given['y'], given['noise_y'],\
                                      given['noise_dy']
        else:
            X, y, noise_y, noise_dy = self.X, self.y, self.noise_y,\
                                      self.noise_dy
        return X, y, noise_y, noise_dy, cov_total, mean_total

    def _build_conditional(self, Xnew, pred_noise, diag, X, y, noise_y,
                           noise_dy, cov_total, mean_total):

        X = self.X
        X_df = self.X_df
        X_shape = X.shape[0] + X_df.shape[0]*X_df.shape[1]
        Kxx = self.Kxx

        #Calc R:
        K01 = self.cov_func([X, None], Xnew).T
        K00 = self.cov_func([None, X_df], Xnew)

        Kxs = tt.concatenate([K00, K01], axis=1).T

        if "bnucustom" in dir( self.mean_func ):
            rxx = y - mean_total(X_f=X, X_df=X_df)
        else:
            rxx = y

        # ## Solve using cholesky
        # # Solve for posterior density
        # L = cholesky(stabilize(Kxx))
        # A = solve_lower(L, Kxs)
        # v = solve_lower(L, rxx)
        # mu = self.mean_func(X_f=Xnew, X_df=None) + tt.dot(tt.transpose(A), v)

        # # Solve for posterior gradients
        # n_d = X_df.shape[0]
        # dx = rxx[:n_d]
        # dy = rxx[n_d:n_d*2]

        # Kxs_d = self.cov_func.full(X=[X_df, None], Xs=Xnew, gp=True, kappa=1.0)
        # Kxx_d = self.cov_func.full(X=[X_df, None], gp=True, kappa=1.0)
        # Knx_d = tt.alloc(tt.square(noise_dy), X_df.shape[0])
        # L_d = cholesky(stabilize(Kxx_d + Knx_d))
        # A_d = solve_lower(L_d, Kxs_d)
        # vx = solve_lower(L_d, dx)
        # vy = solve_lower(L_d, dy)

        # mu_dx = tt.dot(tt.transpose(A_d), vx)
        # mu_dy = tt.dot(tt.transpose(A_d), vy)

        # if diag:
        #     Kss = self.cov_func(Xnew, diag=True)
        #     var = Kss - tt.sum(tt.square(A), 0)
        #     return mu, var, mu_dx, mu_dy
        # else:
        #     Kss = self.cov_func([Xnew, None], Xnew).T
        #     cov = Kss - tt.dot(tt.transpose(A), A)
        #     return mu, cov if pred_noise else stabilize(cov), mu_dx, mu_dy


        ## Solve using full inverse
        # Solve for posterior density
        Kinv = tt.nlinalg.matrix_inverse(Kxx)
        # print("Kxs shape:", Kxs.T.eval().shape)
        # print("Kinv shape:", Kinv.eval().shape)
        # print("y shape:", y.shape)

        mu = tt.dot( Kxs.T, tt.dot(Kinv, rxx) )
        # print("mu shape:", mu.eval().shape)

        # Solve for posterior gradients
        Kxs_d = self.cov_func.full(X=[X_df, None], Xs=Xnew, gp=True, kappa=1.0)
        Kxx_d = self.cov_func.full(X=[X_df, None], gp=True, kappa=1.0)
        Knx_d = tt.alloc(tt.square(noise_dy), X_df.shape[0])
        Kinv_d = tt.nlinalg.matrix_inverse(stabilize(Kxx_d + Knx_d))
        n_d = X_df.shape[0]
        dx = rxx[:n_d]
        dy = rxx[n_d:n_d*2]

        # mu_dx = 1/self.cov_func.alphas[0]*tt.dot( Kxs_d.T, tt.dot(Kinv_d, dx) )
        # mu_dy = 1/self.cov_func.alphas[1]*tt.dot( Kxs_d.T, tt.dot(Kinv_d, dy) )
        mu_dx = tt.dot( Kxs_d.T, tt.dot(Kinv_d, dx) )
        mu_dy = tt.dot( Kxs_d.T, tt.dot(Kinv_d, dy) )

        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.dot( Kxs.T, tt.dot(Kinv, Kxs) )
            # if pred_noise:
            #     var += noise(Xnew, diag=True)
            return mu, var, mu_dx, mu_dy
        else:
            # Kss = self.cov_func(Xnew)
            Kss = self.cov_func([Xnew, None], Xnew).T
            cov = Kss - tt.dot( Kxs.T, tt.dot(Kinv, Kxs) )
            # if pred_noise:
            #     cov += noise(Xnew)
            return mu, cov if pred_noise else stabilize(cov), mu_dx, mu_dy

    def conditional(self, name, Xnew, pred_noise=False, given=None, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that the GP prior was over, the
        conditional distribution over a set of new points, `f_*` is:

        .. math::

           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Can optionally take as key value pairs: `X`, `y`, `noise_y`,
            , `noise_dy` and `gp`.  See the section in the documentation on additive GP
            models in PyMC3 for more information.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        givens = self._get_given_vals(given)
        mu, cov, mu_dx, mu_dy = self._build_conditional(Xnew, pred_noise, False, *givens)
        shape = infer_shape(Xnew, kwargs.pop("shape", None))
        return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def predict(self, Xnew, point=None, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point: pymc3.model.Point
            A specific point to condition on.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        if given is None:
            given = {}

        mu, cov, mu_dx, mu_dy = self.predictt(Xnew, diag, pred_noise, given)
        return draw_values([mu, cov, mu_dx, mu_dy], point=point)

    def predict_trace(self, Xnew, trace=None, retcov=True, diag=False, \
                      pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        trace: pymc3.backends.base.MultiTrace
            A return of pm.sample().
        retcov: boolean
            If `True`, returns posterior covariance as well.
            If `False`, returns None for posterior covariance.
            Default is `True`
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        if given is None:
            given = {}

        from tqdm import tqdm
        nPoints = len( list(trace.points()) )
        pbar = tqdm(total=nPoints)

        #Define symbolic theano mean/covariance
        mu, cov, mu_dx, mu_dy = self.predictt(Xnew, diag, pred_noise, given)

        mu_z = np.zeros( (Xnew.shape[0], 1) )
        mu_dx_z = np.zeros( (Xnew.shape[0], 1) )
        mu_dy_z = np.zeros( (Xnew.shape[0], 1) )
        if retcov == True:
            cov_z = np.zeros( (Xnew.shape[0], Xnew.shape[0] ) )
            for i, point in enumerate( trace.points() ):
                #Pass result of mcmc trace
                post_mean, post_cov, post_dx, post_dy = draw_values([mu, cov, mu_dx, mu_dy], point=point)

                mu_z += post_mean.reshape(-1, 1)
                cov_z += post_cov
                mu_dx_z += post_dx.reshape(-1, 1)
                mu_dy_z += post_dy.reshape(-1, 1)

                pbar.update(1)

            posterior_mean = (mu_z / nPoints)
            posterior_cov = (cov_z / nPoints)
            posterior_dx = (mu_dx_z / nPoints)
            posterior_dy = (mu_dy_z / nPoints)
        else:
            for i, point in enumerate( trace.points() ):
                #Pass result of mcmc trace
                post_mean = draw_values([mu], point=point)
                mu_z += np.array(post_mean).reshape(-1, 1)
                post_dx = draw_values([mu_dx], point=point)
                mu_dx_z += np.array(post_dx).reshape(-1, 1)
                post_dy = draw_values([mu_dy], point=point)
                mu_dy_z += np.array(post_dy).reshape(-1, 1)

                pbar.update(1)

            posterior_mean = (mu_z / nPoints)
            posterior_cov = None
            posterior_dx = (mu_dx_z / nPoints)
            posterior_dy = (mu_dy_z / nPoints)

        pbar.close()

        return posterior_mean, posterior_cov, posterior_dx, posterior_dy

    def predictt(self, Xnew, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        givens = self._get_given_vals(given)
        mu, cov, mu_dx, mu_dy = self._build_conditional(Xnew, pred_noise, diag, *givens)
        return mu, cov, mu_dx, mu_dy

    def get_cov( self, point=None ):
        R"""
        Return covariance matrix as a numpy array.
        """
        if self.Kxx is not None:
            if point == None:
                return self.Kxx.eval()
            else:
                return draw_values([self.Kxx], point=point)
        else:
            print( "Covariance kernel not calculated yet" )
            print( "Need to run gp.marginal_likelihood first" )

    def plot_sv( self ):
        R"""
        Plot singular values of the covariance matrix.
        """
        if self.Kxx is not None:
            U, S, Vh = np.linalg.svd( self.Kxx.eval() )
            svals = np.arange( 0, len(S), 1 )

            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(svals, S, '-k')
            plt.title('Singular value plot')
            plt.xlabel('Number of singular values')
            plt.ylabel('Magnitude of singular value')
            plt.draw()
        else:
            print( "Need to run gp.marginal_likelihood first" )


@conditioned_vars(["X", "y", "noise_y", "noise_dy"])
class MarginalNormGEKPartial(pm.gp.gp.Base):
    R"""
    Marginal Gaussian process for GEK - for normalised Schlieren inputs.

    The `gp.MarginalNormGEK` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed with associated
    function and gradient values.

    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Marginal(cov_func=cov_func)

            # Place a GP prior over the function f.
            sigma_y = pm.HalfCauchy("sigma_y", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise_y=sigma)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def _build_marginal_likelihood(self, X=None, X_df=None, noise_y=None,
                                   noise_dy=None):
        X_shape = X.shape[0] + X_df.shape[0]
        if "bnucustom" in dir( self.mean_func ):
            mu = self.mean_func(X_f=X, X_df=X_df)
        else:
            mu = tt.alloc(0.0, X_shape)
        Kxx = self.cov_func(X=[X, X_df])
        if not isinstance(noise_y, Covariance):
            Knx_y = tt.alloc(tt.square(noise_y), X.shape[0])
        if not isinstance(noise_dy, Covariance):
            Knx_dy = tt.alloc(tt.square(noise_dy), X_df.shape[0])
        Knx = tt.diag( tt.concatenate( [Knx_dy, Knx_y] ) )
        cov = Kxx + Knx
        self.Kxx = cov

        return mu, cov

    def marginal_likelihood(self, name, X=None, X_df=None, y=None, noise_y=None,
                            noise_dy=None, is_observed=True, **kwargs):
        R"""
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.

        This is integral over the product of the GP prior and a normal likelihood.

        .. math::

           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        X_df: array-like
            Gradient input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise_y: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise for input function.
            Can also be a Covariance for non-white noise.
        noise_dy: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise for input gradient.
            Can also be a Covariance for non-white noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        mu, cov = self._build_marginal_likelihood(X, X_df, noise_y, noise_dy)
        self.X = X
        self.X_df = X_df
        self.y = y
        self.noise_y = noise_y
        self.noise_dy = noise_dy
        if is_observed:
            return pm.MvNormal(name, mu=mu, cov=cov, observed=y, **kwargs)
        else:
            shape = infer_shape(X, kwargs.pop("shape", None))
            return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def _get_given_vals(self, given):
        if given is None:
            given = {}

        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'y', 'noise_y', 'noise_dy']):
            X, y, noise_y, noise_dy = given['X'], given['y'], given['noise_y'],\
                                      given['noise_dy']
        else:
            X, y, noise_y, noise_dy = self.X, self.y, self.noise_y,\
                                      self.noise_dy
        return X, y, noise_y, noise_dy, cov_total, mean_total

    def _build_conditional(self, Xnew, pred_noise, diag, X, y, noise_y,
                           noise_dy, cov_total, mean_total):

        X = self.X
        X_df = self.X_df
        X_shape = X.shape[0] + X_df.shape[0]*X_df.shape[1]
        Kxx = self.Kxx

        #Calc R:
        K01 = self.cov_func([X, None], Xnew).T
        K00 = self.cov_func([None, X_df], Xnew)

        Kxs = tt.concatenate([K00, K01], axis=1).T

        if "bnucustom" in dir( self.mean_func ):
            rxx = y - mean_total(X_f=X, X_df=X_df)
        else:
            rxx = y

        ## Solve using cholesky
        # Solve for posterior density
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(X_f=Xnew, X_df=None) + tt.dot(tt.transpose(A), v)

        active_dim = self.cov_func.active_dim

        # Solve for posterior gradients
        n_d = X_df.shape[0]

        Kxs_d = self.cov_func(X=[X_df, None], Xs=Xnew)
        Kxx_d = self.cov_func(X=[X_df, None])
        Knx_d = tt.alloc(tt.square(noise_dy), X_df.shape[0])
        L_d = cholesky(stabilize(Kxx_d + Knx_d))
        A_d = solve_lower(L_d, Kxs_d)
        if (active_dim == 0):
            if active_dim != None:
                alpha = self.cov_func.alphas
            else:
                alpha = self.cov_func.alphas[0]
            dx = rxx[:n_d]
            vx = solve_lower(L_d, dx)
            mu_dx = tt.dot(tt.transpose(A_d), vx)
            mu_dy = tt.zeros_like(mu_dx)
        elif active_dim == 1:
            if active_dim != None:
                dy = rxx[:n_d]
                alpha = self.cov_func.alphas
            else:
                dy = rxx[n_d:n_d*2]
                alpha = self.cov_func.alphas[1]
            vy = solve_lower(L_d, dy)
            mu_dy = tt.dot(tt.transpose(A_d), vy)
            mu_dx = tt.zeros_like(mu_dy)

        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.sum(tt.square(A), 0)
            return mu, var, mu_dx, mu_dy
        else:
            Kss = self.cov_func([Xnew, None], Xnew).T
            cov = Kss - tt.dot(tt.transpose(A), A)
            return mu, cov if pred_noise else stabilize(cov), mu_dx, mu_dy

    def conditional(self, name, Xnew, pred_noise=False, given=None, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that the GP prior was over, the
        conditional distribution over a set of new points, `f_*` is:

        .. math::

           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Can optionally take as key value pairs: `X`, `y`, `noise_y`,
            , `noise_dy` and `gp`.  See the section in the documentation on additive GP
            models in PyMC3 for more information.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        givens = self._get_given_vals(given)
        mu, cov, mu_dx, mu_dy = self._build_conditional(Xnew, pred_noise, False, *givens)
        shape = infer_shape(Xnew, kwargs.pop("shape", None))
        return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def predict(self, Xnew, point=None, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point: pymc3.model.Point
            A specific point to condition on.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        if given is None:
            given = {}

        mu, cov, mu_dx, mu_dy = self.predictt(Xnew, diag, pred_noise, given)
        return draw_values([mu, cov, mu_dx, mu_dy], point=point)

    def predict_trace(self, Xnew, trace=None, retcov=True, diag=False, \
                      pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        trace: pymc3.backends.base.MultiTrace
            A return of pm.sample().
        retcov: boolean
            If `True`, returns posterior covariance as well.
            If `False`, returns None for posterior covariance.
            Default is `True`
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        if given is None:
            given = {}

        from tqdm import tqdm
        nPoints = len( list(trace.points()) )
        pbar = tqdm(total=nPoints)

        #Define symbolic theano mean/covariance
        mu, cov, mu_dx, mu_dy = self.predictt(Xnew, diag, pred_noise, given)

        mu_z = np.zeros( (Xnew.shape[0], 1) )
        mu_dx_z = np.zeros( (Xnew.shape[0], 1) )
        mu_dy_z = np.zeros( (Xnew.shape[0], 1) )
        if retcov == True:
            cov_z = np.zeros( (Xnew.shape[0], Xnew.shape[0] ) )
            for i, point in enumerate( trace.points() ):
                #Pass result of mcmc trace
                post_mean, post_cov, post_dx, post_dy = draw_values([mu, cov, mu_dx, mu_dy], point=point)

                mu_z += post_mean.reshape(-1, 1)
                cov_z += post_cov
                mu_dx_z += post_dx.reshape(-1, 1)
                mu_dy_z += post_dy.reshape(-1, 1)

                pbar.update(1)

            posterior_mean = (mu_z / nPoints)
            posterior_cov = (cov_z / nPoints)
            posterior_dx = (mu_dx_z / nPoints)
            posterior_dy = (mu_dy_z / nPoints)
        else:
            for i, point in enumerate( trace.points() ):
                #Pass result of mcmc trace
                post_mean = draw_values([mu], point=point)
                mu_z += np.array(post_mean).reshape(-1, 1)
                post_dx = draw_values([mu_dx], point=point)
                mu_dx_z += np.array(post_dx).reshape(-1, 1)
                post_dy = draw_values([mu_dy], point=point)
                mu_dy_z += np.array(post_dy).reshape(-1, 1)

                pbar.update(1)

            posterior_mean = (mu_z / nPoints)
            posterior_cov = None
            posterior_dx = (mu_dx_z / nPoints)
            posterior_dy = (mu_dy_z / nPoints)

        pbar.close()

        return posterior_mean, posterior_cov, posterior_dx, posterior_dy

    def predictt(self, Xnew, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        givens = self._get_given_vals(given)
        mu, cov, mu_dx, mu_dy = self._build_conditional(Xnew, pred_noise, diag, *givens)
        return mu, cov, mu_dx, mu_dy

    def get_cov( self, point=None ):
        R"""
        Return covariance matrix as a numpy array.
        """
        if self.Kxx is not None:
            if point == None:
                return self.Kxx.eval()
            else:
                return draw_values([self.Kxx], point=point)
        else:
            print( "Covariance kernel not calculated yet" )
            print( "Need to run gp.marginal_likelihood first" )

    def plot_sv( self ):
        R"""
        Plot singular values of the covariance matrix.
        """
        if self.Kxx is not None:
            U, S, Vh = np.linalg.svd( self.Kxx.eval() )
            svals = np.arange( 0, len(S), 1 )

            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(svals, S, '-k')
            plt.title('Singular value plot')
            plt.xlabel('Number of singular values')
            plt.ylabel('Magnitude of singular value')
            plt.draw()
        else:
            print( "Need to run gp.marginal_likelihood first" )
