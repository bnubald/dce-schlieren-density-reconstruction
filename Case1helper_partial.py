import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
import pandas as pd
import pickle
sns.set_theme(style="ticks", font_scale=2.0)

class analytical():
    """
    Helper class for functionally defined density field.
    """
    def __init__(self, randseed=False):
        self.cmap = plt.cm.inferno
        self.gcmap = plt.cm.Greys_r
        self.interp = "bilinear"
        if randseed != False:
            np.random.seed(randseed)
            random.seed(randseed)

    def __normaliseimg(self, data):
        """
        Private function:: Normalise values to pixel range 0-255

        :param data np.array:
            Numpy array of image vaues.

        :return:
            Normalised pixel values.
        """
        alpha = 255.0/(np.max(data) - np.min(data))
        beta = alpha*np.min(data)
        scal = alpha*data - beta
        return scal

    def create_grid(self, xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0, N=10):
        """
        Create a uniform grid mesh

        :param xmin float:
            Minimum x for uniform grid.
        :param ymin float:
            Minimum y for uniform grid.
        :param xmax float:
            Maximum x for uniform grid.
        :param ymax float:
            Maximum y for uniform grid.
        :param ymax float:
            Maximum y for uniform grid.
        :param N float:
            No. of subdivisions in uniform mesh.

        :return:
            self.Xo np.array:
                Array of x-coordinates
            self.Yo np.array:
                Array of y-coordinates
            self.Xorig np.array:
                2D array of input coordinates
        """
        self.N = N
        x1 = np.linspace(xmin, xmax, N)
        x2 = np.linspace(ymin, ymax, N)
        self.Xo, self.Yo = np.meshgrid(x1, x2)
        self.Xorig = np.hstack([self.Xo.reshape(N*N, 1), self.Yo.reshape(N*N, 1)])

        # Convert image coordinates to 2D array
        self.img_coords = np.c_[self.Xo.ravel(), self.Yo.ravel()]

        return self.Xo, self.Yo, self.Xorig

    def func_eval(self, func, normalise=True):
        """
        Evaluate input function and normalised/true gradients across uniform grid

        :param func function:
            Function returning density and its gradients in x and y.
        :param normalise boolean:
            Boolean on whether to normalise returned data to between 0-255.

        :return:
            self.density np.array:
                Array of density (rho) values evaluated across uniform grid
            self.dx np.array:
                Array of d_\tilde{rho}/dx values evaluated across uniform grid
            self.dy np.array:
                Array of d_\tilde{rho}/dy values evaluated across uniform grid
        """
        self.density, self.dx, self.dy = func(self.Xo, self.Yo)
        if normalise == True:
            self.sch_dx = self.__normaliseimg( self.dx )
            self.sch_dy = self.__normaliseimg( self.dy )
            return self.density, self.sch_dx, self.sch_dy
        else:
            self.sch_dx = self.dx
            self.sch_dy = self.dy
            return self.density, self.dx, self.dy

    def get_true_grads(self):
        """
        Return normalised/true gradient values

        :return:
            self.dx np.array:
                Array of d_\tilde{rho}/dx values evaluated across uniform grid
            self.dy np.array:
                Array of d_\tilde{rho}/dy values evaluated across uniform grid
        """
        return self.dx, self.dy

    def gen_all_pts(self):
        """
        Return all coordinates and corresponding gradients and density values

        :return:
            X np.array:
                2D array of all coordinates in uniform grid
            y np.array:
                Array of all [d_\tilde{rho}/dx, d_\tilde{rho}/dy] values from input image
            d np.array:
                Array of all density (rho) values from input image
        """
        img_coords = self.img_coords
        d = self.density
        vals_dx = self.sch_dx.reshape(-1, 1)
        vals_dy = self.sch_dy.reshape(-1, 1)

        X = img_coords
        y = np.c_[vals_dx, vals_dy]

        self.X = X
        self.y = y

        return X, y, d

    def gen_subsample_pts(self, step=2):
        """
        Return subsample of coordinates and corresponding gradients and density values from
            uniform grid

        :param step int:
            Extract every `step` along x and y of uniform grid

        :return:
            X np.array:
                2D array of every `step` in uniform grid
            y np.array:
                Array of every `step` [d_\tilde{rho}/dx, d_\tilde{rho}/dy] values from 
                input image
            d np.array:
                Array of every `step` density (rho) values from input image
        """
        Xo, Yo = self.Xo, self.Yo
        N = self.N
        ss = step

        ## Use point subsampling for training
        Xo_ss = Xo[::ss, ::ss]
        Yo_ss = Yo[::ss, ::ss]
        img_coords_ss = np.c_[Xo_ss.ravel(), Yo_ss.ravel()]
        d = self.density[::ss]
        vals_dx = self.sch_dx.reshape(N, N)[::ss, ::ss].reshape(-1, 1)
        vals_dy = self.sch_dy.reshape(N, N)[::ss, ::ss].reshape(-1, 1)

        X = img_coords_ss
        y = np.c_[vals_dx, vals_dy]

        self.X = X
        self.y = y

        return X, y, d

    def gen_ran_pts(self, H=200, min_spacing=0.0):
        """
        Return sample of random coordinates and corresponding gradients and density
            values from uniform grid

        :param H int:
            Extract this many samples from uniform grid
        :param min_spacing float:
            Sampled points must be at least this distance away from each other

        :return:
            X np.array:
                2D array of every sampled in uniform grid
            y np.array:
                Array of randomly sampled [d_\tilde{rho}/dx, d_\tilde{rho}/dy] values
                from input image
            d np.array:
                Array of randomly sampled density (rho) values from input image
        """
        # Downsample points with minimum distance between points
        from pykdtree.kdtree import KDTree

        Xo, Yo = self.Xo, self.Yo

        img_coords = self.img_coords
        density = self.density
        vals_dx = self.sch_dx.reshape(-1, 1)
        vals_dy = self.sch_dy.reshape(-1, 1)

        X = np.zeros((H, 2), dtype=np.float64)
        d = np.zeros((H))
        y = np.zeros((H, 2))
        count  = 0

        ranidx = list(range(0, img_coords.shape[0]-1))
        random.shuffle(ranidx)
        ranidx_cpy = ranidx.copy()

        for i, idx in enumerate(ranidx):
            if count == H:
                break
            point = img_coords[idx, :].reshape(1, -1)
            if i > 0:
                kd_tree = KDTree(np.array(X))
                dist, index = kd_tree.query(point, k=1)
                if dist > min_spacing :
                    X[count, :] = point
                    y[count, :] = [ vals_dx[idx], vals_dy[idx] ]
                    d[count] = density.reshape(-1, 1)[idx]
                    count += 1
            else:
                X[count, :] = point
                y[count, :] = [ vals_dx[idx], vals_dy[idx] ]
                d[count] = density.reshape(-1, 1)[idx]
                count += 1
            if i == len(ranidx) - 1:
                H = count + 1
                X = X[ :count, :]
                y = y[ :count, :]

        self.X = X
        self.y = y

        print("No. of training points:", len(X))

        return X, y, d

    def get_ran_density(self, npts=3):
        """
        Return randomly sampled density values

        :param npts int:
            No. of samples to return

        :return:
            X_f np.array:
                2D array of density sampled locations
            f np.array:
                Array of density sampled values
        """
        Xo, Yo = self.Xo, self.Yo
        img_coords = self.img_coords
        rand_f = np.random.default_rng(0).choice(img_coords.shape[0], size=npts, replace=False)

        X_f = img_coords[rand_f, :]
        f = self.density.ravel()[rand_f]

        self.X_f = X_f
        self.f = f

        return X_f, f

    def infer(self, model=None, gp=None, method="MAP"):
        """
        Perform model inference

        :param model pymc3.Model:
            pymc3 model definition
        :param gp pymc3.gp.*:
            pymc3 gp definition
        :param method string:
            options: ["MAP", "ADVI", "MCMC"]

        :return:
            mu_s np.array:
                posterior mean
            cov_s np.array:
                posterior variance
        """
        with model:
            if method.upper() == "MAP":
                mp = pm.find_MAP()	#Standard optimisation method - returns a scalar
                print(mp)

                mu_s, cov_s, mu_dx, mu_dy = gp.predict( Xnew=self.img_coords, point=mp )
            else:
                if method.upper() == "ADVI":
                    advi_fit = pm.fit(method=pm.ADVI(), n=30000)
                    advi_elbo = pd.DataFrame(
                        {'log-ELBO': -np.log(advi_fit.hist),
                        'n': np.arange(advi_fit.hist.shape[0])
                        }
                    )
                    import seaborn as sns
                    _ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
                    trace = advi_fit.sample(1000)

                    self.trace = trace
                    pm.save_trace(trace, 'pickle/Case3_ADVI_trace', overwrite=True)
                    with open('pickle/Case3_ADVI_model', 'wb') as buff:
                        pickle.dump({'model': model}, buff)
                elif method.upper() == "MCMC":
                    trace = pm.sample(
                        draws=1000,
                        tune=1000,
                        discard_tuned_samples=True, #Only discards for trace plot
                        progressbar=True,
                        # step=pm.NUTS(),
                        # init='adapt_diag',
                        # start=start,
                        random_seed=0,
                        # chains=1,
                        # cores=1,
                    )

                    self.trace = trace
                    pm.save_trace(trace, 'pickle/Case3_MCMC_trace', overwrite=True)
                    with open('pickle/Case3_MCMC_model', 'wb') as buff:
                        pickle.dump({'model': model}, buff)
                else:
                    return None, None

                # Plot and print trace summary
                az.plot_trace(trace)
                az.plot_posterior(trace)
                summary = pm.summary(trace).round(2)
                print(summary)

                import corner  # https://corner.readthedocs.io
                names = [ x for x in trace.varnames if "__" not in x ]
                _ = corner.corner(
                    trace,
                    var_names=names,
                )

                mu_s, cov_s, mu_dx, mu_dy = gp.predict_trace( Xnew=self.img_coords, trace=trace, retcov=True )

            mu_s += np.average(self.f)
            mu_dx += np.average(self.y[:,0])
            mu_dy += np.average(self.y[:,1])
            
            self.mu_s, self.cov_s = mu_s.ravel(), cov_s
            self.mu_dx, self.mu_dy = mu_dx, mu_dy

        return mu_s.ravel(), cov_s

    def plot_true_grads(self):
        """
        Plot true gradients in x and y directions
        """
        norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

        f = plt.figure(figsize=(12,6))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        ax1.set_title('Schlieren dx')
        ax2.set_title('Schlieren dy')
        a = ax1.imshow(self.dx, cmap=self.gcmap, interpolation='bicubic', origin="lower")
        b = ax2.imshow(self.dy, cmap=self.gcmap, interpolation='bicubic', origin="lower")
        plt.colorbar(a, ax=ax1, shrink=0.75)
        plt.colorbar(b, ax=ax2, shrink=0.75)
        plt.show()

    def plot_sampling_pts(self):
        """
        Plot sampled points using either gen_all_pts, gen_subsample_pts, or gen_ran_pts
        """
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

        # b = ax2.contourf(Xo, Yo, us.reshape(N, N), 120, cmap="gray", vmin=np.min(us), vmax=np.max(us))
        ax1.scatter(self.X[:,0], self.X[:,1], s=40, edgecolor='k', facecolors='none')
        ax1.scatter(self.X_f[:,0], self.X_f[:,1], s=40, edgecolor='b', facecolors='blue')

        plt.show()

    def plot_posterior(self, savefile=None):
        """
        Plot comparison of true vs predicted density & gradients

        :param savefile string:
            String of filepath to export figure
        """
        Xo, Yo = self.Xo, self.Yo
        mu_s, cov_s = self.mu_s, self.cov_s
        density = self.density
        dx, dy = self.sch_dx, self.sch_dy
        N = self.N

        cmap = self.cmap
        gcmap = self.gcmap
        interp = self.interp

        mu_dx, mu_dy = self.mu_dx.reshape(N, N), self.mu_dy.reshape(N, N)

        # Calculate absolute error in posterior mean
        err = density - mu_s.reshape(N, N)

        # Calculate standard deviation from covariance matrix
        std_dev = np.sqrt(np.diag(cov_s)).reshape(N, N)

        matplotlib.rcParams.update({'font.size': 30})
        fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw = {'wspace':0.2, 'hspace':0.05})
        axes = [ax1, ax1, ax2, ax4, ax5]

        value_a = density
        a = ax1.imshow(value_a, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_a.min(), vmax=value_a.max())

        value_b = dx
        b = ax2.imshow(value_b, interpolation=interp, cmap=gcmap,
                    origin='lower', extent=[Xo.min(), Xo.max(), Yo.min(), Yo.max()],
                    vmin=value_b.min(), vmax=value_b.max())
        # ax2.scatter(self.X[:,0], self.X[:,1], s=40, edgecolor='w', facecolors='None')
        # ax2.scatter(self.X_f[:,0], self.X_f[:,1], s=80, edgecolor='w', facecolors='green')

        value_d = mu_s.reshape(N, N)
        d = ax4.imshow(value_d, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_a.min(), vmax=value_a.max())
                    # vmin=value_d.min(), vmax=value_d.max())

        value_e = mu_dx
        e = ax5.imshow(value_e, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_b.min(), vmax=value_b.max())
                    # vmin=value_e.min(), vmax=value_e.max())

        # Hide ticks
        for ax in axes:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        shrink = 0.7
        fig.colorbar(a, ax=ax1, orientation='vertical', shrink=shrink)
        fig.colorbar(b, ax=ax2, orientation='vertical', shrink=shrink)
        fig.colorbar(d, ax=ax4, orientation='vertical', shrink=shrink)
        fig.colorbar(e, ax=ax5, orientation='vertical', shrink=shrink)

        ax1.set_title(r'True density ($\rho$)', fontdict={'fontsize': 26})
        ax2.set_title(r'True vertical-knife Schlieren $(\partial \tilde{\rho} / \partial x)$', fontdict={'fontsize': 26})
        ax4.set_title(r'Test density ($\rho$)', fontdict={'fontsize': 26})
        ax5.set_title(r'Test vertical-knife Schlieren $(\partial \tilde{\rho} / \partial x)$', fontdict={'fontsize': 26})

        ax1.text(0.5, -0.1, 'a)', transform=ax1.transAxes)
        ax2.text(0.5, -0.1, 'b)', transform=ax2.transAxes)
        ax4.text(0.5, -0.1, 'c)', transform=ax4.transAxes)
        ax5.text(0.5, -0.1, 'd)', transform=ax5.transAxes)

        if savefile != None:
            # fig.savefig(savefile, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

            from os.path import join

            filenames = [
                'delete.png',
                'a_true_density.png',
                'b_true_vertical_knife_schlieren.png',
                'c_test_density.png',
                'd_test_vertical_knife_schlieren.png',
            ]

            for i, ax in enumerate(axes):
                # Save just the portion _inside_ each of the axis's boundaries
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # Take 30% more than width to include colorbar
                extent.x1 = extent.x0 + 1.3*(extent.x1 - extent.x0)
                fig.savefig(join(savefile, filenames[i]), bbox_inches=extent, pad_inches=0)

        plt.show()

    def plot_posterior_error(self, savefile=None):
        """
        Plot density error, and posterior standard deviation

        :param savefile string:
            String of filepath to export figure
        """
        Xo, Yo = self.Xo, self.Yo
        mu_s, cov_s = self.mu_s, self.cov_s
        density = self.density
        dx, dy = self.dx, self.dy
        N = self.N

        cmap = self.cmap
        gcmap = self.gcmap
        interp = self.interp

        mu_dx, mu_dy = self.mu_dx.reshape(N, N), self.mu_dy.reshape(N, N)

        # Calculate absolute error in posterior mean
        err = density - mu_s.reshape(N, N)

        # Calculate standard deviation from covariance matrix
        std_dev = np.sqrt(np.diag(cov_s)).reshape(N, N)

        matplotlib.rcParams.update({'font.size': 30})
        fig, ((ax7, ax8, ax9)) = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw = {'wspace':0.2, 'hspace':0.05})

        norm_g = matplotlib.colors.Normalize(vmin=np.min(err),\
                                        vmax=np.max(err))
        g = ax7.contourf(Xo, Yo, err, 120, cmap=cmap, norm=norm_g)


        n_bins=20
        err_rav = err.ravel()
        H, bins, patches = ax8.hist(err_rav, bins=n_bins, weights=np.ones(len(err_rav)) / len(err_rav))
        # # N is the count in each bin, bins is the lower-limit of the bin
        # # We can also normalize our inputs by the total number of counts using density=True
        # H, bins, patches = ax8.hist(err.ravel(), bins=n_bins, density=True)
        # We'll color code by height, but you could use any scalar
        fracs = H / H.max()
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())
        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
        # Now we format the y-axis to display percentage
        ax8.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax8.set_xlabel('GP Posterior error')


        norm_i = matplotlib.colors.Normalize(vmin=np.min(std_dev),\
                                        vmax=np.max(std_dev))
        i = ax9.contourf(Xo, Yo, std_dev, 120, cmap=cmap, norm=norm_i)


        # Hide ticks
        for ax in [ax7, ax9]:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig.colorbar(g, ax=ax7, orientation='vertical', shrink=0.75, format='%.3f')
        try:
            fig.colorbar(i, ax=ax9, orientation='vertical', shrink=0.75, format='%.3f')
        except:
            pass

        ax7.set_title(r'GP Posterior Error', fontdict={'fontsize': 26})
        # ax8.set_title(r'GP Posterior Error Histogram', fontdict={'fontsize': 26})
        ax9.set_title(r'GP Posterior $(\sigma)$', fontdict={'fontsize': 26})

        ax7.text(0.5, -0.2, 'a)', transform=ax7.transAxes)
        ax8.text(0.5, -0.2, 'b)', transform=ax8.transAxes)
        ax9.text(0.5, -0.2, 'c)', transform=ax9.transAxes)

        if savefile != None:
            # fig.savefig(savefile, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

            from os.path import join

            filenames = [
                'delete.png',
                'a_gp_posterior_error.png',
                'b_gp_posterior_error_histogram.png',
                'c_gp_posterior_std_dev.png',
            ]

            # Save just the portion _inside_ each of the axis's boundaries
            extent = ax7.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Take 30% more than width to include colorbar
            extent.x1 = extent.x0 + 1.32*(extent.x1 - extent.x0)
            fig.savefig(join(savefile, filenames[1]), bbox_inches=extent, pad_inches=0)

            # Save just the portion _inside_ each of the axis's boundaries
            extent = ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent.x0 = extent.x1 - 1.15*(extent.x1 - extent.x0)
            extent.x1 = extent.x0 + 1.025*(extent.x1 - extent.x0)
            extent.y0 = extent.y1 - 1.125*(extent.y1 - extent.y0)
            extent.y1 = extent.y0 + 1.025*(extent.y1 - extent.y0)
            fig.savefig(join(savefile, filenames[2]), bbox_inches=extent, pad_inches=0)

            # Save just the portion _inside_ each of the axis's boundaries
            extent = ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Take 30% more than width to include colorbar
            extent.x1 = extent.x0 + 1.32*(extent.x1 - extent.x0)
            fig.savefig(join(savefile, filenames[3]), bbox_inches=extent, pad_inches=0)

        plt.show()

    def plot_posterior_comparison(self):
        """
        Plot comparison between true density, its gradients and predicted values.
        """
        mu_s = self.mu_s
        density = self.density
        dx, dy = self.dx, self.dy
        mu_dx, mu_dy = self.mu_dx, self.mu_dy
        N = self.N

        matplotlib.rcParams.update({'font.size': 30})
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        a = ax1.scatter(mu_dx, dx, s=20, edgecolor='w', facecolors='green')
        b = ax2.scatter(mu_dy, dy, s=20, edgecolor='w', facecolors='green')
        c = ax3.scatter(mu_s.reshape(N, N), density, s=20, edgecolor='w', facecolors='green')

        # Set axis labels
        for ax in [ax1, ax2, ax3]:
            ax.set(xlabel='prediction', ylabel='truth')

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax1.set_title(r'grad_x truth vs prediction $(\partial \tilde{\rho} / \partial x)$', fontdict={'fontsize': 26})
        ax2.set_title(r'grad_y truth vs prediction $(\partial \tilde{\rho} / \partial y)$', fontdict={'fontsize': 26})
        ax3.set_title(r'density truth vs prediction ($\tilde{\rho}$)', fontdict={'fontsize': 26})

        ax1.text(0.5, -0.175, 'a)', transform=ax1.transAxes)
        ax2.text(0.5, -0.175, 'b)', transform=ax2.transAxes)
        ax3.text(0.5, -0.175, 'c)', transform=ax3.transAxes)

        plt.show()