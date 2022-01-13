import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import squeeze
import pymc3 as pm
import seaborn as sns
import arviz as az
import pandas as pd
import cv2
import pickle
sns.set_theme(style="ticks", font_scale=2.0)
from matplotlib import cm

def point_in_hull(point, hull, tolerance=0):
    """
    Check if point is within convex hull

    https://stackoverflow.com/questions/51771248/checking-if-a-point-is-in-convexhull
    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] < tolerance)
        for eq in hull.equations)


class image():
    def __init__(self, randseed=0, prescale=1.0, scale=1.0):
        self.cmap = plt.cm.inferno  #Colour colour-map
        self.gcmap = plt.cm.Greys_r   #Black & White colour-map
        self.interp = "spline36"    #Interpolation method for imshow
        if (randseed != 0) or (randseed != False):
            np.random.seed(randseed)
            random.seed(randseed)
        else:
            randseed = None

        self.randseed = randseed

        # Pre-scale density values
        self.prescale = prescale
        self.scale = scale

    def __normalise(self, a, dmin=0.0, dmax=1.0):
        '''
        Normalise numpy array to between user-specified inputs

        Parameters:
            a (np.array): Data vector
            dmin (float): Minimum for normalisation
            dmax (float): Maximum for normalisation

        Returns:
            data (np.array): Normalised input data to between dmin and dmax
        '''
        data = np.interp(a, (a.min(), a.max()), (dmin, dmax))
        return data

    def read_images(self, N, denVal=1.0):
        '''
        Reads in density and gradient input images

        Parameters:
            N (int): Grid dimensions for image grid and also used for testing grid (Xnew)

        Returns:
            None
        '''
        # Read in dx image
        img_dx = cv2.imread('inputs/case3/vertical-dx.png', 0)
        img_dx = cv2.flip(img_dx, 0)

        ## Read in dy image
        img_dy = cv2.imread('inputs/case3/horizontal-dy.png', 0)
        img_dy = cv2.flip(img_dy, 0)

        # Read in geometry mask image
        mask_geom = cv2.imread('inputs/case3/geometry-mask.png', cv2.IMREAD_UNCHANGED)
        mask_geom = cv2.flip(mask_geom, 0)

        # Read in density mask image
        mask_density = cv2.imread('inputs/case3/density-mask.png', cv2.IMREAD_UNCHANGED)
        mask_density = cv2.flip(mask_density, 0)

        # Fix farfield density
        density = mask_density[:,:,3].copy().astype(np.float64)
        density[density > 0.5] = denVal

        rho_indices = np.indices(mask_density.shape[:2])
        X_rho = np.c_[rho_indices[1].ravel(), rho_indices[0].ravel()]
        self.rho_indices = rho_indices
        self.X_rho = X_rho

        self.img_dx = img_dx
        self.img_dy = img_dy
        self.mask_geom = mask_geom
        self.mask_density = mask_density
        self.density = density
        self.density_scaled = density / self.prescale
        self.N = N

    def mask_geometry(self):
        '''
        Create a mask of geometry (where pixels are extremely dark)

        Parameters:
            None

        Returns:
            None
        '''
        img_dx, img_dy = self.img_dx, self.img_dy
        mask_geom, mask_density = self.mask_geom, self.mask_density
        N = self.N

        # Threshold image to get masking of solid object
        ret, thresh = cv2.threshold(mask_geom[:, :, 3], 6, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Re-scale geometry mask to same as downscaled schlieren images
        geom_contours = []
        for c in contours:
            geom_contours.append(c.astype(np.float32)* N/len(mask_geom))

        self.geom_thresh = thresh

        fig = plt.figure(figsize=(6, 5))
        c = plt.imshow(thresh, cmap="gray", origin='lower')
        plt.colorbar(c, shrink=0.5)
        plt.show()

        # Threshold image to get masking of density region
        ret, thresh = cv2.threshold(mask_density[:, :, 3], 127, 255, cv2.THRESH_BINARY)
        density_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.density_thresh = thresh
        cv2.drawContours(thresh, density_contours, -1, (128,128,128), 5)

        fig = plt.figure(figsize=(6, 5))
        c = plt.imshow(thresh, cmap="gray", origin='lower')
        plt.colorbar(c, shrink=0.5)
        plt.show()

        # Convert image coordinates to 2D array
        indices = np.indices(img_dx.shape[:2])
        img_coords = np.c_[indices[0].ravel(), indices[1].ravel()]
        sch_dx = img_dx
        sch_dy = img_dy

        self.sch_dx = sch_dx
        self.sch_dy = sch_dy
        self.geom_contours = geom_contours
        self.density_contours = density_contours
        self.contours = contours

    def gen_subsample_pts(self, step=2, mask_geom=True):
        '''
        Extracts subsample from a uniform image grid

        Parameters:
            step (int): Extract every `step` along x and y of uniform grid
            mask_geom (bool): Whether to skip geometry masked region

        Returns:
            X (np.array): Coordinate array
            y (np.array): 2D Gradient data array
        '''
        self.step = step
        ss = step
        vals_x = self.sch_dx
        vals_y = self.sch_dy
        N = self.N

        # Convert image coordinates to 2D array
        indices = np.indices(self.sch_dx.shape[:2])

        Xo, Yo = indices[1][::ss, ::ss], indices[0][::ss, ::ss]
        pts = np.c_[Xo.ravel(), Yo.ravel()]

        valsx_ss = vals_x[::ss, ::ss].reshape(-1, 1)
        valsy_ss = vals_y[::ss, ::ss].reshape(-1, 1)
        print("vals shape:", valsx_ss.shape)

        if mask_geom:
            X = []
            y = []
            for point, valx, valy in zip(pts, valsx_ss, valsy_ss):
                tp = tuple(point.ravel())
                for c in self.geom_contours:
                    hullCheck = cv2.pointPolygonTest(c, tp, True)
                if (hullCheck < 0) or (hullCheck == False):
                    X.append( point )
                    y.append( [valx, valy] )

            X, y = np.array(X, dtype=int), np.array(y)
        else:
            X = pts
            y = np.c_[valsx_ss, valsy_ss]

        y = y.reshape(-1, 2)
        print("Adding density points...")

        # Add density points to subset sampling if not already in it
        for X_f, f in zip(self.X_f, self.f):
            pt = X_f.reshape(1, 2).astype(int)
            val = np.c_[ vals_x[pt[0,1], pt[0,0]], vals_y[pt[0,1], pt[0,0]] ]
            if pt.tolist() not in X.tolist():
                X = np.r_[ X, X_f.reshape(1, 2) ]
                y = np.r_[ y, val ]

        self.H = X.shape[0]
        print("No. of training points: ", self.H)

        self.X = X
        self.y = y

        # Set scale coordinates
        # Normalise coordinates to between 0 and 10 instead of 0 to x/y length.
        self.scale = max(X[:,0].max(), X[:,1].max())/10

        return X, y

    def gen_test_grid(self, step=5):
        """
        Generate test grid

        Parameters:
            step (int): Extract every `step` along x and y of uniform grid
        """
        ss = step
        # Generate new points for testing
        # Convert image coordinates to 2D array
        indices = np.indices(self.sch_dx.shape[:2])

        Xo, Yo = indices[1][::ss, ::ss], indices[0][::ss, ::ss]
        pts = np.c_[Xo.ravel(), Yo.ravel()]
        self.den_ss = self.density[::ss, ::ss]
        self.sch_dx_ss = self.sch_dx[::ss, ::ss]
        self.sch_dy_ss = self.sch_dy[::ss, ::ss]
        self.X_rho = pts/self.scale
        print('Indices:', indices[1].shape, indices[0].shape)
        self.rows_new, self.cols_new = Xo.shape[0], Yo.shape[1]


    def get_nearest(self, points):
        """
        Parameters:
            points (np.array): Points to find nearest neighbour

        Returns:
            nearest_points (np.array): Points from input image nearest to input points
        """
        from pykdtree.kdtree import KDTree
        X = self.X_rho
        kd_tree = KDTree(np.array(X))
        dist, idx = kd_tree.query(points, k=1)
        nearest_points = X[idx, :]

        # When sampling dataset, need to pass in y index first, then x index
        f = np.c_[ self.density_scaled[nearest_points[:,1], nearest_points[:,0]] ]

        return nearest_points, f

    def plot_true_grads(self):
        '''
        Plot assumed true gradients

        Parameters:
            None

        Returns:
            None
        '''
        f = plt.figure(figsize=(36,10))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)

        ax1.set_title('True density')
        ax2.set_title('Schlieren dx')
        ax3.set_title('Schlieren dy')
        a = ax1.imshow(self.density, cmap=self.cmap, interpolation=self.interp, origin='lower')
        b = ax2.imshow(self.sch_dx, cmap=self.cmap, interpolation=self.interp, origin='lower')
        c = ax3.imshow(self.sch_dy, cmap=self.cmap, interpolation=self.interp, origin='lower')
        plt.colorbar(a, ax=ax1, shrink=0.5)
        plt.colorbar(b, ax=ax2, shrink=0.5)
        plt.colorbar(c, ax=ax3, shrink=0.5)
        plt.show()

    def get_true_grads(self):
        """
        Return normalised/true gradient values

        Returns:
            self.dx np.array: Array of d_\tilde{rho}/dx values evaluated across uniform grid
            self.dy np.array: Array of d_\tilde{rho}/dy values evaluated across uniform grid
        """
        return self.dx, self.dy


    def get_ran_density(self, npts=3):
        """
        Return randomly sampled density from input image

        Parameters:
            npts (int): Number of density points to sample

        Returns:
            X_f (np.array): Density coordinate array
            f (np.array): Density data array
        """
        if self.randseed != None:
            random.seed(self.randseed)
        density = self.density_scaled
        contours = self.contours
        X_f = np.zeros((npts, 2), dtype=int)
        f = np.zeros((npts, 2))
        point = np.zeros((1, 2), dtype=int)
        count = 0
        while count < npts:
            point[0, 0] = random.randint(0,density.shape[0]-1)
            point[0, 1] = random.randint(0,density.shape[1]-1)

            tp = tuple(point.ravel())
            for c in contours:
                hullCheck = cv2.pointPolygonTest(c, tp, True)
            if (hullCheck > 0):
                X_f[count, :] = point
                count += 1

        f = density[X_f[:,1], X_f[:,0]]

        self.X_f = X_f
        self.f = f

        return X_f, f

    def infer(self, model=None, gp=None, method="MAP"):
        '''
        Calculate posterior mean and covariance

        Parameters:
            model (pm.Model()): pymc3 model definition
            gp (pm.gp): pymc3 gp definition
            method (string): Hyperparameter optimisation method 
                            {"MAP", "ADVI", "MCMC}

        Returns:
            mu_s (np.array): Posterior mean
            cov_s (np.array): Posterior covariance
        '''
        X_rho = self.X_rho
        with model:
            if method.upper() == "MAP":
                mp = pm.find_MAP()	#Standard optimisation method - returns a scalar
                print(mp)
                self.mp = mp

                mu_s, cov_s, mu_dx, mu_dy = gp.predict( Xnew=X_rho, point=mp )
            else:
                if method.upper() == "ADVI":
                    advi_fit = pm.fit(method=pm.ADVI(), n=20000)
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
                        tune=1500,
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

                mu_s, cov_s, mu_dx, mu_dy = gp.predict_trace( Xnew=X_rho, trace=trace, retcov=True )

            mu_s += np.average(self.f)
            mu_dx += np.average(self.y[:,0])
            mu_dy += np.average(self.y[:,1])

            mu_s *= self.prescale
            cov_s *= self.prescale

            cov_s[cov_s < 1E-5] = 0.0

            # Calculate standard deviation from covariance matrix
            rows_new, cols_new = self.rows_new, self.cols_new
            std_dev = np.sqrt(np.diag(cov_s)).reshape(rows_new, cols_new)

            self.mu_s, self.cov_s = mu_s.ravel(), cov_s
            self.mu_dx, self.mu_dy = mu_dx, mu_dy
            self.std_dev = std_dev

        return mu_s.ravel(), cov_s

    def plot_sampling_pts(self, savefile=None):
        '''
        Plot sampled points

        Parameters:
            None

        Returns:
            None
        '''
        fig, (ax1) = plt.subplots(1, 1)

        dx, dy = self.sch_dx, self.sch_dy
        value = dx
        a = ax1.imshow(value, interpolation=self.interp, cmap=self.gcmap,
                    origin='lower',
                    vmin=value.min(), vmax=value.max())

        X_f = self.X_f
        # b = ax2.contourf(Xo, Yo, us.reshape(N, N), 120, cmap="gray", vmin=np.min(us), vmax=np.max(us))
        # ax1.scatter(self.X[:,0], self.X[:,1], s=20, edgecolor='g', facecolors='green')
        ax1.scatter(X_f[:,0], X_f[:,1], s=40, edgecolor='b', facecolors='blue')

        # Hide ticks
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        # ax1.set_facecolor((0, 0, 0))

        plt.tight_layout()
        if savefile != None:
            fig.savefig(savefile, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        plt.show()

    def plot_posterior(self, savefile=None):
        """
        Plot set of posterior results along with truth and errors 

        Parameters:
            None

        Returns:
            None
        """
        mu_s, cov_s = self.mu_s, self.cov_s
        density = self.density
        dx, dy = self.sch_dx, self.sch_dy
        X_rho = self.X_rho
        N = self.N
        rho_indices = self.rho_indices
        std_dev = self.std_dev

        cmap = self.cmap
        gcmap = self.gcmap
        interp = self.interp

        rows_old, cols_old = self.img_dx.shape[1], self.img_dx.shape[0]
        rows_new, cols_new = self.rows_new, self.cols_new
        mu_dx, mu_dy = self.mu_dx.reshape(rows_new, cols_new), self.mu_dy.reshape(rows_new, cols_new)

        # Geometry mask
        mask = np.zeros_like(self.img_dx)
        cv2.drawContours(mask, self.contours, -1, color=(1,0,0), thickness=cv2.FILLED)
        mask[mask > 0] = 1
        mask = mask.astype(bool)

        # Calculate absolute error in posterior mean
        err = self.den_ss.reshape(rows_new, cols_new) - mu_s.reshape(rows_new, cols_new)        

        matplotlib.rcParams.update({'font.size': 30})
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(30, 30), gridspec_kw = {'wspace':0.05, 'hspace':0.0}, squeeze=True)
        axes = [ax2, ax2, ax3, ax4, ax5, ax6, ax8]

        value_b = dx
        value_b = np.ma.array(value_b, mask=~mask)
        b = ax2.imshow(value_b, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_b.min(), vmax=value_b.max())
        # ax2.scatter(self.X[:,0], self.X[:,1], s=40, edgecolor='w', facecolors='None')
        # ax2.scatter(self.X_f[:,0], self.X_f[:,1], s=80, edgecolor='w', facecolors='green')

        value_c = dy
        value_c = np.ma.array(value_c, mask=~mask)
        c = ax3.imshow(value_c, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_c.min(), vmax=value_c.max())

        value_d = mu_s.reshape(rows_new, cols_new)
        value_d = cv2.resize(value_d, (rows_old, cols_old))
        value_d = np.ma.array(value_d, mask=~mask)
        d = ax4.imshow(value_d, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    # vmin=value_a.min(), vmax=value_a.max())
                    vmin=value_d.min(), vmax=value_d.max())

        value_e = mu_dx
        value_e = cv2.resize(value_e, (rows_old, cols_old))
        value_e = np.ma.array(value_e, mask=~mask)
        e = ax5.imshow(value_e, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_b.min(), vmax=value_b.max())
                    # vmin=value_e.min(), vmax=value_e.max())

        value_f = mu_dy
        value_f = cv2.resize(value_f, (rows_old, cols_old))
        value_f = np.ma.array(value_f, mask=~mask)
        F = ax6.imshow(value_f, interpolation=interp, cmap=gcmap,
                    origin='lower',
                    vmin=value_c.min(), vmax=value_c.max())
                    # vmin=value_f.min(), vmax=value_f.max())

        value_g = std_dev
        value_g = cv2.resize(value_g, (rows_old, cols_old))
        value_g = np.ma.array(value_g, mask=~mask)
        g = ax8.imshow(value_g, interpolation=interp, cmap=cmap,
                    origin='lower',
                    vmin=value_g.min(), vmax=value_g.max())

        ax1.axis('off')
        ax7.axis('off')
        ax9.axis('off')

        # Hide ticks
        for ax in [ax2, ax3, ax4, ax5, ax6, ax8]:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_facecolor((0, 0, 0))

        shrink = 0.75
        fig.colorbar(b, ax=ax2, orientation='vertical', shrink=shrink)
        fig.colorbar(c, ax=ax3, orientation='vertical', shrink=shrink)
        fig.colorbar(d, ax=ax4, orientation='vertical', shrink=shrink)
        fig.colorbar(e, ax=ax5, orientation='vertical', shrink=shrink)
        fig.colorbar(F, ax=ax6, orientation='vertical', shrink=shrink)
        fig.colorbar(g, ax=ax8, orientation='vertical', shrink=shrink)

        ax2.set_title(r'True vertical-knife Schlieren $(\partial \tilde{\rho} / \partial x)$', fontdict={'fontsize': 26})
        ax3.set_title(r'True horizontal-knife Schlieren $(\partial \tilde{\rho} / \partial y)$', fontdict={'fontsize': 26})
        ax4.set_title(r'Test density ($\rho$)', fontdict={'fontsize': 26})
        ax5.set_title(r'Test vertical-knife Schlieren $(\partial \tilde{\rho} / \partial x)$', fontdict={'fontsize': 26})
        ax6.set_title(r'Test horizontal-knife Schlieren $(\partial \tilde{\rho} / \partial y)$', fontdict={'fontsize': 26})
        ax8.set_title(r'GP Posterior $(\sigma)$', fontdict={'fontsize': 26})

        off = -0.1
        ax2.text(0.5, off, 'a)', transform=ax2.transAxes)
        ax3.text(0.5, off, 'b)', transform=ax3.transAxes)
        ax4.text(0.5, off, 'c)', transform=ax4.transAxes)
        ax5.text(0.5, off, 'd)', transform=ax5.transAxes)
        ax6.text(0.5, off, 'e)', transform=ax6.transAxes)
        ax8.text(0.5, off, 'f)', transform=ax8.transAxes)

        if savefile != None:
            # fig.savefig(savefile, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

            from os.path import join

            filenames = [
                'delete.png',
                'a_true_vertical_knife_schlieren.png',
                'b_true_horizontal_knife_schlieren.png',
                'c_test_density.png',
                'd_test_vertical_knife_schlieren.png',
                'e_test_horizontal_knife_schlieren.png',
                'f_gp_posterior_std_dev.png',
            ]

            for i, ax in enumerate(axes):
                # Save just the portion _inside_ each of the axis's boundaries
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # Take 25% more than width to include colorbar
                extent.x1 = extent.x0 + 1.25*(extent.x1 - extent.x0)
                fig.savefig(join(savefile, filenames[i]), bbox_inches=extent, pad_inches=0)

        plt.show()

    def plot_posterior_comparison(self):
        '''
        Plot linear comparison between truth and posterior calculations

        Parameters:
            None

        Returns:
            None
        '''
        rows_new, cols_new = self.rows_new, self.cols_new
        mu_s = self.mu_s.reshape(rows_new, cols_new)
        density = self.den_ss
        dx, dy = self.sch_dx_ss, self.sch_dy_ss
        mu_dx, mu_dy = self.mu_dx, self.mu_dy
        N = self.N

        matplotlib.rcParams.update({'font.size': 30})
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        a = ax1.scatter(mu_dx, dx, s=20, edgecolor='w', facecolors='green')
        b = ax2.scatter(mu_dy, dy, s=20, edgecolor='w', facecolors='green')
        c = ax3.scatter(mu_s, density, s=20, edgecolor='w', facecolors='green')

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

    def resize(self, arr, dims=None):
        """
        Resize input image using openCV

        Parameters:
            arr (np.array): Input image array

        Returns:
            output (np.array): Resized image array
        """
        rows_old, cols_old = self.img_dx.shape[1], self.img_dx.shape[0]
        if dims == None:
            dims = (rows_old, cols_old)
        return cv2.resize(arr, dims)

    def save_results(self, filepath=None):
        """
        Save generated results to vtk file

        Parameters:
            filepath (string): Output file path
        """
        density = self.den_ss
        sch_dx, sch_dy = self.sch_dx_ss, self.sch_dy_ss
        rho, dx, dy = self.mu_s, self.mu_dx, self.mu_dy
        std_dev = self.std_dev
        ss = self.step

        import pyvista as pv

        # Get matplotlib grey-scale colourmap for use later on
        grey_cmap = plt.cm.get_cmap("Greys_r")

        # Create the spatial reference
        grid = pv.UniformGrid()

        # Set grid dimensions based on input image size
        dims = np.array( density.T.shape + (0,) )
        grid.spacing = (ss, ss, 1)  # These are the cell sizes along each axis

        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        grid.dimensions = dims + 1

        # Add the data values to the cell data
        grid.cell_data["truth_density"] = density.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["truth_sch_dx"] = sch_dx.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["truth_sch_dy"] = sch_dy.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["test_density"] = rho.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["test_dx"] = dx.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["test_dy"] = dy.T.flatten(order="F")  # Flatten the array!
        grid.cell_data["test_std_dev"] = std_dev.T.flatten(order="F")  # Flatten the array!

        # (grid is cell data, mesh is point data)
        mesh = grid.cell_data_to_point_data()

        # Save result to file for viewing in ParaView
        mesh.save(filepath);

        self.mesh_result = mesh
