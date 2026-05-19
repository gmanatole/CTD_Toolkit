import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import pandas as pd
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from ctd_toolkit.utils.sql_query import SQLQuery
from ctd_toolkit.utils.workers import _process_file_reader, _reconstruct_single_profile
from ctd_toolkit.base_loader import BaseLoader

class FunctionalPrincipalComponent(BaseLoader):

    K = 50
    lam = 0.05

    def __init__(self,
                 grid,
                 path : str,
                 source: SQLQuery,
                 var : list[str],
                 depth_min : int = None,
                 depth_max : int = None,
                 depth_reconstruction : int = None
                 ):

        self.grid = grid
        self.source = source
        self.path = path if (path.startswith("'") and path.endswith("'")) else f"'{path}'"

        self.var = var
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_reconstruction = depth_reconstruction
        # if self.depth_max & self.depth_reconstruction :
        #     assert self.depth_reconstruction < self.depth_max, "Please choose a reconstruction depth lower than the chosen fPCA's maximum depth"

    @property
    def depth_max(self):
        return self.__depth_max

    @depth_max.setter
    def depth_max(self, value):
        self.__depth_max = value

    @property
    def depth_min(self):
        return self.__depth_min

    @depth_min.setter
    def depth_min(self, value):
        self.__depth_min = value

    def base_projection(self) :
        # Create bspline base
        self.basis = BSplineBasis(domain_range=(self.depth_min, self.depth_max), n_basis=self.K, order=4)
        # Lambda penalty on second derivative
        regularization = L2Regularization(
            LinearDifferentialOperator(2),
            regularization_parameter=self.lam)
        smoother = BasisSmoother(
            basis=self.basis,
            regularization=regularization,
            return_basis=True)
        # Project data to base, shape should be (Nobs x Npoints)
        smoothed_data = []
        for i in range(len(self.var)):
            fd = FDataGrid(data_matrix=self.data[i, :, :], grid_points= self.depth) #np.arange(self.depth_min, self.depth_max + 1))
            smoothed = smoother.fit_transform(fd)
            smoothed_data.append(smoothed)
        # Combine results - adjust based on how you want to store multivariate results
        self.ydf = smoothed_data

    def fpca_univariate(self):
        self.ydf = self.ydf[0]
        self.alpha = self.ydf.coefficients
        alpha_mean = self.alpha.mean(axis=0)
        alpha_std = self.alpha.std(axis=0)
        n = self.alpha.shape[0]
        # Compute covariance matrix
        C = self.alpha - alpha_mean
        C_norm = (self.alpha - alpha_mean) / alpha_std
        self.mat_cov = (C.T @ C) / n
        self.mat_cor = (C_norm.T @ C_norm) / n
        W = self.basis.gram_matrix()
        # Ensure symmetry
        self.W = (W + W.T) / 2
        Wdem = cholesky(W)
        Wdeminv = solve_triangular(Wdem, np.eye(Wdem.shape[0]))
        eigenvalues, eigenvectors = np.linalg.eig(Wdem @ self.mat_cov @ Wdem.T)

        # Handle complex numbers if needed
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        # Handle negative eigenvalues
        eigenvalues = np.abs(eigenvalues)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        vectnotWM = eigenvectors.copy()  # b_k (not W-normalized)
        vectors = Wdeminv @ vectnotWM

        axe = vectors * np.sqrt(eigenvalues)  # deformation axes
        pc = C @ self.W @ vectors  # principal components, shape: (Nobs, K)
        pval = np.round(eigenvalues / eigenvalues.sum() * 100, 3)

        self.mfpca = {
            'values': eigenvalues,
            'vectors': vectors,
            'vectnotWM': vectnotWM,
            'axe': axe,
            'pc': pc,
            'pval': pval}

    def fpca_multivariate(self):
        """
        Multivariate FPCA with variable-wise normalization (metric M)
        and shared B-spline basis across all variables.
        """
        nvar = len(self.var)  # number of functional variables

        # Merge coefficients: shape (N, K * nvar) each variable's coefficients are stored in self.ydf_list[i].coefficients (N, K)
        self.alpha = np.hstack([self.ydf[i].coefficients for i in range(nvar)])

        # Remove profiles with NaN
        na_mask = np.isnan(self.alpha.sum(axis=1))
        self.alpha = self.alpha[~na_mask]
        N = self.alpha.shape[0]
        alpha_mean = self.alpha.mean(axis=0)  # (K * nvar,)
        self.C = self.alpha - alpha_mean  # (N, K * nvar)

        W_i = self.basis.gram_matrix()  # (K, K)
        W_i = (W_i + W_i.T) / 2  # Ensure symmetry
        W = np.zeros((self.K * nvar, self.K * nvar))
        for i in range(nvar):
            i0, i1 = i * self.K, (i + 1) * self.K
            W[i0:i1, i0:i1] = W_i
        W = (W + W.T) / 2  # Ensure symmetry
        Wdem = cholesky(W)
        Wdeminv = solve_triangular(Wdem, np.eye(Wdem.shape[0]))

        sigma2 = np.zeros(nvar)
        for i in range(nvar):
            i0, i1 = i * self.K, (i + 1) * self.K
            V_ii = (self.C[:, i0:i1].T @ self.C[:, i0:i1] @ W_i) / N
            sigma2[i] = np.trace(V_ii)

        # Build diagonal M and its square root / inverse square root
        m_diag = np.repeat(1.0 / sigma2, self.K)  # (K * nvar,)
        M = np.diag(m_diag)
        Mdem = np.diag(np.sqrt(m_diag))
        Mdeminv = np.diag(1.0 / np.sqrt(m_diag))
        self.mat_cov = (self.C.T @ self.C) / N
        VWM = Mdem @ Wdem @ self.mat_cov @ Wdem.T @ Mdem
        VWM = (VWM + VWM.T) / 2
        eigenvalues, eigenvectors = np.linalg.eig(VWM)

        # Handle complex and negative eigenvalues
        eigenvalues = np.abs(np.real(eigenvalues))
        eigenvectors = np.real(eigenvectors)

        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Back-project eigenvectors into original (coefficient) space
        vectnotWM = eigenvectors.copy()  # b_k
        vectors = Mdeminv @ Wdeminv @ vectnotWM  # B_k (W,M-normalized)
        axe = vectors * np.sqrt(eigenvalues)  # deformation axes
        pc = self.C @ W @ M @ vectors  # principal components (N, K*nvar)
        pval = np.round(eigenvalues / eigenvalues.sum() * 100, 3)

        self.mfpca = {
            'values': eigenvalues,
            'vectors': vectors,
            'vectnotWM': vectnotWM,
            'axe': axe,
            'pc': pc,
            'pval': pval,
            'alpha_mean': alpha_mean,
            'W': W,
            'M': M,
            'Wdem': Wdem,
            'Wdeminv': Wdeminv,
            'W_i' : W_i,
            'sigma2': sigma2,
            'na_mask': na_mask,
        }

    def profile_reconstruction(self, K = None) :
        """
        Reconstruct profiles using fPCA
        """
        if not K :
            K = self.K
        grid_points = np.arange(self.depth_min, self.depth_max + 1)
        alpha_recon = self.alpha.mean(axis=0) + self.mfpca['pc'][:, :K] @ self.mfpca['vectors'][:, :K].T
        phi = self.basis(grid_points)
        x_recon = alpha_recon @ phi.squeeze()
        self.X = x_recon

    def bathymetry_mask(self, bathymetry : str, bathymetry_threshold : float = 2000) :
        """
        Join bathymetry from netCDF GEBCO file
        Replace all pixels with bathymetry lower than bathymetry_threshold with nan values
        """
        ds = nc.Dataset(bathymetry)
        grid_bathy = RegularGridInterpolator((ds['lat'][:].data, ds['lon'][:].data), ds['elevation'][:].data)(self.grid.latitude, self.grid.longitude)
        mask = grid_bathy <= bathymetry_threshold
        self.grid['PC'] = self.grid['PC'].where(mask)

    def truncated_reconstruction(self):
        """
        Reconstruct truncated profiles (up to depth_reconstruction) using fPCA to reach depth_max
        """
        alpha_chap = self.mfpca['alpha_mean']
        vectors = self.mfpca['vectors']
        values = self.mfpca['values']
        V = self.mat_cov
        _data_recon = np.full((2, len(self.timestamp_recon), len(self.depth)), np.nan)
        pc_recon = np.full((len(self.timestamp_recon), 2*self.K), np.nan)
        grid_points = np.arange(self.depth_min, self.depth_max + 1)
        X_0 = self.ydf[0].coefficients
        X_1 = self.ydf[1].coefficients
        phi = np.squeeze(self.basis(grid_points), axis=2)
        y_eval_0 = X_0 @ phi
        y_eval_1 = X_1 @ phi
        estimsig0 = 1 / self.data.shape[1] * np.sum((y_eval_0 - self.data[0]) ** 2, axis=0)
        estimsig1 = 1 / self.data.shape[1] * np.sum((y_eval_1 - self.data[1]) ** 2, axis=0)
        regularization = L2Regularization(LinearDifferentialOperator(2), regularization_parameter=1000)
        smoother = BasisSmoother(basis=self.basis, regularization=regularization, return_basis=True)
        sig0 = smoother.fit_transform(FDataGrid(data_matrix=estimsig0, grid_points=grid_points))
        sig1 = smoother.fit_transform(FDataGrid(data_matrix=estimsig1, grid_points=grid_points))
        estimsig0 = np.squeeze(sig0.coefficients @ phi)
        estimsig1 = np.squeeze(sig1.coefficients @ phi)
        for i in tqdm(range(10, len(self.timestamp_recon)), leave = True, position=0, total=len(self.timestamp_recon)):
            # create phi for varying depth
            phi_recon = np.squeeze(self.basis(self.depth_recon[i]))
            # covariance blocks
            G_00 = phi_recon.T @ V[:self.K, :self.K] @ phi_recon
            G_01 = phi_recon.T @ V[:self.K, self.K:] @ phi_recon
            G_10 = phi_recon.T @ V[self.K:, :self.K] @ phi_recon
            G_11 = phi_recon.T @ V[self.K:, self.K:] @ phi_recon
            mu_0 = phi_recon.T @ alpha_chap[:self.K]
            mu_1 = phi_recon.T @ alpha_chap[self.K:]
            mu = np.concatenate((mu_0, mu_1))
            condi = np.concatenate((estimsig0[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1],
                                    estimsig1[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1]))
            y0 = self.data_recon[self.var[0]][i]
            y1 = self.data_recon[self.var[1]][i]
            #compute sigma
            n = G_00.shape[0]
            sigmaY = np.empty((2 * n, 2 * n))
            sigmaY[:n, :n] = G_00
            sigmaY[:n, n:] = G_01
            sigmaY[n:, :n] = G_10
            sigmaY[n:, n:] = G_11
            sigmaY.flat[::2 * n + 1] += condi
            Y = np.concatenate((y0, y1))
            eps0 = vectors.T[:2 * self.K, :self.K] @ phi_recon
            eps1 = vectors.T[:2 * self.K, self.K:] @ phi_recon
            eps = np.hstack([eps0, eps1])
            # solve
            cf = cho_factor(sigmaY, lower=False)
            x = cho_solve(cf, Y - mu)
            pc = values[:2 * self.K] * (eps @ x)
            # reconstruction
            coefs0 = alpha_chap[:self.K] + vectors[:self.K] @ pc
            coefs1 = alpha_chap[self.K:] + vectors[self.K:] @ pc
            recon = np.vstack((coefs0 @ phi, coefs1 @ phi))
            _data_recon[:,i,:] =  recon
            pc_recon[i, :] = pc
        self.method = np.concatenate((['classic']*self.data.shape[1], ['reconstruction']*len(pc_recon)))
        self.mfpca['pc'] = np.concatenate([self.mfpca['pc'], pc_recon], axis=0)
        self.data = np.concatenate([self.data, _data_recon], axis=1)
        self.timestamp = np.concatenate([self.timestamp, self.timestamp_recon])
        self.gps = np.vstack([self.gps, self.gps_recon])
        self.source = np.concatenate([self.source, self.source_recon])
        self.fns = np.concatenate([self.fns, self.fns_recon])
        self.profile = np.concatenate([self.profile, self.profile_recon])

    def truncated_spatial_reconstruction(self, h = 5):
        """
        Reconstruct truncated profiles (up to depth_reconstruction) using fPCA to reach depth_max
        """
        Wdem = self.mfpca['Wdem']
        Wdeminv = self.mfpca['Wdeminv']
        W_i = self.mfpca['W_i']
        _data_recon = np.full((2, len(self.timestamp_recon), len(self.depth)), np.nan)
        pc_recon = np.full((len(self.timestamp_recon), 2*self.K), np.nan)
        grid_points = np.arange(self.depth_min, self.depth_max + 1)
        X_0 = self.ydf[0].coefficients
        X_1 = self.ydf[1].coefficients
        X = np.hstack((X_0, X_1))
        phi = np.squeeze(self.basis(grid_points), axis=2)
        y_eval_0 = X_0 @ phi
        y_eval_1 = X_1 @ phi
        estimsig0 = 1 / self.data.shape[1] * np.sum((y_eval_0 - self.data[0]) ** 2, axis=0)
        estimsig1 = 1 / self.data.shape[1] * np.sum((y_eval_1 - self.data[1]) ** 2, axis=0)
        regularization = L2Regularization(LinearDifferentialOperator(2), regularization_parameter=1000)
        smoother = BasisSmoother(basis=self.basis, regularization=regularization, return_basis=True)
        sig0 = smoother.fit_transform(FDataGrid(data_matrix=estimsig0, grid_points=grid_points))
        sig1 = smoother.fit_transform(FDataGrid(data_matrix=estimsig1, grid_points=grid_points))
        estimsig0 = np.squeeze(sig0.coefficients @ phi)
        estimsig1 = np.squeeze(sig1.coefficients @ phi)
        for i in tqdm(range(len(self.timestamp_recon)), leave = True, total=len(self.timestamp_recon)):
            # create phi for varying depth
            # grid_recon = np.arange(self.depth_min, np.nanmax(self.depth_recon[i]) + 1)
            phi_recon = np.squeeze(self.basis(self.depth_recon[i]))
            # Create reconstruction informed based on gps localization
            weights = multivariate_normal.pdf(self.gps, mean=self.gps_recon[i], cov=np.array([[h, 0], [0, h]]))
            norm_weights = weights / (np.sum(weights) + 1e-12)
            alpha_chap = np.sum(X * norm_weights[:, None], axis=0)
            C = X - alpha_chap
            V = (norm_weights[:, None] * C).T @ C
            sigma2_0 = np.sum(np.diag(V[:self.K, :self.K] @ W_i))
            sigma2_1 = np.sum(np.diag(V[self.K:, self.K:] @ W_i))
            # M = np.diag(np.concatenate([np.repeat(1 / sigma2_0, self.K), np.repeat(1 / sigma2_1, self.K)]))
            # Mdem = np.sqrt(M)  # M^(1/2)
            M = np.concatenate([
                np.full(self.K, 1 / sigma2_0),
                np.full(self.K, 1 / sigma2_1)
            ])
            sqrtM = np.sqrt(M)
            #Mdeminv = np.linalg.inv(Mdem)
            Mdem_diag = np.sqrt(M)
            Mdeminv_diag = 1.0 / Mdem_diag
            # VWM = Mdem @ self.mfpca['Wdem'] @ V @ self.mfpca['Wdem'].T @ Mdem
            A = Wdem @ V @ Wdem.T
            VWM = sqrtM[:, None] * A * sqrtM[None, :]
            values, vectors = np.linalg.eigh(VWM)
            idx = np.argsort(values)[::-1]
            values = values[idx]
            vectors = vectors[:, idx]
            # vectors = Mdeminv @ self.mfpca['Wdeminv'] @ vectors
            vectors = Mdeminv_diag[:, None] * (Wdeminv @ vectors)
            y0 = self.data_recon[self.var[0]][i]
            y1 = self.data_recon[self.var[1]][i]
            G_00 = phi_recon.T @ V[:self.K, :self.K] @ phi_recon  # inst.mat_cov[:inst.K, :inst.K] @ phi_recon
            G_01 = phi_recon.T @ V[:self.K, self.K:] @ phi_recon
            G_10 = phi_recon.T @ V[self.K:, :self.K] @ phi_recon
            G_11 = phi_recon.T @ V[self.K:, self.K:] @ phi_recon
            mu_0 = phi_recon.T @ alpha_chap[:self.K]  # @ phi_recon # mu_0 = C[:, :inst.K] @ phi_recon
            mu_1 = phi_recon.T @ alpha_chap[self.K:]  # @ phi_recon  # mu_1 = C[:, inst.K:] @ phi_recon
            mu = np.concatenate((mu_0, mu_1))  # mu = np.concatenate((mu_0, mu_1))
            # condi = np.diag(np.concatenate((estimsig0[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1],
            #                                 estimsig1[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1])))
            # sigmaY = np.block([[G_00, G_01], [G_10, G_11]]) + condi
            condi = np.concatenate((estimsig0[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1],
                                            estimsig1[:np.nanmax(self.depth_recon[i]) - self.depth_min + 1]))
            n = G_00.shape[0]
            sigmaY = np.empty((2 * n, 2 * n))
            sigmaY[:n, :n] = G_00
            sigmaY[:n, n:] = G_01
            sigmaY[n:, :n] = G_10
            sigmaY[n:, n:] = G_11
            sigmaY.flat[::2 * n + 1] += condi
            Y = np.concatenate((y0, y1))
            eps0 = vectors.T[:2 * self.K, :self.K] @ phi_recon
            eps1 = vectors.T[:2 * self.K, self.K:] @ phi_recon
            eps = np.hstack([eps0, eps1])
            # R = cholesky(sigmaY)
            cf = cho_factor(sigmaY, lower=False)
            x = cho_solve(cf, Y - mu) # np.linalg.solve(R, np.linalg.solve(R.T, Y - mu))
            pc = values[:2 * self.K] * (eps @ x)
            # coefs0 = alpha_chap[:self.K] + np.sum(vectors[:self.K] @ np.diag(pc), axis=1)
            coefs0 = alpha_chap[:self.K] + vectors[:self.K] @ pc
            # coefs1 = alpha_chap[self.K:] + np.sum(vectors[self.K:] @ np.diag(pc), axis=1)  # C[i, inst.K:]
            coefs1 = alpha_chap[self.K:] + vectors[self.K:] @ pc
            recon = np.vstack((coefs0 @ phi, coefs1 @ phi))
            _data_recon[:,i,:] =  recon
            pc_recon[i, :] = pc
        self.method = np.concatenate((['classic']*self.data.shape[1], ['reconstruction']*len(pc_recon)))
        self.mfpca['pc'] = np.concatenate([self.mfpca['pc'], pc_recon], axis=0)
        self.data = np.concatenate([self.data, _data_recon], axis=1)
        self.timestamp = np.concatenate([self.timestamp, self.timestamp_recon])
        self.gps = np.vstack([self.gps, self.gps_recon])
        self.source = np.concatenate([self.source, self.source_recon])
        self.fns = np.concatenate([self.fns, self.fns_recon])
        self.profile = np.concatenate([self.profile, self.profile_recon])

    def get_profiles(self, df, workers = 8) :
        # group by file
        file_groups = list(df.groupby("fn"))
        # prepare tasks for workers
        tasks = [
            (fn, group.reset_index(drop=True), self.var)
            for fn, group in file_groups
        ]
        depth, data, gps, timestamp, fns, profile, source = [], {key:[] for key in self.var}, [], [], [], [], []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = executor.map(_process_file_reader, tasks)
            for result in tqdm(futures, total=len(tasks), desc="Processing files"):
                for i in range(len(result[1])) :
                    data[self.var[i]].extend(result[1][i])
                depth.extend(result[0])
                gps.extend(result[2])
                timestamp.extend(result[3])
                fns.extend([result[4]]*len(result[2]))
                profile.extend([result[5]]*len(result[2]))
                source.extend([result[6]]*len(result[2]))
        self.data = data
        self.depth = depth
        self.gps = np.array(gps)
        self.profile = profile
        self.timestamp = timestamp
        self.fns = fns
        self.source = source

    def format_model(self) :
        out = np.empty((len(self.var), *self.data[self.var[0]].shape), dtype=np.float32)
        for j, i in enumerate(self.var):
            out[j] = self.data[i]
        self.data = out

    def visualize_projection(self, var = 'TEMP_ADJUSTED') :
        """
        Plot 10 random profiles and their reconstruction using fPCA
        """
        idx = np.random.choice(self.data.shape[1], 10, replace=False)
        xgrid = np.arange(self.depth_min, self.depth_max + 1)
        # Evaluate basis on grid - equivalent to eval.basis()
        ygrid = self.basis.to_basis()(xgrid)
        # shape: (n_basis, n_points)
        var = np.where(np.array(self.var) == var)[0].item()
        indices = list(range(self.K*var, self.K*(var+1)))
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.flatten()
        for i, k in enumerate(idx):
            ylabel = self.data[var, k]
            ypred = ygrid.T @ self.alpha[k, indices]  # ygrid %*% yfd$coefs[, k]
            axes[i].plot(ylabel, -xgrid, color='maroon', label='observed')
            axes[i].plot(ypred.flatten(), -xgrid, color='navy', label='fitted')
            axes[i].set_title(f'Obs {k}')
            axes[i].legend()
        plt.tight_layout()
        plt.show()

    def map_distribution(self, save = False):
        """
        Visualize number of points per pixel
        """
        lat_idx = np.digitize(self.gps[:, 0], self.grid.dataset.latitude.values) - 1
        lon_idx = np.digitize(self.gps[:, 1], self.grid.dataset.longitude.values) - 1
        Y = len(self.grid.dataset.latitude)
        X = len(self.grid.dataset.longitude)
        count_array = np.zeros((Y, X, 1))
        np.add.at(count_array, (lat_idx, lon_idx, 0), 1)
        fig, ax = plt.subplots(
            subplot_kw={'projection': ccrs.PlateCarree()},
            gridspec_kw={'wspace': 0.05})
        bounds = self._extract_grid_bounds()
        lat_min, lat_max = bounds['lat_min'], bounds['lat_max']
        lon_min, lon_max = bounds['lon_min'], bounds['lon_max']
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        img = ax.imshow(
            count_array[::-1],
            origin='upper',
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            norm=LogNorm(vmin=1, vmax=2000))
        ax.coastlines(linewidth=0.5, zorder=0)
        ax.set_xticks(np.arange(lon_min, lon_max + 1, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(lat_min, lat_max + 1, 5), crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, facecolor='lightgrey')
        fig.colorbar(img, ax=ax, label = "Number of profiles", shrink = 0.5)
        fig.show()
        if save :
            fig.savefig("spatial_distribution.pdf", bbox_inches='tight')

    def year_distribution(self, save = False):
        year = pd.Series(self.timestamp).dt.strftime('%Y').to_numpy().astype(int)
        fig, ax = plt.subplots(figsize=(6, 6))
        for source, color in zip(np.unique(self.source)[::-1], ['firebrick', 'steelblue']) :
            _year, count = np.unique(year[np.array(self.source) == source], return_counts=True)
            ax.bar(_year, count,
                   label = source,
                   color = color, alpha = 0.7)
        # ax.set_xticks(np.arange(np.nanmin(year), np.nanmax(year)+1) + 0.5)
        # ax.set_xticklabels(np.arange(np.nanmin(year), np.nanmax(year)+1))
        ax.set_ylabel('Number of profiles')
        ax.set_ylim(0, 110000)
        ax.set_xlabel('Year')
        ax.legend(loc = 'upper left')
        fig.show()
        if save :
            fig.savefig("year_distribution.pdf", bbox_inches='tight')

    def month_distribution(self, save = False):
        year = pd.Series(self.timestamp).dt.strftime('%m').to_numpy().astype(int)
        fig, ax = plt.subplots(figsize=(6, 6))
        for source, color in zip(np.unique(self.source)[::-1], ['firebrick', 'steelblue']) :
            _year, count = np.unique(year[np.array(self.source) == source], return_counts=True)
            ax.bar(_year, count,
                   label = source,
                   color = color, alpha = 0.7)
        # ax.set_xticks(np.arange(np.nanmin(year), np.nanmax(year)+1) + 0.5)
        # ax.set_xticklabels(np.arange(np.nanmin(year), np.nanmax(year)+1))
        ax.set_ylabel('Number of profiles')
        ax.set_ylim(0, 200000)
        ax.set_xlabel('Month')
        ax.legend(loc = 'upper left')
        fig.show()
        if save :
            fig.savefig("month_distribution.pdf", bbox_inches='tight')

    def depth_distribution(self, save = False) :
        """
        Distribution of profiles per depth bin
        """
        max_depth = []
        for d, x in zip(self.depth, self.data[self.var[0]]):
            valid_mask = ~np.isnan(x)
            if np.any(valid_mask):
                max_depth.append(np.nanmax(d[valid_mask]))
            else :
                max_depth.append(np.nan)
        max_depth = np.array(max_depth)
        fig, ax = plt.subplots(figsize=(6, 6))
        for source, color in zip(np.unique(self.source)[::-1], ['firebrick', 'steelblue']) :
            ax.hist(max_depth[np.array(self.source) == source],
                    np.arange(np.nanmin(max_depth), np.nanmax(max_depth), 50),
                    label = source, alpha = 0.7, color = color)
        ax.set_xlim(0, 2001)
        ax.set_ylabel('Number of profiles')
        ax.set_xlabel('Maximum depth of profile (m)')
        ax.legend()
        fig.show()
        if save :
            fig.savefig("depth_distribution.pdf", bbox_inches='tight')


    def save_data(self, save_path: str, npc=15, **kwargs):

        lat_idx = np.digitize(self.gps[:, 0], self.grid.dataset.latitude.values) - 1
        lon_idx = np.digitize(self.gps[:, 1], self.grid.dataset.longitude.values) - 1
        time_idx = np.digitize(pd.to_datetime(self.timestamp).values.astype("datetime64[ns]").astype('int64'),
                               self.grid.dataset.time.values.astype("datetime64[ns]").astype('int64')) - 1
        pc_vals = self.mfpca["pc"][:, : npc]
        T = len(self.grid.dataset.time)
        Y = len(self.grid.dataset.latitude)
        X = len(self.grid.dataset.longitude)
        sum_array = np.zeros((T, Y, X, npc))
        count_array = np.zeros((T, Y, X, 1))
        np.add.at(sum_array, (time_idx, lat_idx, lon_idx, slice(None)), pc_vals)
        np.add.at(count_array, (time_idx, lat_idx, lon_idx, 0), 1)
        mean_array = sum_array / count_array
        mean_array[np.tile(count_array, npc) == 0] = np.nan
        # ds = self.grid.dataset.copy()
        self.grid.dataset["PC"] = xr.DataArray(
            mean_array,
            dims=("time", "latitude", "longitude", "pc"))
        self.grid.dataset.to_netcdf(os.path.join(save_path, kwargs['grid_file'] if 'grid_file' in kwargs.keys() else 'PC_grid.nc'))
        np.savez(os.path.join(save_path, kwargs['mfpca'] if 'mfpca' in kwargs.keys() else 'mfpca'),
                 timestamp = self.timestamp,
                 gps = self.gps,
                 fns = self.fns,
                 source = self.source,
                 profile = self.profile,
                 depth = self.depth,
                 alpha = self.alpha,
                 data = self.data,
                 **self.mfpca)

    def filter_depth(self):

        gps_list, timestamp_list, fns_list, profile_list, source_list = [], [], [], [], []
        data_filtered = {key: [] for key in self.data.keys()}
        depth_target = np.arange(self.depth_min, self.depth_max + 1, 1).astype(int)
        if self.depth_reconstruction :
            gps_recon, timestamp_recon, fns_recon, profile_recon, source_recon, depth_recon = [], [], [], [], [], []
            data_recon = {key: [] for key in self.data.keys()}
        data_keys = list(self.data.keys())
        data_values = [self.data[key] for key in data_keys]
        n = len(self.timestamp)
        #for i, (_depth, _gps, _timestamp, _fn, _profile, _source) in enumerate(zip(self.depth, self.gps, self.timestamp, self.fns, self.profile, self.source)):
        for i in range(n):
            _depth = self.depth[i].flatten()
            _data = [values[i] for values in data_values]
            mask = np.isnan(_data[0])
            if np.all(mask) :
                continue
            max_depth = np.nanmax(_depth[~mask])
            min_depth = np.nanmin(_depth[~mask])
            _gps = self.gps[i]
            _timestamp = self.timestamp[i]
            _fn = self.fns[i]
            _profile = self.profile[i]
            _source = self.source[i]
            if (max_depth >= self.depth_max) and (min_depth <= self.depth_min):
                sample_valid, interpolated_sample = self.filter_helper(_depth, depth_target, _data)
                if not sample_valid:
                    continue
                for j, key in enumerate(self.data.keys()):
                    data_filtered[key].append(interpolated_sample[j])
                gps_list.append(_gps)
                timestamp_list.append(_timestamp)
                fns_list.append(_fn)
                source_list.append(_source)
                profile_list.append(_profile)
            elif self.depth_reconstruction and (max_depth >= self.depth_reconstruction) and (min_depth <= self.depth_min):
                recon_target = np.arange(self.depth_min, max_depth + 1, 1).astype(int)
                sample_valid, interpolated_sample = self.filter_helper(_depth, recon_target, _data)
                if not sample_valid:
                    continue
                for j, key in enumerate(self.data.keys()):
                    data_recon[key].append(interpolated_sample[j])
                depth_recon.append(recon_target)
                gps_recon.append(_gps)
                timestamp_recon.append(_timestamp)
                fns_recon.append(_fn)
                source_recon.append(_source)
                profile_recon.append(_profile)
        self.depth = depth_target
        self.gps = np.array(gps_list)
        self.timestamp = np.array(timestamp_list)
        self.fns = np.array(fns_list)
        self.source = np.array(source_list)
        self.profile = np.array(profile_list)
        data = {key: np.array(values) for key, values in data_filtered.items()}
        self.data = np.stack([data[key] for key in data.keys()], axis = 0)
        if self.depth_reconstruction:
            self.depth_recon = depth_recon
            self.gps_recon = np.array(gps_recon)
            self.timestamp_recon = np.array(timestamp_recon)
            self.fns_recon = np.array(fns_recon)
            self.source_recon = np.array(source_recon)
            self.profile_recon = np.array(profile_recon)
            self.data_recon = data_recon
            # data = {key: np.array(values) for key, values in data_recon.items()}
            # self.data_recon = np.stack([data[key] for key in data.keys()], axis = 0)
            del data_recon
        del data_filtered
        del data

    @staticmethod
    def filter_helper(depth, depth_target, data) :
        depth_range = np.arange(int(np.floor(np.nanmin(depth))), np.nanmax(depth) + 1, 1).astype(int)
        sample_valid = True
        interpolated_sample = []
        for data_array in data:
            _data = np.asarray(data_array).flatten()
            _data_interp = np.interp(depth_range, depth, _data)
            _data_interp = _data_interp[np.isin(depth_range, depth_target)]
            if np.isnan(_data_interp).any():
                sample_valid = False
                break
            interpolated_sample.append(_data_interp)
        return sample_valid, interpolated_sample

    def parallel_reconstruction(self, h=5):
        Wdem = self.mfpca['Wdem']
        Wdeminv = self.mfpca['Wdeminv']
        W_i = self.mfpca['W_i']
        _data_recon = np.empty((2, len(self.timestamp_recon), len(self.depth)))
        pc_recon = []
        grid_points = np.arange(self.depth_min, self.depth_max + 1)
        X_0 = self.ydf[0].coefficients
        X_1 = self.ydf[1].coefficients
        X = np.hstack((X_0, X_1))
        phi = np.squeeze(self.basis(grid_points), axis=2)
        y_eval_0 = X_0 @ phi
        y_eval_1 = X_1 @ phi
        estimsig0 = 1 / self.data.shape[1] * np.sum((y_eval_0 - self.data[0]) ** 2, axis=0)
        estimsig1 = 1 / self.data.shape[1] * np.sum((y_eval_1 - self.data[1]) ** 2, axis=0)
        regularization = L2Regularization(LinearDifferentialOperator(2), regularization_parameter=1000)
        smoother = BasisSmoother(basis=self.basis, regularization=regularization, return_basis=True)
        sig0 = smoother.fit_transform(FDataGrid(data_matrix=estimsig0, grid_points=grid_points))
        sig1 = smoother.fit_transform(FDataGrid(data_matrix=estimsig1, grid_points=grid_points))
        estimsig0 = np.squeeze(sig0.coefficients @ phi)
        estimsig1 = np.squeeze(sig1.coefficients @ phi)
        tasks = range(15, 20000)

        def worker_chunk(idxs):
            return [_reconstruct_single_profile(i,
                                                X,
                                                self.gps,
                                                self.gps_recon,
                                                self.depth_recon,
                                                self.basis,
                                                phi,
                                                estimsig0,
                                                estimsig1,
                                                self.depth_min,
                                                self.K,
                                                Wdem,
                                                Wdeminv,
                                                W_i,
                                                h,
                                                self.data_recon[self.var[0]],
                                                self.data_recon[self.var[1]]) for i in idxs]

        chunks = np.array_split(range(0, 100), 4)

        with tqdm_joblib(tqdm(total=len(tasks), leave=True)):
            results = Parallel(n_jobs=-1, backend="loky")(delayed(worker_chunk)(chunk) for chunk in chunks)
        for i, recon, pc in results:
            return recon
            _data_recon[:, i, :] = recon
            pc_recon.append(pc)
        self.method = np.concatenate((['classic'] * self.data.shape[1], ['reconstruction'] * len(pc_recon)))
        self.mfpca['pc'] = np.concatenate([self.mfpca['pc'], pc_recon], axis=0)
        self.data = np.concatenate([self.data, _data_recon], axis=1)
        self.timestamp = np.concatenate([self.timestamp, self.timestamp_recon])
        self.gps = np.vstack([self.gps, self.gps_recon])
        self.source = np.concatenate([self.source, self.source_recon])
        self.fns = np.concatenate([self.fns, self.fns_recon])
        self.profile = np.concatenate([self.profile, self.profile_recon])

        # def load_model(self, path : str):
        #     cmems = nc.Dataset(path)
        #     K, M, L = cmems['time'].shape[0], cmems['latitude'].shape[0], cmems['longitude'].shape[0]  # time, lat, lon
        #     time = cmems['time'][:].data.reshape(K, 1, 1)
        #     self.timestamp = np.broadcast_to(time, (K, M, L)).flatten()
        #     lat, lon = np.meshgrid(cmems['latitude'][:].data, cmems['longitude'][:].data, indexing="ij")
        #     self.gps = [np.broadcast_to(lat[None, :, :], (K, M, L)).flatten(),
        #                 np.broadcast_to(lon[None, :, :], (K, M, L)).flatten()]
        #     # temp has shape (K, 31, M, L) — move the 31-axis to the end
        #     self.data = {"TEMP_ADJUSTED": cmems['thetao'][:].data.transpose(0, 2, 3, 1).reshape(K * M * L, cmems['depth'][:].data.shape[0]),
        #                  "PSAL_ADJUSTED": cmems['so'][:].data.transpose(0, 2, 3, 1).reshape(K * M * L, cmems['depth'][:].data.shape[0])}
        #     self.depth = cmems['depth'][:].data
        #     cmems.close()

        #
                # elif self.depth_reconstruction & (np.nanmax(_depth) > self.depth_reconstruction) & (np.nanmin(_depth) < self.depth_min):
                #     depth_range = np.arange(int(np.floor(np.nanmin(_depth))), np.nanmax(_depth) + 1)
                #     sample_valid = True
                #     interpolated_sample = {}
                #     for key, data_array in self.data.items():
                #         _data = np.asarray(data_array[i]).flatten()
                #         _data_interp = np.interp(depth_range, _depth, _data)
                #         _data_interp = _data_interp[np.isin(depth_range, recon_target)]
                #         if np.isnan(_data_interp).any():
                #             sample_valid = False
                #             break
                #         interpolated_sample[key] = _data_interp
                #     if not sample_valid:
                #         continue
                #     gps_recon.append(_gps)
                #     timestamp_recon.append(_timestamp)
                #     fns_recon.append(_fn)
                #     source_recon.append(_source)
                #     profile_recon.append(_profile)
                #     for key in self.data.keys():
                #         data_recon[key].append(interpolated_sample[key])

        # def filter_depth_univariate(self):
        #     depth, data, gps, fns, profile, source = [], [], [], [], [], []
        #     for _depth, _data, _gps, _fn, _profile, _source in zip(self.depth, self.data, self.gps, self.fns,
        #                                                            self.profile, self.source):
        #         _depth = _depth.flatten()
        #         _data = _data.flatten()
        #         if (np.nanmax(_depth) > self.depth_max) & (np.nanmin(_depth) < self.depth_min):
        #             depth_range = np.arange(int(np.floor(np.nanmin(_depth))), np.nanmax(_depth) + 1)
        #             _data_interp = np.interp(depth_range, _depth, _data)
        #             _depth_interp = np.arange(self.depth_min, self.depth_max + 1, 1).astype(int)
        #             _data_interp = _data_interp[np.isin(depth_range, _depth_interp)]
        #             if np.isnan(_data_interp).sum() > 0:
        #                 continue
        #             gps.append(_gps)
        #             depth.append(_depth_interp)
        #             data.append(_data_interp)
        #             fns.append(_fn)
        #             source.append(_source)
        #             profile.append(_profile)
        #     self.data = np.array(data)
        #     self.depth = np.array(depth)
        #     self.gps = np.array(gps)
        #     self.fns = np.array(fns)
        #     self.source = np.array(source)
        #     self.profile = np.array(profile)