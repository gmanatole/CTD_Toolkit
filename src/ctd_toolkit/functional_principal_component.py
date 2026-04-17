from ctd_toolkit.utils.sql_query import SQLQuery
from typing import Union
from concurrent.futures import ProcessPoolExecutor
from ctd_toolkit.utils.workers import _process_file_reader
from tqdm import tqdm
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from scipy.linalg import cholesky, solve_triangular
import numpy as np
import matplotlib.pyplot as plt

class FunctionalPrincipalComponent:

    K = 50
    lam = 0.05

    def __init__(self,
                 grid,
                 path : str,
                 source: SQLQuery,
                 var : Union[str, list[str]],
                 depth_min : int = None,
                 depth_max : int = None,
                 ):

        self.grid = grid
        self.source = source
        self.path = path if (path.startswith("'") and path.endswith("'")) else f"'{path}'"

        self.var = var
        self.depth_min = depth_min
        self.depth_max = depth_max

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

    def filter_depth(self) :
        depth, data, gps = [], [], []
        for _depth, _data, _gps in zip(self.depth, self.data, self.gps):
            _depth = _depth.flatten()
            _data = _data.flatten()
            if (np.nanmax(_depth) > self.depth_max) & (np.nanmin(_depth) < self.depth_min):
                depth_range = np.arange(int(np.floor(np.nanmin(_depth))), np.nanmax(_depth) + 1)
                _data_interp = np.interp(depth_range, _depth, _data)
                _depth_interp = np.arange(self.depth_min, self.depth_max + 1, 1).astype(int)
                _data_interp = _data_interp[np.isin(depth_range, _depth_interp)]
                if np.isnan(_data_interp).sum() > 0:
                    continue
                gps.append(_gps)
                depth.append(_depth_interp)
                data.append(_data_interp)
        self.data = np.array(data)
        self.depth = np.array(depth)
        self.gps = np.array(gps)

    def base_projection(self) :
        # Create bspline base
        self.basis = BSplineBasis(domain_range=(self.depth_min, self.depth_max), n_basis=self.K, order=4)
        # Project data to base, shape should be (Nobs x Npoints)
        fd = FDataGrid(data_matrix=self.data, grid_points=np.arange(self.depth_min, self.depth_max + 1))
        # Lambda penalty on second derivative
        regularization = L2Regularization(
            LinearDifferentialOperator(2),
            regularization_parameter=self.lam)
        smoother = BasisSmoother(
            basis=self.basis,
            regularization=regularization,
            return_basis=True)
        self.ydf = smoother.fit_transform(fd)

    def fpca_hilbert(self):
        self.alpha = self.ydf.coefficients
        alpha_mean = self.alpha.mean(axis=0)
        alpha_std = self.alpha.std(axis=0)
        n = self.alpha.shape[0]
        C = self.alpha - alpha_mean
        C_norm = (self.alpha - alpha_mean) / alpha_std
        self.mat_cov = (C.T @ C) / n
        self.mat_cor = (C_norm.T @ C_norm) / n
        W = self.basis.gram_matrix()
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

    def profile_reconstruction(self, K = None) :
        if not K :
            K = self.K
        grid_points = np.arange(self.depth_min, self.depth_max + 1)
        alpha_recon = self.alpha.mean(axis=0) + self.mfpca['pc'][:, :K] @ self.mfpca['vectors'][:, :K].T
        phi = self.basis(grid_points)
        x_recon = alpha_recon @ phi.squeeze()
        self.X = x_recon

    def visualize_projection(self) :
        idx = np.random.choice(self.data.shape[0], 10, replace=False)
        xgrid = np.arange(self.depth_min, self.depth_max + 1)
        # Evaluate basis on grid - equivalent to eval.basis()
        ygrid = self.basis.to_basis()(xgrid)
        # shape: (n_basis, n_points)
        coefficients = self.ydf.coefficients  # shape: (Nobs, K)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.flatten()
        for i, k in enumerate(idx):
            ylabel = self.data[k, :]
            ypred = ygrid.T @ coefficients[k]  # ygrid %*% yfd$coefs[, k]
            axes[i].plot(ylabel, -xgrid, color='maroon', label='observed')
            axes[i].plot(ypred.flatten(), -xgrid, color='navy', label='fitted')
            axes[i].set_title(f'Obs {k}')
            axes[i].legend()
            plt.tight_layout()
            plt.show()

    def corresponding_files(self):

        bounds = self._extract_grid_bounds()
        sql = self._file_query(bounds=bounds)
        df = self.source.query(sql)
        return df

    def get_profiles(self, df, workers = 8) :
        # group by file
        file_groups = list(df.groupby("fn"))
        # prepare tasks for workers
        tasks = [
            (fn, group.reset_index(drop=True), self.var)
            for fn, group in file_groups
        ]
        depth, data, gps = [], [], []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = executor.map(_process_file_reader, tasks)
            for result in tqdm(futures, total=len(tasks), desc="Processing files"):
                data.extend(result[1])
                depth.extend(result[0])
                gps.extend(result[2])
        self.data = data
        self.depth = depth
        self.gps = gps

    def _extract_grid_bounds(self):

        ds = self.grid.dataset

        return {
            "time_min": str(ds.time.min().values),
            "time_max": str(ds.time.max().values),
            "lat_min": float(ds.latitude.min().values),
            "lat_max": float(ds.latitude.max().values),
            "lon_min": float(ds.longitude.min().values),
            "lon_max": float(ds.longitude.max().values),
        }

    def _file_query(self, bounds):
        sql = f"""
        SELECT *
        FROM {self.path}
        WHERE
            timestamp BETWEEN TIMESTAMP '{bounds["time_min"]}' AND TIMESTAMP '{bounds["time_max"]}'
            AND lat BETWEEN {bounds["lat_min"]} AND {bounds["lat_max"]}
            AND lon BETWEEN {bounds["lon_min"]} AND {bounds["lon_max"]}
        """
        return sql