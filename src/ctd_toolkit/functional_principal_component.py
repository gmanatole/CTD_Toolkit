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
import numpy as np
import random
import matplotlib.pyplot as plt

class FunctionalPrincipalComponent:

    K = 100
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
        return self.__depth_max

    @depth_min.setter
    def depth_min(self, value):
        self.__depth_min = value

    def corresponding_files(self):

        bounds = self._extract_grid_bounds()
        sql = self._file_query(bounds=bounds)
        df = self.source.query(sql)
        return df

    def filter_depth(self) :
        pass

    def base_projection(self) :
        # Create bspline base
        self.basis = BSplineBasis(domain_range=(self.depth_min, self.depth_max), n_basis=self.K, order=4)
        self.data = self.data[self.depth_min:self.depth_max + 1, :].T
        # Project data to base, shape should be (Nobs x Npoints)
        fd = FDataGrid(data_matrix=self.data, grid_points=np.arange(self.depth_min, self.depth_max + 1))
        # Lambda penalty on second derivative
        regularization = L2Regularization(
            LinearDifferentialOperator(2),
            regularization_parameter=self.lam)
        smoother = BasisSmoother(
            basis=self.basis,
            regularization=regularization)
        self.ydf = smoother.fit_transform(fd)

    def visualize_projection(self) :
        idx = random.sample(range(0, self.data.shape[1] - 1), 4)
        y = list(range(0, self.data.shape[0] - 1))
        x = self.basis(y)
        fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
        for i, _idx in enumerate(idx) :
            ax[i].scatter(x @ self.yfd.coefficients[_idx, :], x)
            ax[i].scatter(self.data[:, _idx], x)
            ax[i].invert_yaxis()
        fig.show()

    def get_profiles(self, df, workers = 8) :

        # group by file
        file_groups = list(df.groupby("fn"))
        # prepare tasks for workers
        tasks = [
            (fn, group.reset_index(drop=True), self.var)
            for fn, group in file_groups
        ]
        data = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = executor.map(_process_file_reader, tasks)
            for result in tqdm(futures, total=len(tasks), desc="Processing files"):
                data.append(result)
        self.data = data

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