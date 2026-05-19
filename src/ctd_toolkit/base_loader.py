from abc import ABC, abstractmethod
from ctd_toolkit.utils.sql_query import SQLQuery
import netCDF4 as nc
import numpy as np
import pickle

class BaseLoader(ABC):
    """
    Abstract class that handles the join of in-situ data profiles to a grid
    """

    def __init__(self,
                 grid,
                 path : str,
                 source: SQLQuery,
                 ):

        self.grid = grid
        self.source = source
        self.path = path if (path.startswith("'") and path.endswith("'")) else f"'{path}'"


    def corresponding_files(self):

        bounds = self._extract_grid_bounds()
        sql = self._file_query(bounds=bounds)
        df = self.source.query(sql)
        return df

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

    def save_raw(self, path : str) :
        with open(path, "wb") as f:
            pickle.dump({
                "data": self.data,
                "depth": self.depth,
                "timestamp": self.timestamp,
                "gps": self.gps,
                "profile": self.profile,
                "source": self.source,
                "fns": self.fns,
            }, f)

    def load_raw(self, path : str) :
        with open(path, "rb") as f:
            _data = pickle.load(f)
        self.data = _data['data']
        self.depth = _data['depth']
        self.timestamp = _data['timestamp']
        self.gps = _data['gps']
        self.profile = _data['profile']
        self.source = _data['source']
        self.fns = _data['fns']

    def load_model(self, path : str):
        cmems = nc.Dataset(path)
        K = cmems['time'].shape[0]
        M = cmems['latitude'].shape[0]
        L = cmems['longitude'].shape[0]
        D = cmems['depth'].shape[0]

        self.timestamp = np.repeat(cmems['time'][:], M * L)
        lat = np.repeat(cmems['latitude'][:], L)
        lat = np.tile(lat, K)
        lon = np.tile(cmems['longitude'][:], M)
        lon = np.tile(lon, K)
        self.gps = np.column_stack((lat, lon))
        N = K * M * L
        temp = np.empty((N, D), dtype=np.float32)
        psal = np.empty((N, D), dtype=np.float32)
        for k in range(K):
            t = cmems['thetao'][k]  # shape (D, M, L)
            s = cmems['so'][k]
            t = np.transpose(t, (1, 2, 0)).reshape(M * L, D)
            s = np.transpose(s, (1, 2, 0)).reshape(M * L, D)
            start = k * M * L
            end = (k + 1) * M * L
            temp[start:end] = t
            psal[start:end] = s
        self.data = {
            "TEMP_ADJUSTED": temp.astype(np.float32),
            "PSAL_ADJUSTED": psal.astype(np.float32)
        }
        del temp, psal, lat, lon
        self.depth = cmems['depth'][:].data
        self.fns = np.full(len(self.timestamp), np.nan)
        self.profile = np.full(len(self.timestamp), np.nan)
        self.source = np.full(len(self.timestamp), 'cmems')
        cmems.close()

    @abstractmethod
    def get_profiles(self):
        pass