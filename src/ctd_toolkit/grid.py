from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from pyproj import CRS


@dataclass
class SpatialRange:
    """
    Spatial range for grid.
    Latitude and longitude are considered in decimal degrees.
    """
    start: float
    end: float
    step: float

    def validate(self, name: str):
        if self.step <= 0:
            raise ValueError(f"{name} step must be positive.")
        if self.end <= self.start:
            raise ValueError(f"{name} end must be greater than start.")

@dataclass
class DepthRange:
    """
    Depth range for grid.
    Depth is considered in meters.
    """
    start: float
    end: float
    step: float

    def validate(self):
        if (self.step <= 0) or (self.end < 0 | self.start < 0):
            raise ValueError("Depth step and bounds must be positive.")
        if self.end <= self.start:
            raise ValueError("Depth end must be greater than start. Depth data is considered positive.")

@dataclass
class TemporalRange:
    """
    Temporal range for grid.
    Timestep adjustable.
    freq = "D"      # every day
    freq = "W-MON"  # weekly on Mondays
    freq = "MS"     # first day of each month
    freq = "Q"      # quarter end
    freq = "YS"     # start of each year
    """
    start: Union[str, datetime]
    end: Union[str, datetime]
    freq: str

    def validate(self):
        if pd.Timestamp(self.end) <= pd.Timestamp(self.start):
            raise ValueError("End time must be after start time.")


class SpatioTemporalGrid:
    """
    Create a 4D spatio-temporal grid
    """

    def __init__(
        self,
        latitude: SpatialRange,
        longitude: SpatialRange,
        depth: DepthRange,
        time: TemporalRange,
        crs: Optional[str] = "EPSG:4326",
        chunks: tuple =  None
    ):
        latitude.validate("Latitude")
        longitude.validate("Longitude")
        depth.validate()
        time.validate()

        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time = time
        self.crs = CRS.from_user_input(crs)
        self.chunks = chunks

        self.dataset: Optional[xr.Dataset] = None


    def _generate_coordinates(self):
        lat = np.arange(
            self.latitude.start,
            self.latitude.end + self.latitude.step,
            self.latitude.step
        )

        lon = np.arange(
            self.longitude.start,
            self.longitude.end + self.longitude.step,
            self.longitude.step
        )

        depth = np.arange(
            self.depth.start,
            self.depth.end + self.depth.step,
            self.depth.step
        )

        time = pd.date_range(
            start=self.time.start,
            end=self.time.end,
            freq=self.time.freq
        )

        return lat, lon, depth, time

    def build(self):

        lat, lon, depth, time = self._generate_coordinates()
        if not self.chunks:
            self.chunks = (min(len(time), 10), min(len(depth), 50), min(len(lat), 50), min(len(lon), 50))

        ds = xr.Dataset(
            coords={
                "time": time,
                "depth": depth,
                "latitude": lat,
                "longitude": lon,
            },
            attrs={
                "crs": self.crs.to_string(),
                "description": "4D spatio-temporal grid"
            }
        )

        self.dataset = ds
        return ds

    # def build(self, lazy: bool = True):
    #     lat, lon, depth, time = self._generate_coordinates()
    #
    #     shape = (len(time), len(depth), len(lat), len(lon))
    #
    #     if lazy:
    #         if not self.chunks :
    #             self.chunks = (min(len(time), 10), min(len(depth), 50), min(len(lat), 50), min(len(lon), 50))
    #         data = da.empty(shape, chunks=self.chunks, dtype='float32')
    #         data[:] = np.nan
    #     else:
    #         data = np.full(shape, np.nan, dtype='float32')
    #
    #     ds = xr.Dataset(
    #         data_vars={
    #             "values": (("time", "depth", "latitude", "longitude"), data)
    #         },
    #         coords={
    #             "time": time,
    #             "depth": depth,
    #             "latitude": lat,
    #             "longitude": lon,
    #         },
    #         attrs={
    #             "crs": self.crs.to_string(),
    #             "description": "4D spatio-temporal grid"
    #         }
    #     )
    #
    #     self.dataset = ds
    #     return ds


    def to_crs(self, new_crs: str):
        """
        Reproject coordinates (only valid for projected CRS use cases).
        For lat/lon (EPSG:4326) this stores metadata only.
        """
        new_crs_obj = CRS.from_user_input(new_crs)
        self.dataset.attrs["crs"] = new_crs_obj.to_string()
        self.crs = new_crs_obj
        return self.dataset


    def compute(self):
        """
        Trigger computation if using dask.
        """
        if self.dataset is None:
            raise RuntimeError("Build the grid first.")
        self.dataset = self.dataset.compute()
        return self.dataset

    @property
    def shape(self):
        if self.dataset is None:
            raise RuntimeError("Build the grid first.")
        return self.dataset["values"].shape

    def __repr__(self):
        return (
            f"AdvancedSpatioTemporalGrid(\n"
            f"  Time: {self.time.start} to {self.time.end} every {self.time.freq}\n"
            f"  Depth: {self.depth.start} m to {self.depth.end} m every {self.depth.step} m\n"
            f"  Latitude: {self.latitude.start} to {self.latitude.end}, resolution {self.latitude.step}°\n"
            f"  Longitude: {self.longitude.start} → {self.longitude.end}, resolution {self.longitude.step}°\n"
            f"  CRS: {self.crs.to_string()}\n"
            f")"
        )

    def save(self, path: str):

        import netCDF4

        if self.dataset is None:
            raise RuntimeError("Build the grid first.")

        ds = self.dataset

        nc = netCDF4.Dataset(path, "w")

        # create dimensions
        nc.createDimension("time", len(ds.time))
        nc.createDimension("depth", len(ds.depth))
        nc.createDimension("latitude", len(ds.latitude))
        nc.createDimension("longitude", len(ds.longitude))

        # create coordinate variables
        time_var = nc.createVariable("time", "f8", ("time",))
        depth_var = nc.createVariable("depth", "f4", ("depth",))
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))

        time_var[:] = ds.time.values.astype("datetime64[s]").astype(float)
        time_var.units = 'seconds since 1970-01-01'
        depth_var[:] = ds.depth.values
        lat_var[:] = ds.latitude.values
        lon_var[:] = ds.longitude.values

        # create main variable (chunked + compressed)
        nc.createVariable(
            "values",
            "f4",
            ("time", "depth", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=np.nan,
            chunksizes=self.chunks
        )

        nc.setncattr("crs", self.crs.to_string())
        nc.setncattr("description", "4D spatio-temporal grid")

        nc.close()

    # def save(self, path: str, engine: str = "netcdf4", compute: bool = False):
    #     """
    #     Save the grid to a NetCDF4 file.
    #
    #     Parameters
    #     ----------
    #     path : str
    #         Output file path (.nc)
    #     engine : str
    #         Backend engine ("netcdf4", "h5netcdf", or "scipy")
    #     compute : bool
    #         If True, compute dask arrays before saving
    #     """
    #     if self.dataset is None:
    #         raise RuntimeError("Build the grid first.")
    #
    #     ds = self.dataset
    #
    #     encoding = {
    #         "values": {
    #             "zlib": True,
    #             "complevel": 4,
    #             "dtype": "float32",
    #             "_FillValue": np.nan,
    #             "chunksizes": self.chunks
    #         }
    #     }
    #
    #     if compute and hasattr(ds["values"].data, "compute"):
    #         ds = ds.compute()
    #
    #     ds.to_netcdf(path, engine=engine, encoding=encoding)


    # def add_variable(self, name: str, data: Union[np.ndarray, da.Array]):
    #     if self.dataset is None:
    #         raise RuntimeError("Build the grid first.")
    #
    #     if data.shape != (
    #         len(self.dataset.time),
    #         len(self.dataset.depth),
    #         len(self.dataset.latitude),
    #         len(self.dataset.longitude),
    #     ):
    #         raise ValueError("Data shape does not match grid dimensions.")
    #
    #     self.dataset[name] = (("time", "depth", "latitude", "longitude"), data)