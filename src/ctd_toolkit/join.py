import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from typing import Optional
from ctd_toolkit.utils.sql_query import SQLQuery
from ctd_toolkit.backend.read_meop import ReadMEOP
from ctd_toolkit.backend.read_argo import ReadArgo

class Join:
    """
    Class that handles the join of in-situ data profiles to the grid
    onto an AdvancedSpatioTemporalGrid.
    """

    def __init__(self,
                 grid,
                 path : str,
                 source: SQLQuery,
                 view : bool = True):

        self.grid = grid
        self.source = source
        self.path = path if (path.startswith("'") and path.endswith("'")) else f"'{path}'"

        if self.grid.dataset is None:
            raise RuntimeError("Grid must be built before joining.")
        if view :
            self._create_view()
            self.path = "profiles"

    def corresponding_files(self):

        bounds = self._extract_grid_bounds()
        sql = self._file_query(bounds=bounds)
        df = self.source.query(sql)
        return df

    def join_data(self, var : str):

        ds = self.grid.dataset
        df = self.corresponding_files()

        time_idx = ds.indexes["time"].get_indexer(df["timestamp"], method="nearest")
        lat_idx = ds.indexes["latitude"].get_indexer(df["lat"], method="nearest")
        lon_idx = ds.indexes["longitude"].get_indexer(df["lon"], method="nearest")
        data = ds["values"].data
        temp_df = pd.DataFrame({
            "fn": df.fn,
            "profile": df.profile,
            "source": df.source,
            "time": time_idx,
            "lat": lat_idx,
            "lon": lon_idx})
        temp_df = temp_df.groupby(["lat", "lon", "time"]).agg(list).reset_index()
        depth_index = ds.indexes["depth"]

        for cell in tqdm(temp_df.itertuples(index=False), total=len(temp_df),
                         desc="Joining in-situ data to grid"):
            pres_list = []
            val_list = []
            for j, fn in enumerate(cell.fn):
                if cell.source[j] == "MEOP":
                    reader = ReadMEOP(fn).read(var=var, profiles=cell.profile[j])
                else:
                    reader = ReadArgo(fn).read(var=var, profiles=cell.profile[j])
                pres_list.append(reader["PRES"].ravel())
                val_list.append(reader[var].ravel())
            pres = np.concatenate(pres_list)
            vals = np.concatenate(val_list)
            pres_idx = depth_index.get_indexer(pres, method="nearest")
            tmp = pd.DataFrame({"PRES": pres_idx, var: vals})
            tmp = tmp.groupby("PRES").mean()
            for pres_i, value in tmp[var].items():
                data[cell.time, pres_i, cell.lat, cell.lon] = value

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

    def _create_view(self) :
        self.source.query(f"""
                   CREATE VIEW profiles AS
                   SELECT *
                   FROM
                   {self.path}
                   """)

    # def join_files(self, var : str):
    #     ds = self.grid.dataset
    #     df = self.corresponding_files()
    #     time_idx = ds.indexes["time"].get_indexer(df["timestamp"], method="nearest")
    #     lat_idx = ds.indexes["latitude"].get_indexer(df["lat"], method="nearest")
    #     lon_idx = ds.indexes["longitude"].get_indexer(df["lon"], method="nearest")
    #     data = ds["values"].data
    #
    #     temp_df = pd.DataFrame({'fn': df.fn, 'profile': df.profile, 'source': df.source, 'time': time_idx, 'lat': lat_idx, 'lon': lon_idx})
    #     temp_df = temp_df.groupby(['lat', 'lon', 'time']).agg(list).reset_index()
    #     for i, cell in tqdm(temp_df.iterrows(), total=len(temp_df), desc='Joining in-situ data to grid') :
    #         values = pd.DataFrame(columns = ['PRES', var])
    #         for j, fn in enumerate(cell["fn"]) :
    #             if cell.source[j] == 'MEOP' :
    #                 reader = ReadMEOP(fn).read(var = var, profiles = cell.profile[j])
    #             elif cell.source[j] == 'ARGO' :
    #                 reader = ReadArgo(fn).read(var = var, profiles = cell.profile[j])
    #             pressure = reader['PRES']
    #             value = reader[var]
    #             values = pd.concat((values,
    #                            pd.DataFrame({"PRES": pressure.ravel(), var: value.ravel()})))
    #         values['PRES'] = ds.indexes["depth"].get_indexer(values['PRES'], method="nearest")
    #         values = values.groupby('PRES').agg('mean').reset_index()
    #         for k, row in values.iterrows() :
    #             data[cell.time, row.PRES, cell.lat, cell.lon] = row[var]

    # def join_complex(
    #     self,
    #     table: str,
    #     variable_name: str,
    #     time_col: str = "time",
    #     lat_col: str = "latitude",
    #     lon_col: str = "longitude",
    #     value_col: str = "value",
    #     aggregation: Optional[str] = None,
    # ) -> xr.Dataset:
    #     """
    #     Main entry point for joining SQL data to grid.
    #
    #     Parameters:
    #         variable_name : name of variable to store in grid
    #         aggregation : aggregation method
    #     """
    #
    #     bounds = self._extract_grid_bounds()
    #
    #     sql = self._build_query(
    #         table=table,
    #         time_col=time_col,
    #         lat_col=lat_col,
    #         lon_col=lon_col,
    #         value_col=value_col,
    #         bounds=bounds,
    #         aggregation=aggregation,
    #     )
    #
    #     df = self.source.query(sql)
    #
    #     data_array = self._map_to_grid(
    #         df=df,
    #         time_col=time_col,
    #         lat_col=lat_col,
    #         lon_col=lon_col,
    #         value_col=value_col,
    #     )
    #
    #     self.grid.dataset[variable_name] = data_array
    #
    #     return self.grid.dataset
    #
    #
    #
    #
    # def _map_to_grid(
    #     self,
    #     df: pd.DataFrame,
    #     time_col: str,
    #     lat_col: str,
    #     lon_col: str,
    #     value_col: str,
    # ) -> xr.DataArray:
    #
    #     ds = self.grid.dataset
    #
    #     # Create empty array
    #     arr = np.full(ds["values"].shape, np.nan)
    #
    #     time_index = {t: i for i, t in enumerate(pd.to_datetime(ds.time.values))}
    #     lat_index = {v: i for i, v in enumerate(ds.latitude.values)}
    #     lon_index = {v: i for i, v in enumerate(ds.longitude.values)}
    #
    #     for _, row in df.iterrows():
    #         t = pd.to_datetime(row[time_col])
    #         lat = row[lat_col]
    #         lon = row[lon_col]
    #
    #         if t in time_index and lat in lat_index and lon in lon_index:
    #             arr[
    #                 time_index[t],
    #                 lat_index[lat],
    #                 lon_index[lon],
    #             ] = row[value_col]
    #
    #     return xr.DataArray(
    #         arr,
    #         dims=("time", "latitude", "longitude"),
    #         coords=ds.coords,
    #     )
    #
    # def _build_query(
    #     self,
    #     table,
    #     time_col,
    #     lat_col,
    #     lon_col,
    #     value_col,
    #     bounds,
    #     aggregation=None,
    # ):
    #
    #     value_expr = (
    #         f"{aggregation}({value_col})"
    #         if aggregation
    #         else value_col
    #     )
    #
    #     sql = f"""
    #     SELECT
    #         {time_col},
    #         {lat_col},
    #         {lon_col},
    #         {value_expr} AS value
    #     FROM {table}
    #     WHERE
    #         {time_col} BETWEEN '{bounds["time_min"]}' AND '{bounds["time_max"]}'
    #         AND {lat_col} BETWEEN {bounds["lat_min"]} AND {bounds["lat_max"]}
    #         AND {lon_col} BETWEEN {bounds["lon_min"]} AND {bounds["lon_max"]}
    #     """
    #
    #     if aggregation:
    #         sql += f"""
    #         GROUP BY {time_col}, {lat_col}, {lon_col}
    #         """
    #
    #     return sql