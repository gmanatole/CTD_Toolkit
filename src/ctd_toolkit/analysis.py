
import xarray as xr
import numpy as np
from pathlib import Path


class Analysis:

    def __init__(
        self,
        grids_path: str,
        lat_range: tuple,
        lon_range: tuple,
        depth_range: tuple,
        time_range: tuple,
        chunks="auto"
    ):
        """
        Parameters
        ----------
        grids_path : str
            Directory containing NetCDF grids
        lat_range : (min, max)
        lon_range : (min, max)
        depth_range : (min, max)
        time_range : (start, end)
        """

        self.path = Path(grids_path)

        self.lat_range = lat_range
        self.lon_range = lon_range
        self.depth_range = depth_range
        self.time_range = time_range

        self.ds = self._load_dataset(chunks)
        self.subset = self._subset()

    def _load_dataset(self, chunks):

        ds = xr.open_mfdataset(
            self.path,
            combine="by_coords",
            parallel=True,
            chunks=chunks
        )

        return ds


    def _subset(self):

        ds = self.ds.sel(
            latitude=slice(*self.lat_range),
            longitude=slice(*self.lon_range),
            depth=slice(*self.depth_range),
            time=slice(*self.time_range)
        )

        return ds

    def latitude_gradient(self, var = 'TEMP_ADJUSTED') :
        fig, ax = plt.subplots(figsize=(5, 10))
        cmap = plt.get_cmap('viridis')
        month_mask = (np.isin(self.month_data, self.months))
        for year in np.unique(pd.to_datetime(self.timestamp).year):
            time_mask = (pd.to_datetime(self.ds['time']).year == year) & month_mask
            ax.scatter(np.nanmean(self.ds[var][:].data[time_mask, :, :], axis=(0, 2)),
                        self.ds['latitude'],
                        color=cmap((year - 2000) / 20),
                        s = 10, alpha = 0.7,
                        label=str(year))
        ax.legend()
        ax.invert_xaxis()

    def spatial_mean(self):

        return self.subset.mean(dim=["latitude", "longitude"])


    def vertical_mean(self):

        return self.subset.mean(dim="depth")


    def rolling_mean(self, window=12):
        return self.subset.rolling(time=window, center=True).mean()


    def climatology(self):
        return self.subset.groupby("time.month").mean()


    def anomalies(self):

        clim = self.climatology()
        anomalies = self.subset.groupby("time.month") - clim
        return anomalies


    def tendency(self):

        """
        Linear trend along time dimension
        """

        da = self.subset["values"]
        # convert time to numeric index
        t = xr.DataArray(
            np.arange(da.sizes["time"]),
            dims="time",
            coords={"time": da.time}
        )
        t_mean = t.mean("time")
        da_mean = da.mean("time")
        cov = ((t - t_mean) * (da - da_mean)).mean("time")
        var = ((t - t_mean) ** 2).mean("time")

        slope = cov / var
        slope.name = "tendency"

        return slope


    def summary(self, rolling_window=12):

        results = {}

        results["subset"] = self.subset
        results["spatial_mean"] = self.spatial_mean()
        results["vertical_mean"] = self.vertical_mean()
        results["rolling_mean"] = self.rolling_mean(rolling_window)
        results["climatology"] = self.climatology()
        results["anomalies"] = self.anomalies()
        results["tendency"] = self.tendency()

        return results