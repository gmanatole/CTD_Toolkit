import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime

class ReadMEOP :

    """
    Class to read MEOP profile from netCDF4 file
    """

    def __init__(self, fn) :
        """
        Stores absolute path and xarray object of profile
        parameters
        ----------
        fn : str : absolute path of netCDF4 file
        """

        self.fn = fn
        self.__ds = nc.Dataset(self.fn, 'r')
        self.__gps = None
        self.__timestamp = None
        self.__maximum_depth = None

    @property
    def timestamp(self) -> pd.Timestamp :
        """
        Returns timestamp in UTC of profiles
        """
        if self.__timestamp is None :
            if 'REFERENCE_DATE_TIME' in self.__ds.variables :
                ref_time = datetime.strptime(b''.join(self.__ds['REFERENCE_DATE_TIME']).decode('utf-8'), '%Y%m%d%H%M%S')
                self.__timestamp = ref_time + self.__ds['JULD'][:].data.astype('timedelta64[D]').astype('O')
            else :
                ref_time = datetime(1950,1,1)
                self.__timestamp = ref_time + self.__ds['TIME'][:].data.astype('timedelta64[D]').astype('O')
        return self.__timestamp

    @property
    def gps(self) -> np.ndarray :
        """
        Returns latitude, longitude of the N profiles in np.array of shape (N, 2)
        """
        if self.__gps is None :
            self.__gps = np.column_stack((self.__ds['LATITUDE'][:].filled(np.nan),
                                          self.__ds['LONGITUDE'][:].filled(np.nan)))
        return self.__gps

    @property
    def maximum_depth(self, reference = 'TEMP_ADJUSTED') -> np.ndarray :
        """
        Returns maximum depth of profile in meters computed wrt reference data
        """
        if self.__maximum_depth is None :
            if reference in self.__ds.variables.keys() :
                self.__maximum_depth = (~np.isnan(self.__ds[reference][:].filled(np.nan))).cumsum(1).argmax(1)
            else :
                self.__maximum_depth = np.nan
        return self.__maximum_depth

    def read(self, var : str | list[str], profiles : int | list[int]) :
        profiles = np.atleast_1d(profiles)
        latitude = self.gps[profiles, 0]
        longitude = self.gps[profiles, 1]
        timestamp = np.asarray(self.timestamp)[profiles]
        pressure = self.__ds["PRES_ADJUSTED"][profiles, :].filled(np.nan)

        data = {
            "TIMESTAMP": timestamp,
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "PRES": pressure
        }

        if isinstance(var, str):
            values = self.__ds[var][profiles, :].filled(np.nan)
            data[var] = values
            return data

        for _var in var:
            data[_var] = self.__ds[_var][profiles, :].filled(np.nan)

        return data

    def unformatted_data(self, var : str | list[str], profiles : int | list[int]) :
        profiles = np.atleast_1d(profiles)
        pressure = self.__ds["PRES_ADJUSTED"][profiles, :].filled(np.nan)
        data = {"PRES": pressure}
        if isinstance(var, str):
            values = self.__ds[var][profiles, :].filled(np.nan)
            data[var] = values
            return data
        for _var in var:
            data[_var] = self.__ds[_var][profiles, :].filled(np.nan)
        return data
        # pressure = self.__ds['PRES_ADJUSTED'][profiles].filled(np.nan)
        # if type(var) is str :
        #     data = self.__ds[var][profiles].filled(np.nan)
        #     return {'PRES': pressure, var : data}
        # data = {'PRES': pressure}
        # for _var in var :
        #     data = {**data, **{_var : self.__ds[_var][profiles].filled(np.nan)}}
        # return data
