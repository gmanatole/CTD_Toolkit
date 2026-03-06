import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime

class ReadArgo :

    """
    Class to read ARGO float profile from netCDF4 file
    """

    def __init__(self, fn) :
        """
        Stores absolute path and xarray object of profile
        parameters
        ----------
        fn : str : absolute path of netCDF4 file
        """
        self.fn = fn
        self.__ds = nc.Dataset(self.fn)
        self.__gps = None
        self.__timestamp = None


    @property
    def timestamp(self) -> pd.Timestamp :
        """
        Returns timestamp in UTC of profile
        """
        if self.__timestamp is None :
            ref_time = datetime.strptime(b''.join(self.__ds['REFERENCE_DATE_TIME']).decode('utf-8'), '%Y%m%d%H%M%S')
            self.__timestamp = ref_time + self.__ds['JULD'][:].data.astype('timedelta64[D]').astype('O')
        return self.__timestamp


    @property
    def gps(self) -> np.ndarray :
        """
        Returns latitude, longitude of profile in np.array
        """
        if self.__gps is None :
            self.__gps = np.column_stack((self.__ds['LATITUDE'][:].filled(np.nan),
                                          self.__ds['LONGITUDE'][:].filled(np.nan)))
        return self.__gps

    def read(self, var : str | list[str], profiles : int | list[int]) :
        latitude, longitude = self.gps[profiles]
        timestamp = self.timestamp[profiles]
        pressure = self.__ds['PRES_ADJUSTED'][profiles].filled(np.nan)
        if type(var) is str :
            data = self.__ds[var][profiles].filled(np.nan)
            return {'TIMESTAMP': timestamp, 'LATITUDE': latitude, 'LONGITUDE': longitude,
                    'PRES': pressure, var : data}
        data = {'TIMESTAMP': timestamp, 'LATITUDE': latitude, 'LONGITUDE': longitude,
                'PRES': pressure}
        for _var in var :
            data = {**data, **{_var : self.__ds[_var][profiles].filled(np.nan)}}
        return data