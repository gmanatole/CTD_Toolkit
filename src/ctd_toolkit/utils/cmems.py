import copernicusmarine
from datetime import datetime
import numpy as np

def download_cmems(dates, longitude, latitude, variables = ['thetao', 'so'], output = '.') :
    """
    dates : list of dates as UTC Unix timestamps
    longitude : longitude coordinates
    latitude : latitude coordinates
    """
    lon_min, lon_max, lat_min, lat_max = np.nanmin(longitude), np.nanmax(longitude), np.nanmin(latitude), np.nanmax(latitude)
    dataset = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
    if dates[0].timestamp() > datetime(2026,2,1).timestamp() :
        dataset = "cmems_mod_glo_phy_myint_0.083deg_P1M-m"   #For recent year uncontrolled data
    date_start = datetime.strftime(dates[0], '%Y-%m-%dT%H:%M:%S')
    date_end = datetime.strftime(dates[-1], '%Y-%m-%dT%H:%M:%S')
    depth_min, depth_max = 0, 500
    copernicusmarine.subset(
            dataset_id=dataset,
            variables=variables,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=date_start,
            end_datetime=date_end,
            minimum_depth=0,
            maximum_depth=depth_max,
            output_directory = output)