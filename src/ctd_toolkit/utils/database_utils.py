from ctd_toolkit.backend.read_argo import ReadArgo
from ctd_toolkit.backend.read_meop import ReadMEOP
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def get_directory_profiles(directory : str, source : str, resolution = '') :
    """
    Finds all netCDF4 files within a directory and subdirectories.

    Parameters
    ----------
    directory : str
        Absolute path of directory containing netCDF4 files
    source : str {'Argo', 'MEOP'}
        Source of profiles. Handles Argo and MEOP files.
    resolution : str {'', 'hr1', 'fr1'}
        Resolution of MEOP profiles to keep
    Returns
    -------
    fns : list
        list of netCDF4 filenames
    """
    fns = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith(".nc")) & (resolution in file) :
                fns.append(os.path.join(root, file))
            elif (file.endswith(".nc")) & ('hr1' in file) & (~os.path.exists(os.path.join(root, file.replace('hr1', 'fr1')))):
                fns.append(os.path.join(root, file))
    return list(zip([source]*len(fns), fns))

def process_file(args):
    source, fn, depth = args
    if source == 'Argo':
        inst = ReadArgo(fn)
    elif source == 'MEOP':
        inst = ReadMEOP(fn)
    else:
        return None
    try:
        n = len(inst.timestamp)
        timestamps = inst.timestamp
    except:
        n = len(inst.gps[:, 0])
        timestamps = np.full(n, np.nan)

    profile = list(range(n))
    timestamps = pd.to_datetime(timestamps, errors='coerce')
    timestamps = timestamps.astype('datetime64[us]')
    data = {
        'fn': [fn] * n,
        'source': [source] * n,
        'profile': profile,
        'timestamp': timestamps,
        'lat': inst.gps[:, 0].astype(float),
        'lon': inst.gps[:, 1].astype(float)}
    if depth:
        data['maximum_depth'] = inst.maximum_depth
    return pd.DataFrame(data)

def profile_list_parallel(fns, save_path=os.getcwd(), depth=False):

    assert os.path.isdir(save_path)
    save_path = os.path.join(save_path, 'profile_list.parquet')
    args = [(source, fn, depth) for source, fn in fns]

    writer = None
    with Pool(cpu_count()) as pool:
        for df in tqdm(pool.imap(process_file, args), total=len(args)):
            if df is None:
                continue
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)
            writer.write_table(table)
    if writer:
        writer.close()


def profile_list(fns: list[tuple], save_path: str = os.getcwd(), depth = False):
    """
    Creates a PARQUET file for list of all available profiles along with
    timestamp, latitude and longitude
    """
    assert os.path.isdir(save_path), 'Directory does not exist or path points to file'
    save_path = os.path.join(save_path, 'profile_list.parquet')

    writer = None

    for source, fn in tqdm(fns, desc='Iterating through profile list'):

        if source == 'Argo':
            inst = ReadArgo(fn)
        elif source == 'MEOP':
            inst = ReadMEOP(fn)
        else:
            continue
        try:
            n = len(inst.timestamp)
            timestamps = inst.timestamp
        except:
            n = len(inst.gps[:, 0])
            timestamps = np.full(n, np.nan)
        profile = list(range(n))
        fns_list = [fn]*n
        source_list = [source]*n

        timestamps = pd.to_datetime(timestamps, errors='coerce')
        timestamps = timestamps.astype('datetime64[us]')
        if depth :
            df = pd.DataFrame({
                'fn': fns_list,
                'source': source_list,
                'profile': profile,
                'timestamp': pd.to_datetime(timestamps),
                'maximum_depth': inst.maximum_depth,
                'lat': inst.gps[:, 0].astype(float),
                'lon': inst.gps[:, 1].astype(float)
            })
        else :
            df = pd.DataFrame({
                'fn': fns_list,
                'source': source_list,
                'profile': profile,
                'timestamp': pd.to_datetime(timestamps),
                'lat': inst.gps[:, 0].astype(float),
                'lon': inst.gps[:, 1].astype(float)
            })

        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()