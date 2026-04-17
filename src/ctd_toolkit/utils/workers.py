from ctd_toolkit.backend.read_meop import ReadMEOP
from ctd_toolkit.backend.read_argo import ReadArgo
import pandas as pd
import numpy as np
import os

def _process_file_reader(args):

    fn, group, var = args

    if group.source.iloc[0] == "MEOP":
        reader = ReadMEOP(fn)
        data = reader.read(var=var, profiles=group.profile.tolist())
        gps = reader.gps
    else:
        reader = ReadArgo(fn)
        data = reader.read(var=var, profiles=group.profile.tolist())
        gps = reader.gps

    pres = data["PRES"]
    vals = data[var]

    return pres, vals, gps

def _process_file_joiner(args):

    fn, group, var, depth_values = args

    if group.source.iloc[0] == "MEOP":
        reader = ReadMEOP(fn).read(var=var, profiles=group.profile.tolist())
    else:
        reader = ReadArgo(fn).read(var=var, profiles=group.profile.tolist())

    pres = reader["PRES"]
    vals = reader[var]

    results = []

    depth_index = pd.Index(depth_values)

    for i in range(len(group)):

        pres_profile = pres[i].ravel()
        val_profile = vals[i].ravel()

        pres_idx = depth_index.get_indexer(pres_profile, method="nearest")

        tmp = pd.DataFrame({
            "depth": pres_idx,
            "value": val_profile
        }).groupby("depth").mean()

        for depth_i, value in tmp["value"].items():

            results.append(
                (
                    int(group.time.iloc[i]),
                    int(depth_i),
                    int(group.lat.iloc[i]),
                    int(group.lon.iloc[i]),
                    float(value)
                )
            )

    return results
