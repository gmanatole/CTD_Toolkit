from ctd_toolkit.backend.read_meop import ReadMEOP
from ctd_toolkit.backend.read_argo import ReadArgo
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import os

def _process_file_reader(args):

    fn, group, var = args

    if group.source.iloc[0] == "MEOP":
        reader = ReadMEOP(fn)
        data = reader.read(var=var, profiles=group.profile.tolist())
        gps = reader.gps[group.profile]
        timestamp = reader.timestamp[group.profile]
    else:
        reader = ReadArgo(fn)
        data = reader.read(var=var, profiles=group.profile.tolist())
        gps = reader.gps[group.profile]
        timestamp = reader.timestamp[group.profile]
    pres = data["PRES"]
    vals = np.array([data[_var] for _var in var])
    return pres, vals, gps, timestamp, fn, group.profile.iloc[0], group.source.iloc[0]

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

def _reconstruct_single_profile(i, X, gps, gps_recon, depth_recon, basis, phi_basis_grid, estimsig0, estimsig1, depth_min, K, Wdem, Wdeminv, W_i, h, var0, var1):
    # create phi for varying depth
    phi_recon = np.squeeze(basis(depth_recon[i]))
    # Create reconstruction with weighted gps localization
    weights = multivariate_normal.pdf(gps, mean=gps_recon[i], cov=np.array([[h, 0], [0, h]]))
    norm_weights = weights / (np.sum(weights) + 1e-12)
    # mean / covariance
    alpha_chap = np.sum(X * norm_weights[:, None], axis=0)
    C = X - alpha_chap
    V = (norm_weights[:, None] * C).T @ C
    sigma2_0 = np.sum(np.diag(V[:K, :K] @ W_i))
    sigma2_1 = np.sum(np.diag(V[K:, K:] @ W_i))
    # diagonal M
    M = np.concatenate([
        np.full(K, 1 / sigma2_0),
        np.full(K, 1 / sigma2_1)])
    sqrtM = np.sqrt(M)
    Mdem_diag = np.sqrt(M)
    Mdeminv_diag = 1.0 / Mdem_diag
    # generalized eigen problem
    A = Wdem @ V @ Wdem.T
    VWM = sqrtM[:, None] * A * sqrtM[None, :]
    values, vectors = np.linalg.eigh(VWM)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    vectors = Mdeminv_diag[:, None] * (Wdeminv @ vectors)
    y0 = var0[i]
    y1 = var1[i]
    # covariance blocks
    G_00 = phi_recon.T @ V[:K, :K] @ phi_recon
    G_01 = phi_recon.T @ V[:K, K:] @ phi_recon
    G_10 = phi_recon.T @ V[K:, :K] @ phi_recon
    G_11 = phi_recon.T @ V[K:, K:] @ phi_recon
    # mean
    mu_0 = phi_recon.T @ alpha_chap[:K]
    mu_1 = phi_recon.T @ alpha_chap[K:]
    mu = np.concatenate((mu_0, mu_1))
    condi = np.concatenate((estimsig0[:np.nanmax(depth_recon[i]) - depth_min + 1],
                            estimsig1[:np.nanmax(depth_recon[i]) - depth_min + 1]))
    # build sigmaY
    n = G_00.shape[0]
    sigmaY = np.empty((2 * n, 2 * n))
    sigmaY[:n, :n] = G_00
    sigmaY[:n, n:] = G_01
    sigmaY[n:, :n] = G_10
    sigmaY[n:, n:] = G_11
    # add noise
    sigmaY.flat[::2 * n + 1] += condi
    # solve system
    Y = np.concatenate((y0, y1))
    # principal components
    eps0 = vectors.T[:2 * K, :K] @ phi_recon
    eps1 = vectors.T[:2 * K, K:] @ phi_recon
    eps = np.hstack([eps0, eps1])
    cf = cho_factor(sigmaY, lower=False)
    x = cho_solve(cf, Y - mu)
    pc = values[:2 * K] * (eps @ x)
    # reconstruction
    coefs0 = alpha_chap[:K] + vectors[:K] @ pc
    coefs1 = alpha_chap[K:] + vectors[K:] @ pc
    recon = np.vstack((coefs0 @ phi_basis_grid, coefs1 @ phi_basis_grid))
    return i, recon, pc