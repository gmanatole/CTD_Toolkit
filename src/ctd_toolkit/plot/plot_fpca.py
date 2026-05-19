import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import xarray as xr
import netCDF4 as nc
from skfda.representation.basis import BSplineBasis
from shapely.geometry import Point, Polygon
import pandas as pd
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]})



class PlotFPCA :

    def __init__(self,
                 K : int,
                 var : list[str],
                 npc : int,
                 grid_path : str,
                 mfpca_path : str,
                 months : list[int] = None) :
        self.K = K
        self.var = var
        self.npc = npc
        self.ds = xr.load_dataset(grid_path)
        self.month_data = pd.to_datetime(self.ds['time']).month
        _mfpca = np.load(mfpca_path, allow_pickle = True)
        self.timestamp = _mfpca['timestamp']
        self.gps = _mfpca['gps']
        self.data = _mfpca['data']
        self.depth = _mfpca['depth']
        self.alpha = _mfpca['alpha']
        self.mfpca = {k: v for k, v in _mfpca.items() if k not in ('timestamp', 'gps', 'depth', 'data')}

        self.depth_min = np.min(self.depth)
        self.depth_max = np.max(self.depth)
        self.grid_points = np.arange(self.depth_min, self.depth_max + 1)
        self.basis = BSplineBasis(domain_range=(self.depth_min, self.depth_max), n_basis=self.K, order=4)
        self.months = months

    @property
    def months(self):
        return self._months

    @months.setter
    def months(self, months=None):
        if not months:
            self._months = list(range(1, 13))
        else:
            self._months = months

    def basis_functions(self):
        phi = self.basis(self.grid_points)
        fig, axes = plt.subplots(1, len(self.var), figsize=(3 * len(self.var), 5))
        if len(self.var) == 1:
            axes = [axes]
        for i in range(len(self.var)):
            ax = axes[i]
            id_range = np.arange(i * self.K, (i + 1) * self.K)
            # Equivalent to: basismat %*% mfpca$vectors[id,1:3]
            Y = np.squeeze(phi).T @ self.mfpca["vectors"][id_range, :3]
            ax.plot(Y[:, 0], self.depth[0])
            ax.plot(Y[:, 1], self.depth[0])
            ax.plot(Y[:, 2], self.depth[0])
            ax.set_xlabel("Basis functions")
            ax.set_title(self.var[i])
            ax.set_ylabel("Depth (m)")
            ax.axvline(0, linestyle="--")
            ax.invert_yaxis()
        plt.tight_layout()

    def basis_function_variation(self, factor = 1, npc = 0):
        phi = np.squeeze(self.basis(self.grid_points)).T
        fig, axes = plt.subplots(1, len(self.var), figsize=(3 * len(self.var), 5))
        if len(self.var) == 1:
            axes = [axes]
        for i in range(len(self.var)):
            ax = axes[i]
            id_range = np.arange(i * self.K, (i + 1) * self.K)
            inter_i = round(100 * np.sum(self.mfpca["vectnotWM"][id_range, npc] ** 2), 2)
            alpha = self.mfpca["alpha_mean"][id_range]
            axe = self.mfpca["axe"][id_range, npc]
            coef_plus = alpha + factor * axe
            coef_minus = alpha - factor * axe
            vp = phi @ coef_plus
            vm = phi @ coef_minus
            mu = phi @ alpha
            ax.plot(mu, self.depth, linewidth=2)
            ax.plot(vp, self.depth, color="red", linewidth=2)
            ax.plot(vm, self.depth, color="blue", linewidth=2)
            ax.set_xlabel(f"{self.var[i]} ({inter_i} \%)")
            ax.set_ylabel("Depth (m)")
            ax.set_title(self.var[i])
            ax.invert_yaxis()
        fig.suptitle(f"PC{npc+1} ({self.mfpca['pval'][npc]} \%)", fontsize=12)
        plt.tight_layout()

    def map_pc(self, pc = 0, time_range = None):
        fig, ax = plt.subplots(
            figsize=(15, 15),
            subplot_kw={'projection': ccrs.PlateCarree()},
            gridspec_kw={'wspace': 0.05})  # ccrs.PlateCarree()

        ax.set_extent([np.min(self.ds['longitude']), np.max(self.ds['longitude']), np.min(self.ds['latitude']), np.max(self.ds['latitude'])])
        ax.coastlines(linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, facecolor='lightgrey')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(np.min(self.ds['longitude']), np.max(self.ds['longitude']), 5))
        gl.ylocator = mticker.FixedLocator(np.arange(np.min(self.ds['latitude']), np.max(self.ds['latitude']), 5))

        if not time_range :
            time_range = (pd.Timestamp(2004,1,1), pd.Timestamp(2025,12,31))
        _data = self.ds.sel(time=slice(*time_range)).where(
            self.ds['time'].dt.month.isin(self.months))
        _data = np.nanmean(_data['PC'][:][:,:,:,pc], axis = 0) #np.isin(self.month_data, self.months)
        _data[~self.points] = np.nan
        sc = ax.imshow(_data[::-1], extent = [np.min(self.ds['longitude']), np.max(self.ds['longitude']), np.min(self.ds['latitude']), np.max(self.ds['latitude'])])
        fig.subplots_adjust(hspace=0.05, top=0.95, right=0.85, bottom=0.05)
        cbar_ax = fig.add_axes([0.9, 0.35, 0.02, 0.3])
        fig.colorbar(sc, cax=cbar_ax, orientation='vertical', label='PC 2')

        fronts = nc.Dataset('C:/Users/m1_gui01/Desktop/postdoc/oceano/fronts.nc')
        ax.scatter(fronts['LonNB'][:].data, fronts['LatNB'][:].data, c='dimgrey', s=2)
        ax.scatter(fronts['LonSAF'][:].data, fronts['LatSAF'][:].data, c='k', s=2)
        ax.scatter(fronts['LonPF'][:].data, fronts['LatPF'][:].data, c='k', s=2)
        ax.scatter(fronts['LonSB'][:].data, fronts['LatSB'][:].data, c='dimgrey', s=2)
        ax.scatter(fronts['LonSACCF'][:].data, fronts['LatSACCF'][:].data, c='k', s=2)

    def latitude_gradient(self, pc = 0) :
        fig, ax = plt.subplots(figsize=(5, 10))
        cmap = plt.get_cmap('viridis')
        month_mask = (np.isin(self.month_data, self.months))
        for year in np.unique(pd.to_datetime(self.timestamp).year):
            time_mask = (pd.to_datetime(self.ds['time']).year == year) & month_mask
            ax.scatter(np.nanmean(self.ds['PC'][:].data[time_mask, :, :, pc], axis=(0, 2)),
                        self.ds['latitude'],
                        color=cmap((year - 2000) / 20),
                        s = 10, alpha = 0.7,
                        label=str(year))
        ax.legend()
        ax.invert_xaxis()

    def define_boundaries(self, front_path : str = '', lower_front = None, upper_front = None) :
        """
        Define boundaries using fronts (from https://www.seanoe.org/data/00486/59800/)
        """

        if lower_front or upper_front:
            fronts = nc.Dataset(front_path)

        front_coords = []
        for i, front in enumerate([upper_front, lower_front]):
            if front:
                lon = fronts[f'Lon{front}'][:].data
                lat = fronts[f'Lat{front}'][:].data
                nan_mask = ~np.isnan(lon) & ~np.isnan(lat)
                lat = lat[nan_mask]
                lon = lon[nan_mask]
                start = np.where(np.sign(lon) == np.sign(-1))[0][0]
                lon = np.concatenate((lon[start:], lon[:start]))
                lat = np.concatenate((lat[start:], lat[:start]))
                if i == 0:
                    front_coords.extend(list(zip(lon, lat)))
                if i == 1:
                    front_coords.extend(list(zip(lon, lat))[::-1])
            else:
                if i == 0:
                    front_coords.extend([(-180, -40), (180, -40)])
                if i == 1:
                    front_coords.extend([(180, -90), (-180, -90)])
        polygon = Polygon(front_coords)

        points = np.full(self.ds['PC'].shape[1:3], False).flatten()
        grid = np.meshgrid(self.ds['longitude'][:].data, self.ds['latitude'][:].data)
        for i, (lon, lat) in enumerate(np.array(grid).reshape(2, -1).T) :
            p = Point(lon, lat)
            if polygon.contains(p):
                points[i] = True
        self.points = points.reshape(self.ds['PC'].shape[1:3])

    def reconstruction(self, var = 'TEMP_ADJUSTED'):
        if var == self.var[0]:
            offset = 0
            idx_min, idx_max = 0, 50
        elif var == self.var[1]:
            offset = len(self.data) // 2
            idx_min, idx_max = 50, 100
        fig, ax = plt.subplots(1, 3)
        alpha_recon = self.alpha.mean(axis=0) + self.mfpca['pc'][:, :self.npc] @ self.mfpca['vectors'][:, :self.npc].T
        phi = self.basis(self.grid_points)
        x_recon = alpha_recon[:, idx_min:idx_max] @ phi.squeeze()
        idx = np.random.randint(0, len(self.data) // 2, 3)
        for j, i in enumerate(idx):
            ax[j].plot(self.data[i + offset], self.depth[i])
            ax[j].plot(x_recon[i], self.depth[i])
            ax[j].invert_yaxis()
        ax[0].set_ylabel('Depth (m)')
        fig.text(0.5, 0, var, ha='center')
        fig.tight_layout()