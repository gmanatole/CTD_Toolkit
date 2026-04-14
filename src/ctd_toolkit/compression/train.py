import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import PurePosixPath, PureWindowsPath
import scipy
from torch.utils.data import DataLoader
from ctd_toolkit.backend.read_meop import ReadMEOP
from ctd_toolkit.backend.read_argo import ReadArgo
from ctd_toolkit.compression.model import Autoencoder


torch.set_num_threads(16)

class Train :

    encoding_dim = 10  # Desired number of output dimensions

    def __init__(self,
                 fns : np.ndarray,
                 profiles : np.ndarray,
                 source : np.ndarray,
                 num_epochs: int = 20,
                 data_augmentation: bool = False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fns = np.array([PurePosixPath('/mnt/f/', *PureWindowsPath(elem).parts[1:]) for elem in fns])
        idx = list(range(len(fns)))
        random.shuffle(idx)
        train_idx = idx[:int(len(fns)*0.8)]
        test_idx = idx[int(len(fns)*0.8):int(len(fns))]
        self.input_size = 1992
        print('Loading datasets')
        self.train_loader = DataLoader(CTDDataset(profiles=profiles[train_idx],
                                                  fns=fns[train_idx],
                                                  source=source[train_idx],
                                                  data_augmentation = data_augmentation),
                                       batch_size=64, shuffle=True, num_workers = 0, pin_memory=True)
        self.test_loader = DataLoader(CTDDataset(profiles=profiles[test_idx],
                                                 fns=fns[test_idx],
                                                 source=source[test_idx]),
                                      batch_size=64, shuffle=True, num_workers = 0, pin_memory=True)
        self.model = Autoencoder(self.input_size, self.encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir="/mnt/c/Users/m1_gui01/Desktop/postdoc/runs")
        
    def check_profiles(self):
        pass

    def train(self) :

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch in tqdm(self.train_loader):
                batch = batch.to(self.device, non_blocking = True)
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = batch.to(self.device, non_blocking = True)
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                    test_loss += loss.item()

            test_loss /= len(self.test_loader)
            train_loss /= len(self.train_loader)
            np.save("outputs", outputs.detach().cpu().numpy())
            np.save("inputs", batch.cpu().numpy())
            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Test Loss: {test_loss:.4f}")

            # Log scalar losses
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/test", test_loss, epoch)

            # Log first input/output profile as a line plot
            input_profile = batch[0].cpu().numpy()
            output_profile = outputs[0].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(input_profile[:996], label='Input temperature', alpha=0.7)
            ax.plot(output_profile[:996], label='Output temperature', alpha=0.7)
            ax.set_title(f'Temperature reconstruction for epoch {epoch+1}')
            ax.legend()
            self.writer.add_figure('Temperature reconstruction', fig, epoch)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(input_profile[996:], label='Input salinity', alpha=0.7)
            ax.plot(output_profile[996:], label='Output salinity', alpha=0.7)
            ax.set_title(f'Salinity reconstruction for epoch {epoch+1}')
            ax.legend()
            self.writer.add_figure("Salinity reconstruction", fig, epoch)
            plt.close(fig)

        self.writer.close()


class CTDDataset(Dataset):

    def __init__(self, profiles, fns, source, data_augmentation = False):
        self.profiles = profiles
        self.fns = fns
        self.source = source
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        if self.source[idx] == 'MEOP' :
            try :
                data = ReadMEOP(self.fns[idx]).read(var = ['TEMP_ADJUSTED', 'PSAL_ADJUSTED'], profiles = self.profiles[idx])
            except :
                data = ReadMEOP(self.fns[idx]).unformatted_data(var = ['TEMP_ADJUSTED', 'PSAL_ADJUSTED'], profiles = self.profiles[idx])
            data = np.vstack([data[key] for key in ['TEMP_ADJUSTED', 'PSAL_ADJUSTED']])[:, 4:].flatten()
        elif self.source[idx] == 'Argo':
            data = ReadArgo(self.fns[idx]).read(var = ['TEMP_ADJUSTED', 'PSAL_ADJUSTED'], profiles = self.profiles[idx])
            pres = list(range(5,1001,1))
            temp = np.interp(pres, data['PRES'].flatten(), data['TEMP_ADJUSTED'].flatten())
            sal = np.interp(pres, data['PRES'].flatten(), data['PSAL_ADJUSTED'].flatten())
            data = np.vstack([temp, sal]).flatten()
        data[:996] = (data[:996] + 3) / 18
        data[996:] = (data[996:] - 31) / (35.5-31)
        if self.data_augmentation :
            data[:996] = magnitude_warp(data[:996])
            data[:996] = baseline_shift(data[:996])
            data[996:] = magnitude_warp(data[:996])
            data[996:] = baseline_shift(data[:996])
        return torch.nan_to_num(torch.tensor(data, dtype=torch.float32))


if __name__ == '__main__' :
    import pandas as pd
    path = '/mnt/c/Users/m1_gui01/Desktop/postdoc/oceano/dataset_autoencoder.csv'
    df = pd.read_csv(path)
    inst = Train(df.fn.to_numpy(), df.profile.to_numpy(), df.source.to_numpy())
    inst.train()
