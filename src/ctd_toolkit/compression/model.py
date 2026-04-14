from torch import nn


class Autoencoder(nn.Module):
   def __init__(self, input_size, encoding_dim):
       super(Autoencoder, self).__init__()
       self.encoder = nn.Sequential(
           nn.Linear(input_size, 512),
           nn.ReLU(),
           nn.Linear(512, 128),
           nn.ReLU(),
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Linear(64, 16),
           nn.ReLU(),
           nn.Linear(16, encoding_dim),
           nn.ReLU()
       )
       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, 16),
           nn.ReLU(),
           nn.Linear(16, 64),
           nn.ReLU(),
           nn.Linear(64, 128),
           nn.ReLU(),
           nn.Linear(128, 512),
           nn.ReLU(),
           nn.Linear(512, input_size),
           nn.Sigmoid()
       )

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x