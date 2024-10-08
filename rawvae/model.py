import torch
import torch.nn as nn
from torch.nn import functional as F

  """ 
  Kelsey Changes: 
  - [ ] No burp bass, maybe try a different activation function on the last dense layer of the decoder? 
  - [ ] Model experiments: No burp bass, since the window is small, capturing bass may be challenging. How about an LSTM layer? 
  """

class VAE(nn.Module):
  def __init__(self, segment_length, n_units, latent_dim, lstm_hidden_size): #kelsey addition
    super(VAE, self).__init__()

    self.segment_length = segment_length
    self.n_units = n_units
    self.latent_dim = latent_dim
    self.lstm_hidden_size = lstm_hidden_size #kelsey addition
    
    # Encoder
    self.fc1 = nn.Linear(segment_length, n_units)
    self.lstm_encoder = nn.LSTM(n_units, lstm_hidden_size, batch_first=True) #kelsey addition
    self.fc21 = nn.Linear(lstm_hidden_size, latent_dim) #kelsey addition
    self.fc22 = nn.Linear(lstm_hidden_size, latent_dim) #kelsey addition
    
    # Decoder
    self.fc3 = nn.Linear(latent_dim, lstm_hidden_size)
    self.lstm_decoder = nn.LSTM(lstm_hidden_size, n_units, batch_first=True)
    self.fc4 = nn.Linear(n_units, segment_length)

  def encode(self, x):
      h1 = F.relu(self.fc1(x))
      h1 = h1.unsqueeze(1)  # Add time dimension for LSTM #kelsey addition
      _, (h_n, _) = self.lstm_encoder(h1) #kelsey addition
      h_n = h_n.squeeze(0) #kelsey addition
      return self.fc21(h_n), self.fc22(h_n) #kelsey addition

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std

  def decode(self, z):
      h3 = F.relu(self.fc3(z))
      h3 = h3.unsqueeze(1)  # Add time dimension for LSTM #kelsey addition
      h3, _ = self.lstm_decoder(h3) #kelsey addition
      h3 = h3.squeeze(1) #kelsey addition
      return F.tanh(self.fc4(h3))

  def forward(self, x):
      mu, logvar = self.encode(x.view(-1, self.segment_length))
      z = self.reparameterize(mu, logvar)
      return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_beta, segment_length):
  recon_loss = F.mse_loss(recon_x, x.view(-1, segment_length))

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

  return recon_loss + ( kl_beta * KLD)