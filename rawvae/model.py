import torch
import torch.nn as nn
from torch.nn import functional as F

# Kelsey Changes: 
# - [x] No burp bass, maybe try a different activation function on the last dense layer of the decoder? 
# - [ ] Model experiments: No burp bass, since the window is small, capturing bass may be challenging. How about an LSTM layer? 

class VAE(nn.Module):
  def __init__(self, segment_length, n_units, latent_dim): # OG model
  # def __init__(self, segment_length, n_units, latent_dim, lstm_hidden_size): #lstm_layers_model #kelsey addition
    super(VAE, self).__init__()

    self.segment_length = segment_length
    self.n_units = n_units
    self.latent_dim = latent_dim
    # self.lstm_hidden_size = lstm_hidden_size #kelsey addition
    
    # EncoderDecoder
    self.fc1 = nn.Linear(segment_length, n_units)
    # self.lstm_encoder = nn.LSTM(n_units, lstm_hidden_size, batch_first=True) #lstm_layers_model #kelsey addition

    self.fc21 = nn.Linear(n_units, latent_dim) # OG model
    self.fc22 = nn.Linear(n_units, latent_dim) # OG model
    self.fc3 = nn.Linear(latent_dim, n_units) # OG model
    self.fc4 = nn.Linear(n_units, segment_length) # OG model

    # self.fc21 = nn.Linear(lstm_hidden_size, latent_dim) #lstm_layers_model #kelsey addition
    # self.fc22 = nn.Linear(lstm_hidden_size, latent_dim) #lstm_layers_model #kelsey addition
    
    # self.fc3 = nn.Linear(latent_dim, lstm_hidden_size) #lstm_layers_model #kelsey addition
    # self.lstm_decoder = nn.LSTM(lstm_hidden_size, n_units, batch_first=True) #lstm_layers_model #kelsey addition
    # self.fc4 = nn.Linear(n_units, segment_length)

  def encode(self, x):
      h1 = F.relu(self.fc1(x))
      return self.fc21(h1), self.fc22(h1)

      # h1 = F.relu(self.fc1(x)) #lstm_layers_model #kelsey addition
      # h1 = h1.unsqueeze(1) #lstm_layers_model # Add time dimension for LSTM #kelsey addition
      # _, (h_n, _) = self.lstm_encoder(h1) #lstm_layers_model  #kelsey addition
      # h_n = h_n.squeeze(0) #lstm_layers_model  #kelsey addition
      # return self.fc21(h_n), self.fc22(h_n) #lstm_layers_model  #kelsey addition

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std
 
  # ORIGINAL  
  def decode(self, z):
      h3 = F.relu(self.fc3(z))      
      return F.tanh(self.fc4(h3)) 
  
  #ReLU ACTIVATION
  # def decode(self, z):
  #   h3 = F.relu(self.fc3(z))
  #     return F.relu_(self.fc4(h3)) 
  
  # SiLU ACTIVATION
  # def decode(self, z):
      # h3 = F.relu(self.fc3(z))
      # return F.silu(self.fc4(h3)) # Apply SiLU activation to the last layer 

  # LSTM LAYERS
  # def decode(self, z):     
      # h3 = F.relu(self.fc3(z)) 
      # h3 = h3.unsqueeze(1) # Add time dimension for LSTM 
      # h3, _ = self.lstm_decoder(h3) #lstm_layers_model  
      # h3 = h3.squeeze(1) #lstm_layers_model  
      # return F.tanh(self.fc4(h3)) 

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