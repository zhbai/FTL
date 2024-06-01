import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from scipy.interpolate import splrep, splev
import warnings
warnings.filterwarnings('ignore')

class Encoder(nn.Module):
    
    def __init__(self, encoded_z_dim, fc2_in_dim):
        super().__init__()
        
        ## convolutional section
        self.encoder_cnn = nn.Sequential(
            # 1st convolutional layer
            nn.Conv2d(1, 8, 3, stride=3, padding=1), # 8, (101-3+2+3)/3 = 34, (181-3+2+3)/3 = 61
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # 2nd convolutional layer 
            nn.Conv2d(8, 16, 3, stride=3, padding=1), # 16, (34-3+2+3)/3 = 12, (61-3+2+3)/3 = 21   
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 3rd convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0), # 32, (12-3+2)/2 = 5, (21-3+2)/2 = 10
            nn.ReLU(True)
        )
        ## flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ## linear section
        self.encoder_lin = nn.Sequential(
            # 1st linear layer
            nn.Linear(5 * 10 * 32, fc2_in_dim),  
            # hyperbolic tangent
            nn.Tanh(),
            # 2nd linear layer
            nn.Linear(fc2_in_dim, encoded_z_dim)
        )
        
    def forward(self, x):
        # encoder conv layers 
        x = self.encoder_cnn(x)
        # flatten
        x = self.flatten(x)
        # encoder linear layers
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_z_dim, fc2_in_dim):
        super().__init__()
        
        self.decoder_lin = nn.Sequential(
            # 1st linear layer
            nn.Linear(encoded_z_dim, fc2_in_dim),
            # hyperbolic tangent
            nn.Tanh(),
            # 2nd linear layer
            nn.Linear(fc2_in_dim, 5 * 10 * 32),
            nn.ReLU(True)
        )
        
        # unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 5, 10))

        self.decoder_conv = nn.Sequential(
            # 2D transposed convolution operator 
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding= (1, 0)), # 5*2-2+3+1 = 12, 10*2-2+3 = 21  
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 2nd transposed convolution operator 
            nn.ConvTranspose2d(16, 8, 3, stride=3, padding=1), # 12*3-3-2+3 = 34, 21*3-3-2+3 = 61 
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # 3rd transposed convolution operator 
            nn.ConvTranspose2d(8, 1, 3, stride=3, padding=1, output_padding= (1, 0)) # 34*3-3-2+3+1= 101, 61*3-3-2+3 = 181
        )
        
    def forward(self, x):
        # decoder linear layers
        x = self.decoder_lin(x)
        # unflatten
        x = self.unflatten(x)
        # transposed convolutions
        x = self.decoder_conv(x)
        return x
    
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    
    encoder.train()
    decoder.train()
    train_loss = []
    # iterate the dataloader
    for mini_batch in dataloader: 
        mini_batch = mini_batch.to(device)
        encoded_dsnap = encoder(mini_batch)
        decoded_dsnap = decoder(encoded_dsnap)
        loss = loss_fn(decoded_dsnap, mini_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        
    return np.mean(train_loss)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):

    encoder.eval()
    decoder.eval()
    test_loss = []
    with torch.no_grad(): 
        for mini_batch in dataloader:        
            mini_batch = mini_batch.to(device)
            encoded_dsnap = encoder(mini_batch)
            decoded_dsnap = decoder(encoded_dsnap)
            # global loss
            test_loss.append(loss_fn(decoded_dsnap, mini_batch).detach().cpu().numpy())
            
    return np.mean(test_loss)
    
def load_trained_network(p_encoder, p_decoder, device):
    encoder = torch.load(p_encoder, map_location=device)
    decoder = torch.load(p_decoder, map_location=device)
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = True
    return encoder.to(device), decoder.to(device)
    

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
    
def plot_ae_outputs(data, X, Z, device, encoder, decoder, epoch, n, i_seq):
    cm = 'RdBu_r'
    fig = plt.figure(figsize=(14,8))
    for i in range(n):
        ax = plt.subplot(3,n,i+1)
        dsnap = torch.tensor(data).float()[i_seq[i]].unsqueeze(0).unsqueeze(0).to(device)
        dsnap /= torch.max(torch.abs(dsnap))
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encoder(dsnap)
            rec_dsnap  = decoder(z)
        ax.pcolor(X, Z, dsnap.cpu().squeeze().numpy(), cmap=cm, vmin=-1, vmax=1, shading='auto') #cmap='RdBu'
        ax.plot(X[48,:],Z[48,:],'k-',linewidth=1.5,label='q=1')
        ax.plot(X[78,:],Z[78,:],'k--',linewidth=1.5,label='q=2')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        # if i == n//2:
        #     ax.set_title('Original')
        ax = plt.subplot(3, n, i + 1 + n)
        ax.pcolor(X, Z, rec_dsnap.cpu().squeeze().numpy(), cmap=cm, vmin=-1, vmax=1, shading='auto')
        ax.plot(X[48,:],Z[48,:],'k-',linewidth=1.5,label='q=1')
        ax.plot(X[78,:],Z[78,:],'k--',linewidth=1.5,label='q=2')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        # if i == n//2:
        #     ax.set_title('Reconstructed')
        ax = plt.subplot(3, n, i + 1 + 2*n)
        ax.pcolor(X, Z, (rec_dsnap - dsnap).cpu().squeeze().numpy(), cmap=cm, vmin=-1, vmax=1, shading='auto')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        # if i == n//2:
        #     ax.set_title('Residual')
    plt.tight_layout()
    plt.show()