import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

    
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

if __name__ == "__main__":
    main()
