from torch import nn

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
            nn.ConvTranspose2d(8, 1, 3, stride=3, padding=1, output_padding= (1, 0)) # 34*3-3-2+3+1 = 101, 61*3-3-2+3 = 181
        )
        
    def forward(self, x):
        # decoder linear layers
        x = self.decoder_lin(x)
        # unflatten
        x = self.unflatten(x)
        # transposed convolutions
        x = self.decoder_conv(x)
        return x
