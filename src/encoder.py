from torch import nn

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
