import torch  
from torch import nn  

class Autoencoder(nn.Module):  
    #def __init__(self):
    def __init__(self, input_dim, encoding_dim, hidden_dim1, hidden_dim2):     
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim1),  
            nn.ReLU(),  
            nn.Linear(hidden_dim1, encoding_dim),  
            nn.ReLU() 

        )  
        self.decoder = nn.Sequential(  
            nn.Linear(encoding_dim, hidden_dim2),  
            nn.ReLU(),  
            nn.Linear(hidden_dim2, input_dim),  
            nn.Sigmoid()

        )  
  
    def forward(self, x):  
        x = self.encoder(x)  
        x = self.decoder(x)  
        return x  
    
class Autoencoder1(nn.Module):  
    #def __init__(self):
    def __init__(self, input_dim, encoding_dim, hidden_dim1, hidden_dim2):     
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim1),  
            nn.ReLU(),  
            nn.Linear(hidden_dim1, encoding_dim),  
            nn.ReLU() 
            #nn.Linear(600, 300),  
            #nn.ReLU(),  
            #nn.Linear(300, 100),  
            #nn.ReLU(),  
        )  
        self.decoder = nn.Sequential(  
            nn.Linear(encoding_dim, hidden_dim2),  
            nn.ReLU(),  
            nn.Linear(hidden_dim2, input_dim),  
            nn.Sigmoid()
            #nn.Linear(100, 300),  
            #nn.ReLU(),  
            #nn.Linear(300, 600),  
            #nn.Sigmoid(),  #  
        )  
  
    def forward(self, x):  
        x = self.encoder(x)  
        x = self.decoder(x)  
        return x 
         
class Autoencoder2(nn.Module):  
    #def __init__(self):
    def __init__(self, input_dim, encoding_dim, hidden_dim1, hidden_dim2):     
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim1),  
            nn.ReLU(),  
            nn.Linear(hidden_dim1, encoding_dim),  
            nn.ReLU() 
            #nn.Linear(600, 300),  
            #nn.ReLU(),  
            #nn.Linear(300, 100),  
            #nn.ReLU(),  
        )  
        self.decoder = nn.Sequential(  
            nn.Linear(encoding_dim, hidden_dim2),  
            nn.ReLU(),  
            nn.Linear(hidden_dim2, input_dim),  
            nn.Sigmoid()
            #nn.Linear(100, 300),  
            #nn.ReLU(),  
            #nn.Linear(300, 600),  
            #nn.Sigmoid(),  #  
        )  
  
    def forward(self, x):  
        x = self.encoder(x)  
        x = self.decoder(x)  
        return x      