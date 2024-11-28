import torch
import torch.nn as nn
from KAN.KAN import KAN_layer

class bilineal(nn.Module):
    def __init__(self, 
                 in_features:int,
                 out_features:int,
                 ):
        '''
        '''
        super().__init__()
        self.up = KAN_layer(in_features=in_features,
                            out_features=out_features)
        self.right = KAN_layer(in_features=in_features,
                               out_features = out_features)
    
    def forward(self, x):
        return self.up(x), self.right(x)
    
class biKAN(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 hidden_features,
                 width,
                 height,
                 **kwargs):
        self.height = height
        self.width = width
        super().__init__()
        self.model = nn.ModuleList()
        for i in range(height):
            for j in range(width):
                aux = nn.ModuleList()
                if i == 0:
                    if j == 0:
                        aux.append(bilineal(in_features, hidden_features, **kwargs))
                    elif j == width -1:
                        aux.append(KAN_layer(hidden_features, hidden_features, **kwargs))
                    else: 
                        aux.append(bilineal(hidden_features, hidden_features, **kwargs))
                elif i == height - 1:
                    if j == 0:
                        aux.append(KAN_layer(hidden_features, hidden_features, **kwargs))
                    elif j == width -1:
                        aux.append(KAN_layer(hidden_features*2, hidden_features, **kwargs))
                    else: 
                        aux.append(KAN_layer(hidden_features*2, out_features, **kwargs))
                else:
                    if j == 0:
                        aux.append(bilineal(hidden_features, hidden_features, **kwargs))
                    elif j == width -1:
                        aux.append(KAN_layer(hidden_features*2, hidden_features, **kwargs))
                    else: 
                        aux.append(bilineal(hidden_features*2, hidden_features, **kwargs))
            self.model.append(aux)

    def forward(self, x):
        h = []
        for i in self.height:
            for j in self.width:
                h_glob = []
                if i == 0:
                    if j == self.width - 1:
                        h = self.model[i][j](x)
                    else: 
                        x, h = self.model[i][j](x)
                elif i == self.height - 1:
                    if j == 0:
                        x = h_glob[j]
                    else:
                        x = torch.cat(x, h_glob[j], axis=1)
                    x = self.model[i][j](x)
                else:
                    if j == 0:
                        x, h = self.model[i][j](h_glob[0])
                    else:
                        x = torch.cat(x, h_glob[j], axis=1)
                        if j == self.width - 1:
                            h = self.model[i][j](x)
                        else:
                            x, h = self.model[i][j](x)
                h_glob.append(h)
        return x