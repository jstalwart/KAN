import torch
import torch.nn as nn
from KAN.KAN import KAN_layer
from Splines.BSplineActivation import BSplineActivation

class KAN_MLP_layer(nn.Sequential):
   def __init__(self, 
               in_features:int, 
               out_features:int,
               dropout: float,
               **kwargs):
    super().__init__(nn.Linear(in_features, out_features),
                     nn.Dropout(p=dropout),
                     KAN_layer(out_features, out_features))
    
    

class KAN_MLP(nn.Module):
  def __init__(self, 
               in_features:int, 
               out_features:int, 
               hidden_states:list,
               p = .2,
               **kwargs):
    super().__init__()
    self.in_layer = KAN_MLP_layer(in_features = in_features, out_features = hidden_states[0], dropout=p, **kwargs)
    if len(hidden_states) > 1:
      self.hidden_layers = nn.Sequential(*[KAN_MLP_layer(in_features = hidden_states[i], out_features = hidden_states[i+1], dropout=p, **kwargs) for i in range(len(hidden_states)-1)])
    else:
      self.hidden_layers = False
    self.out_layer = KAN_MLP_layer(in_features = hidden_states[-1], out_features = out_features, dropout=p)
  
  def forward(self, X):
    X = self.in_layer(X)
    if self.hidden_layers:
      X = self.hidden_layers(X)
    X = self.out_layer(X)
    return X