from KAN.KAN import KAN_layer
import torch.nn as nn
import torch

class KAN_norm(nn.Module):
  def __init__(self, 
               in_features:int, 
               out_features:int, 
               hidden_states:list,
               **kwargs):
    super().__init__()
    self.in_layer = KAN_layer(in_features = in_features, out_features = hidden_states[0], **kwargs)
    if len(hidden_states) > 1:
      self.hidden_layers = nn.Sequential()
      for i in range(len(hidden_states)-1):
        self.hidden_layers.append(nn.BatchNorm1d(hidden_states[i]))
        self.hidden_layers.append(KAN_layer(in_features = hidden_states[i], out_features = hidden_states[i+1], **kwargs))
    else:
      self.hidden_layers = False
    self.out_layer = nn.Sequential(nn.BatchNorm1d(hidden_states[-1]),
                        KAN_layer(in_features = hidden_states[-1], out_features = out_features))
  
  def forward(self, X):
    X = self.in_layer(X)
    if self.hidden_layers:
      X = self.hidden_layers(X)
    X = self.out_layer(X)
    return X