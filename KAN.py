import torch
import torch.nn as nn
from KAN.Splines.BSplineActivation import BSplineActivation

class KAN_layer(nn.Module):
  def __init__(self, 
               in_features: int,
               out_features: int,
               grid_size: int = 5,
               device: str = None,
               **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    # SiLU:
    self.SiLU = nn.SiLU()
    # Functions
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.edges = nn.ModuleList([BSplineActivation(num_activations=1, grid = grid_size, mode="linear", device=device) for i in range(out_features)])
    # Weights
    self.linears = nn.ModuleList([nn.Linear(in_features = 2, out_features=1) for i in range(out_features)])

  def forward(self, X):
    #X = rearrange(X, "batch (weights edge) -> batch edge weights", weights=2)
    X = X.unsqueeze(2)
    i= 0
    for i in range(len(self.edges)):
      silu = self.SiLU(X[:, i%self.in_features, :])
      spline = self.edges[i](X[:, i%self.in_features, :])
      aux = torch.cat([silu, spline], dim = 1)
      aux = self.linears[i](aux).unsqueeze(2)
      if i == 0:
        Y = aux
      if i > 0:
        if i % self.out_features < Y.shape[1]:
          Y[:, i] += aux
        else:
          Y = torch.cat([Y, aux], dim=1)
      i += 1
    return Y.squeeze(2)

class KAN(nn.Module):
  def __init__(self, 
               in_features:int, 
               out_features:int, 
               hidden_states:list,
               **kwargs):
    super().__init__()
    self.in_layer = KAN_layer(in_features = in_features, out_features = hidden_states[0], **kwargs)
    if len(hidden_states) > 1:
      self.hidden_layers = nn.Sequential(*[KAN_layer(in_features = hidden_states[i], out_features = hidden_states[i+1], **kwargs) for i in range(len(hidden_states)-1)])
    else:
      self.hidden_layers = False
    self.out_layer = KAN_layer(in_features = hidden_states[-1], out_features = out_features)
  
  def forward(self, X):
    X = self.in_layer(X)
    if self.hidden_layers:
      X = self.hidden_layers(X)
    X = self.out_layer(X)
    return X