import torch
import torch.nn as nn
from einops import rearrange
from KAN.Splines.BSplineActivation import BSplineActivation

class Spreecher(nn.Module):
    def __init__(self,
                 input_size, 
                 output_size,
                 base=10,
                 r_max=10,
                 grid_size = 5,
                 device = None,
                 **kwargs):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"


        aux = torch.Tensor([1]+[torch.sum(torch.Tensor([base**(-(p-1)*(input_size**r-1)/(input_size-1)) for r in range(1, r_max)])) for p in range(2, input_size+1)])
        self.alfa = aux * torch.ones((output_size, input_size))
        self.alfa = self.alfa.T
        self.alfa = self.alfa.reshape((1,)+self.alfa.shape)
        self.alfa = self.alfa.to(device)

        a = (base*(base-1))**(-1)
        self.qa = torch.Tensor([a * q for q in range(output_size)])
        self.qa = self.qa.to(device)

        self.spline = BSplineActivation(num_activations=input_size, 
                                        grid = grid_size, 
                                        mode="linear", 
                                        device=device)

    def forward(self, 
                X):
        batch, input_size = X.shape
        output_size = self.qa.shape[0]

        X = X.reshape((batch, 1, input_size)) * torch.ones((output_size, input_size))
        X = rearrange(X, "b o i -> b i o")

        qa = torch.ones((batch, input_size, output_size)) * self.qa

        y = torch.einsum("b i o -> b o", self.alfa * self.spline(X+qa))

        return y