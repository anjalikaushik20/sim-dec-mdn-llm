import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob

class MDNHead(nn.Module):
    def __init__(self, input_dim, output_dim, gaussians):
        super().__init__()
        self.input_dim = input_dim
        self.gaussians = gaussians
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, (2 * output_dim + 1) * gaussians)
        
    def forward(self, x):
        out = self.linear(x)
        stride = self.gaussians * self.output_dim
        mus = out[..., :stride].view(*out.shape[:-1], self.gaussians, self.output_dim)
        sigmas = out[..., stride:2 * stride].view(*out.shape[:-1], self.gaussians, self.output_dim)
        sigmas = torch.exp(sigmas)  # Ensure positive
        logpi = F.log_softmax(out[..., 2 * stride:2 * stride + self.gaussians], dim=-1)
        return mus, sigmas, logpi
        

# if a wrapper class is needed, then use the following code for the wrapper class
# class MDN(nn.Module):
    # """ Simple MDN model wrapper using MDNHead. """
    # def __init__(self, input_dim, output_dim, gaussians):
    #     super().__init__()
    #     self.mdn_head = MDNHead(input_dim, output_dim, gaussians)

    # def forward(self, x):
    #     return self.mdn_head(x)

    # def compute_loss(self, targets, mus, sigmas, logpis):
    #     """ Compute GMM loss for training. """
    #     return