import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, logpi, reduce=True, entropy_reg=0.0):
    # batch: [B, D] or [B, 1] -> make [B, 1, D] for broadcast with [B, K, D]
    batch = batch.unsqueeze(-2)  # [B, 1, D]
    normal_dist = Normal(mus, sigmas)  # mus/sigmas: [B, K, D]
    g_log_probs = normal_dist.log_prob(batch)  # [B, K, D]
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)  # [B, K]
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]  # [B, 1]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)  # [B, K]
    probs = torch.sum(g_probs, dim=-1)  # [B]
    log_prob = max_log_probs.squeeze(-1) + torch.log(probs)  # [B]
    nll = -log_prob

    # Entropy regularization on mixture weights
    if entropy_reg > 0.0:
        pi = torch.softmax(logpi, dim=-1)  # convert logπ to π
        entropy = -torch.sum(pi * logpi, dim=-1)  # [B]
        nll = nll - entropy_reg * entropy  # encourage higher entropy

    if reduce:
        return torch.mean(nll)
    return nll

class MDNHead(nn.Module):
    def __init__(self, input_dim, output_dim, gaussians):
        super().__init__()
        self.input_dim = input_dim
        self.gaussians = gaussians  # K
        self.output_dim = output_dim  # D
        self.linear = nn.Linear(input_dim, (2 * output_dim + 1) * gaussians)
        
    def forward(self, x):
        # x: [B, H]
        out = self.linear(x)  # [B, (2D+1)K]
        B = out.size(0)
        K = self.gaussians
        D = self.output_dim

        pi_logits = out[:, :K]                    # [B, K]
        params = out[:, K:]                       # [B, 2*D*K]
        mus, log_sigmas = params.split(D * K, dim=1)  # each [B, D*K]

        # Shape to [B, K, D]
        mus = mus.view(B, D, K).transpose(1, 2)          # [B, K, D]
        sigmas = F.softplus(log_sigmas).view(B, D, K).transpose(1, 2)  # [B, K, D]
        logpi = F.log_softmax(pi_logits, dim=-1)         # [B, K]
        
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