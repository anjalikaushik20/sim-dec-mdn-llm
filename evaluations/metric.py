import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_rec_loss(output, target):
    
    output = output.reshape(-1, output.size(-1))  
    target = target.reshape(-1).long()  
    
    
    loss = F.cross_entropy(output, target)
    return loss

def compute_error_rates(output, target):
    
    predicted_classes = output.argmax(dim=-1)  
    
    
    target = target.transpose(0, 1).long()
    
    
    position_errors = (predicted_classes != target).float().mean(dim=1)  
    
    
    total_errors = (predicted_classes != target).float().mean()  
    
    return total_errors.item(), position_errors

def loss_function(recon_x, x, mean, logvar, prior_mean, prior_logvar):
    
    recon_loss = nn.MSELoss()(recon_x, x)

    
    mean = torch.clamp(mean, min=-10, max=10)
    logvar = torch.clamp(logvar, min=-10, max=10)

    
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp().clamp(min=1e-10))

    return recon_loss, kl_div


def focal_loss(logits, targets, alpha=1, gamma=2):
    
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  
    pt = torch.exp(-ce_loss)  
    
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()  

def weighted_label_smoothing_loss(pred, target, smoothing=0.1, class_weights=None):
    
    num_classes = pred.size(1)  
    
    
    target_one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

    
    target_smooth = target_one_hot * (1 - smoothing) + smoothing / num_classes

    
    log_prob = F.log_softmax(pred, dim=1)

    
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(pred.device)
        weights = class_weights_tensor[target]  
        loss = -weights * (target_smooth * log_prob).sum(dim=1)
    else:
        loss = -(target_smooth * log_prob).sum(dim=1)

    
    return loss.mean()
    
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(pred.device)
        weights = class_weights_tensor[target]  
        loss = -weights * (target_smooth * log_prob).sum(dim=2)
    else:
        loss = -(target_smooth * log_prob).sum(dim=2)

    
    return loss.mean()
