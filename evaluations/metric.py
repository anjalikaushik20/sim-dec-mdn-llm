import torch
import torch.nn.functional as F
import torch.nn as nn

# 计算loss函数
def compute_rec_loss(output, target):
    # 将目标转换为 (seq_length * batch_size) 的形状
    output = output.reshape(-1, output.size(-1))  # (seq_length * batch_size, vocab_size)
    target = target.reshape(-1).long()  # (seq_length * batch_size)
    
    # 使用交叉熵损失函数
    loss = F.cross_entropy(output, target)
    return loss

def compute_error_rates(output, target):
    # 获取每个位置的预测类别（概率最高的索引）
    predicted_classes = output.argmax(dim=-1)  # 取概率最高的索引
    
    # 确保 target 类型为长整型，并调整维度使其与 output 一致
    target = target.transpose(0, 1).long()
    
    # 计算每个位置的错误数，每个样本单独计算
    position_errors = (predicted_classes != target).float().mean(dim=1)  # 每个位置的错误率，按 batch 计算
    
    # 计算整体错误率，平均所有位置和批次的错误率
    total_errors = (predicted_classes != target).float().mean()  # 整体的错误率
    
    return total_errors.item(), position_errors

def loss_function(recon_x, x, mean, logvar, prior_mean, prior_logvar):
    # 重构误差
    recon_loss = nn.MSELoss()(recon_x, x)

    # 对 mean 和 logvar 进行裁剪，避免数值过大或过小
    mean = torch.clamp(mean, min=-10, max=10)
    logvar = torch.clamp(logvar, min=-10, max=10)

    # KL 散度的稳定计算
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp().clamp(min=1e-10))

    return recon_loss, kl_div


def focal_loss(logits, targets, alpha=1, gamma=2):
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # 使用 reduction='none' 保留每个样本的损失
    pt = torch.exp(-ce_loss)  # 计算 pt
    # 计算 Focal Loss
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()  # 返回平均损失

def weighted_label_smoothing_loss(pred, target, smoothing=0.1, class_weights=None):
    # 获取类别数量
    num_classes = pred.size(1)  # 因为 pred 是 [batch_size, num_classes]
    
    # 将 target 转换为 one-hot 表示
    target_one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

    # 引入标签平滑
    target_smooth = target_one_hot * (1 - smoothing) + smoothing / num_classes

    # 计算 log softmax
    log_prob = F.log_softmax(pred, dim=1)

    # 如果有类别权重，应用类别权重
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(pred.device)
        weights = class_weights_tensor[target]  # 根据 target 选择对应的权重
        loss = -weights * (target_smooth * log_prob).sum(dim=1)
    else:
        loss = -(target_smooth * log_prob).sum(dim=1)

    # 返回平均损失
    return loss.mean()
    # 如果有类别权重，应用类别权重
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(pred.device)
        weights = class_weights_tensor[target]  # 根据 target 选择对应的权重
        loss = -weights * (target_smooth * log_prob).sum(dim=2)
    else:
        loss = -(target_smooth * log_prob).sum(dim=2)

    # 返回平均损失
    return loss.mean()