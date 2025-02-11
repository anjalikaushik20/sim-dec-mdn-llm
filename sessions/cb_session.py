import sys
sys.path.append('/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain')


import os
import time
import random
import faiss
import torch
import wandb
import numpy as np
from torch import autograd
from tqdm import tqdm
from tools.utils import AverageMeter
from tools import feature_list
from evaluations.metric import compute_rec_loss, compute_error_rates, weighted_label_smoothing_loss, loss_function, focal_loss
from torch.utils.data import DataLoader
from tools.logger import info
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class CB_Session(object):
    def __init__(self, env, model, dataset):
        self.env = env
        self.model = model
        self.value_network = None
        self.optimizer_dm = None
        self.loader = DataLoader(dataset, batch_size=self.env.args.batch_size, shuffle=True)
        self.train_inputs = dataset.train_inputs
        self.val_inputs = dataset.val_inputs
        self.test_inputs = dataset.test_inputs
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}], weight_decay=self.env.args.decay_coeff)
        

        self.action_dim = 4
        self.epsilon = 0.1
        # self.action_counts = np.zeros(self.action_dim)   # 每个动作的选择计数
        self.early_stop = 0
        self.best_epoch = 0
        self.best_dm_epoch = 0
        self.total_epoch = 0
        self.best_overall_accuracy = 0
        self.best_acc1, self.best_acc2, self.best_acc3 = 0, 0, 0
        self.best_dm_accuracy = 0
        self.cost_dic = dataset.cost_mrp
        self.avg_profit = dataset.avg_profit
        self.test_rec_loss = 99999
        self.scaler = StandardScaler()
        self.best_p = 0
        self.best_o = 0
        self.best_pmp1 = 0
        self.best_pmp2 = 0
        self.best_pmp3 = 0

        self.min_profit, self.max_profit = float('inf'), float('-inf')
        self.min_on_time, self.max_on_time = float('inf'), float('-inf')

    def init_value_network(self, value_network):
        self.value_network = value_network
        self.optimizer_dm = torch.optim.Adam(
            [
                # {'params': self.model.decision_maker.parameters(), 'lr': self.env.args.dm_lr},
                {'params': self.value_network.parameters(), 'lr': self.env.args.dm_lr}
            ]
            , weight_decay=self.env.args.dm_decay_coeff
        )

    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1

        all_rec_loss = AverageMeter()
        all_kl_loss = AverageMeter()
        all_classification_loss = AverageMeter()

        # feature_dim = len(feature_list.DataCo_condition)

        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )

        label_dim =  len(feature_list.label[self.env.args.dataset])

        for input_id in tqdm(self.loader):
            # cost_feature = input_id[:,-len(feature_list.DataCo_cost):]
            # input_id = input_id[:,:-len(feature_list.DataCo_cost)]
            ori_input = input_id.to(self.env.device)
            input_id = self.scaler.fit_transform(input_id)
            input_id = torch.FloatTensor(input_id).to(self.env.device)


            predicted_tokens = self.model(input_id[:,:feature_dim], ori_input[:,feature_dim].long(), ori_input[:,feature_dim+1:].long())
            total_loss = 0
            for i in range(label_dim):
                classification_loss = torch.nn.CrossEntropyLoss()(predicted_tokens[i], ori_input[:, feature_dim+1 + i].long())
                total_loss += classification_loss

            loss = total_loss / label_dim
            # if self.total_epoch < 1000:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            # else:
            #     self.optimizer_decoder.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            #     self.optimizer_decoder.step()               
            # all_rec_loss.update(rec_loss, 1)
            # all_kl_loss.update(self.env.args.kl_coeff * kl_loss, 1)
            all_classification_loss.update(loss, 1)

        return all_classification_loss.avg, time.time() - t
 


    def train(self):
        self.early_stop = 0
        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.epochs):
            classification_loss, train_time = self.train_epoch()
            info('-' * 50)
            info(
                f'TRAIN:epoch = {epoch}/{self.env.args.epochs} classification_loss = {classification_loss:.5f} train_time = {train_time:.2f}')
            if self.env.args.wandb:
                wandb.log({"loss/classification_loss":classification_loss}, epoch)
            self.test('val')
            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                accuracies, val_time = self.test('val')
                # test_rec_loss, test_time = self.test('test')
                info('-' * 10)
                # info(
                #     f'TRAIN:epoch = {epoch}/{self.env.args.epochs} overall_accuracy = {overall_accuracy:.5f} val_time = {val_time:.2f}')
                for i, accuracy in enumerate(accuracies):
                    info(f"{feature_list.label[self.env.args.dataset][i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                    if self.env.args.wandb:
                        wandb.log({f"eval/{feature_list.label[self.env.args.dataset][i]}":accuracy}, epoch)

                if (sum(accuracies) / len(accuracies) if accuracies else 0) > self.best_overall_accuracy:
                    info('-' * 10)
                    self.best_overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                    self.best_acc1 = accuracies[0]
                    self.best_acc2 = accuracies[1]
                    self.best_acc3 = accuracies[2]
                    if self.env.args.wandb:
                        wandb.log({f"eval/best_overall_accuracy":self.best_overall_accuracy}, epoch)
                    info(f"best_overall_accuracy: {self.best_overall_accuracy * 100:.2f}% ")
                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(epoch, 'sim')
                    self.best_epoch = epoch
                    
            if self.early_stop > self.env.args.early_stop:
                break


        
    def dm_train_epoch(self):
        t = time.time()
        self.model.train()
        self.value_network.train()
        self.total_epoch += 1

        all_mi_loss = AverageMeter()

        all_ma_loss = AverageMeter()
        all_profit = AverageMeter()
        all_on_time = AverageMeter()

        all_profit_loss = AverageMeter()
        all_late_loss = AverageMeter()
        all_loss = AverageMeter()

        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )

        # 分离 self.cost_dic_data 和 self.cost_dic_y
        self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
        self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

        # 创建 faiss 索引，只包含 f1, f2, f3 列
        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
        index.add(self.cost_dic_data)  # 添加成本数据

        for param in self.model.parameters():
            param.requires_grad = False

        for input_id in tqdm(self.loader):

            ori_input = input_id.to(self.env.device)
            input_id = self.scaler.fit_transform(input_id)
            input_id = torch.FloatTensor(input_id).to(self.env.device)
            state = input_id[:, :feature_dim]


            # 批次选择动作
            # with torch.no_grad():
            decision_prob_value = F.softmax(self.value_network(state), dim=1)  # batch x action_dim
            decision_prob = F.gumbel_softmax(decision_prob_value, tau=1, hard=True)
            # action = decision_prob.argmax(dim=1).squeeze() 


            # 5. 根据 `decision_prob` 选择动作的嵌入
            selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)
            predicted_tokens = self.model(input_id[:,:feature_dim], selected_embedding, ori_input[:,feature_dim+1:])
        
            # 计算类别权重，基于利润值（与利润成正比）
            profits = torch.tensor(self.avg_profit)
            weights = profits / profits.max()  # 归一化到 [0, 1] 区间

            # 设置类别目标，选择利润最高的类别为目标
            target_class_decision = torch.argmax(profits).expand(decision_prob.size(0)).to(self.env.device)

            # 类别权重传递给 cross_entropy
            decision_weights = weights.to(self.env.device)

            # 损失1：鼓励 decision_prob 更倾向于高利润类别
            profit_loss = F.cross_entropy(decision_prob, target_class_decision, weight=decision_weights)

            # 设置 predicted_tokens[-1] 的分类权重，鼓励分类结果为1
            # late_loss_weights = torch.tensor([0.1, 1.0], device=self.env.device)  # 类别1的权重高于类别0
            target_class_predicted = torch.ones(predicted_tokens[-1].size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 1

            # 损失2：鼓励 predicted_tokens[-1] 分类结果为1，带权重的交叉熵
            late_loss = F.cross_entropy(predicted_tokens[-1], target_class_predicted)

            # 合并损失
            mi_loss = self.env.args.mip_coeff * profit_loss + self.env.args.mil_coeff * late_loss


            # # =============== v1 =====================
            # # 原始收益
            # profits = torch.tensor(self.avg_profit)

            # # 将收益的倒数计算为权重
            # weights = 1.0 / profits

            # # 归一化，使最大权重为1（可选）
            # weights = weights / weights.max()

            # # 将权重传入交叉熵损失
            # decision_weights = weights.to(self.env.device)
            # target_class_decision = torch.zeros(decision_prob.size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 0

            # # 损失1：鼓励 decision_prob 更倾向于分类为第0类，带权重的交叉熵
            # profit_loss = F.cross_entropy(decision_prob, target_class_decision, weight=decision_weights)

            # # 设置 predicted_tokens[-1] 的分类权重，鼓励分类结果为1
            # late_loss_weights = torch.tensor([0.1, 1.0], device=self.env.device)  # 类别1的权重高于类别0
            # target_class_predicted = torch.ones(predicted_tokens[-1].size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 1

            # # 损失2：鼓励 predicted_tokens[-1] 分类结果为1，带权重的交叉熵
            # # late_loss = F.cross_entropy(predicted_tokens[-1], target_class_predicted, weight=late_loss_weights)
            # late_loss = F.cross_entropy(predicted_tokens[-1], target_class_predicted)

            # # 将损失合并
            # # print(profit_loss, late_loss)
            # mi_loss = self.env.args.p_coeff * profit_loss + late_loss
            # # info(f'profit_loss {profit_loss}, late_loss {late_loss}')
            # ============================================

            
            all_profit_loss.update(self.env.args.mip_coeff * profit_loss)
            all_late_loss.update(self.env.args.mil_coeff * late_loss)

            action = decision_prob.argmax(dim=1).squeeze() 
            # info(f'{(action == 0).sum()}')
            # info(f'{(action == 1).sum()}')
            # info(f'{(action == 2).sum()}')
            # info(f'{(action == 3).sum()}')

            query_vectors = np.array([
                [
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][0]].cpu().item(),
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][1]].cpu().item(),
                    action[i].item()
                ]
                for i in range(len(state))
            ], dtype='float32')

            _, nearest_indices = index.search(query_vectors, 1)  # 批量查询最近邻
            nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()  # 最近邻样本

            # -------------------
            # 初始化存储每个动作的 profit 和 on_time 统计
            action_profit_sum = torch.zeros(self.action_dim, device=self.env.device)  # 每个动作的利润总和
            action_profit_count = torch.zeros(self.action_dim, device=self.env.device)  # 每个动作的计数
            action_on_time_sum = torch.zeros(self.action_dim, device=self.env.device)  # 每个动作的准时交付总和
            action_on_time_count = torch.zeros(self.action_dim, device=self.env.device)  # 每个动作的计数

            # 比较 query_vectors 和 nearest_samples 是否相等（向量化）
            query_vectors_tensor = torch.tensor(query_vectors, device=self.env.device)  # 转为 Tensor
            nearest_samples_tensor = torch.tensor(nearest_samples, device=self.env.device)

            # 判断每个样本的 query 和 nearest 是否相等
            matches = torch.all(query_vectors_tensor == nearest_samples_tensor, dim=1)  # 形状为 [batch_size]

            # 如果匹配，使用 cost_dic_y，否则根据 action 索引 profit
            selected_y = torch.where(
                matches,
                self.cost_dic_y[nearest_indices.flatten()].to(self.env.device),
                torch.tensor(self.avg_profit, device=self.env.device)[action]
            )

            # 分别统计每个动作的 profit_sum 和 count
            one_hot_action = F.one_hot(action, num_classes=self.action_dim).float()  # 转为 one-hot 编码
            action_profit_sum = torch.matmul(one_hot_action.T, selected_y.unsqueeze(1)).squeeze(1)  # 按动作分类累计 profit
            action_profit_count = one_hot_action.sum(dim=0)  # 每个动作的样本计数

            # 按动作统计 on-time 交付情况
            on_time = predicted_tokens[-1].argmax(dim=1)  # 假设 argmax 得到准时交付（1 表示准时）
            action_on_time_sum = torch.matmul(one_hot_action.T, on_time.unsqueeze(1).float()).squeeze(1)  # 按动作累计 on-time
            action_on_time_count = action_profit_count  # 因为 on-time 的统计基于相同的 action

            # 计算每个动作的平均 profit 和平均 on-time
            avg_profit_per_action = action_profit_sum / (action_profit_count + 1e-8)  # 避免除零
            avg_on_time_per_action = action_on_time_sum / (action_on_time_count + 1e-8)  # 避免除零

            # 计算每个动作的综合奖励
            reward_per_action =  avg_profit_per_action + self.env.args.otr_reward_coeff * avg_on_time_per_action  # [action_dim]

            # # 将动作对应的 reward 分配到每个样本
            # reward = reward_per_action[action]  # 根据 action 为每个样本分配 reward

            # # 打印或返回结果
            # print("Reward per action:", reward_per_action)
            # print("Reward for each sample:", reward)


            # profit_sum, profit_count, on_time_sum, on_time_count = 0, 0, 0, 0



            # # 假设 decision_prob 是一个形状为 [batch_size, num_actions] 的张量
            # # decision_prob_shape = decision_prob.shape

            # # # 随机生成 action，取值范围是 [0, num_actions - 1]
            # # batch_size = decision_prob_shape[0]
            # # num_actions = decision_prob_shape[1]
            # # action = torch.randint(0, num_actions, (batch_size,), device=decision_prob.device)

            # action = decision_prob.argmax(dim=1).squeeze() 
            # query_vectors = np.array([
            #     [
            #         ori_input[i, 0].cpu().item(),
            #         ori_input[i, len(feature_list.product_info[self.env.args.dataset])].cpu().item(),
            #         action[i].item()
            #     ]
            #     for i in range(len(state))
            # ], dtype='float32')

            # _, nearest_indices = index.search(query_vectors, 1)  # 批量查询最近邻


            # nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()  # 最近邻样本

            # for idx, (query, nearest) in enumerate(zip(query_vectors, nearest_samples)):
            #     if np.array_equal(query, nearest):
            #         selected_y = self.cost_dic_y[nearest_indices[idx, 0]]
            #     else:
            #         selected_y = torch.tensor(feature_list.profit[self.env.args.dataset])[action[idx].cpu()]

            #     profit_sum += selected_y
            #     profit_count += 1
        
                    
            
            # # 计算批次奖励
            # avg_profit = profit_sum / profit_count if profit_count > 0 else 0
            # on_time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            # on_time_count += len(predicted_tokens[-1])
            # avg_on_time = on_time_sum / on_time_count if on_time_count > 0 else 0
            
            
            # # self.min_profit = min(self.min_profit, avg_profit)
            # # self.max_profit = max(self.max_profit, avg_profit)
            # # self.min_on_time = min(self.min_on_time, avg_on_time)
            # # self.max_on_time = max(self.max_on_time, avg_on_time)            
            
            # # # 分别归一化 avg_profit 和 avg_on_time
            # # if self.max_profit > self.min_profit:
            # #     normalized_profit = (avg_profit - self.min_profit) / ((self.max_profit - self.min_profit) + 1e-8)
            # # else:
            # #     normalized_profit = 0.5  # 如果所有 profit 相同，设为中间值 0.5

            # # if self.max_on_time > self.min_on_time:
            # #     normalized_on_time = (avg_on_time - self.min_on_time) / ((self.max_on_time - self.min_on_time) + 1e-8)
            # # else:
            # #     normalized_on_time = 0.5  # 如果所有 on_time 相同，设为中间值 0.5

            # # 合并归一化后的两个指标为总奖励
            # # reward = normalized_profit + normalized_on_time
            # reward = avg_profit + avg_on_time

            # # 更新动作的收益记录
            # self.action_counts[action] += 1
            # info(action)
            # 更新网络：计算损失并更新价值估计网络
            self.optimizer_dm.zero_grad()

            # 初始化 target_reward
            if not hasattr(self, 'smoothed_reward'):
                self.smoothed_reward = reward_per_action

            # info(self.smoothed_reward[action])
            # info(reward)
            batch_size = state.shape[0]

            # 找到需要更新的索引
            # info(reward_per_action)
            # info(self.smoothed_reward)
            # 初始状态直接采纳
            self.smoothed_reward =  self.env.args.reward_smoothing_factor * reward_per_action + \
                (1 - self.env.args.reward_smoothing_factor) * self.smoothed_reward

            

            
            # target_reward[action] = reward  # 仅更新选择的动作的收益
            predicted_rewards = self.value_network(state).mean(dim=0)
            # loss = self.env.args.v_coeff * F.mse_loss(predicted_rewards, target_reward)

            ma_loss = F.mse_loss(predicted_rewards, self.smoothed_reward)
            loss = self.env.args.mi_coeff * mi_loss + self.env.args.ma_coeff * ma_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_dm.step()

            # all_profit.update(avg_profit, profit_count)
            # all_on_time.update(avg_on_time, on_time_count)

            all_mi_loss.update(self.env.args.mi_coeff * mi_loss)
            all_ma_loss.update(self.env.args.ma_coeff * ma_loss)
            all_loss.update(loss, len(input_id))
           

        return all_loss.avg, all_profit_loss.avg, all_late_loss.avg, all_profit.avg, all_on_time.avg, all_mi_loss.avg, all_ma_loss.avg, time.time() - t

    

    def dm_train(self):
        self.early_stop = 0

        # profit, on_time_ratio, profit_min_percent, val_time = self.dm_test('ori')
        # info(f'ori_profit_min_percent_10 = {profit_min_percent[0.1]:.5f} ori_profit_min_percent_20 = {profit_min_percent[0.2]:.5f} ori_profit_min_percent_30 = {profit_min_percent[0.3]:.5f}')
        # info(
        #     f'ori_avg_profit = {profit:.5f} ori_on_time_ratio = {on_time_ratio:.5f} overall = {profit+on_time_ratio:.5f} val_time = {val_time:.2f}')
        # if self.env.args.wandb:
        #         wandb.log({f"eval/avg_profit":profit, 'eval/on_time_ratio':on_time_ratio}, self.env.args.epochs)
        #         wandb.log({f"eval/profit_min_percent_10":profit_min_percent[0.1], \
        #                     'eval/profit_min_percent_20':profit_min_percent[0.2], \
        #                     'eval/profit_min_percent_30':profit_min_percent[0.3]}, self.env.args.epochs)

        for epoch in range(self.env.args.dm_epochs):

            loss, profit_loss, late_loss, profit_r, on_time_r, mi_loss, ma_loss, train_time = self.dm_train_epoch()
            info('-' * 50)
            info(
                f'TRAIN:epoch = {epoch}/{self.env.args.dm_epochs} loss = {loss:.5f} profit_loss = {profit_loss:.5f} late_loss = {late_loss:.5f} train_time = {train_time:.2f}')
            info(
                f'profit = {profit_r:.5f} on_time = {on_time_r:.5f} mi_loss = {mi_loss:.5f} ma_loss = {ma_loss:.5f}')
            if self.env.args.wandb:
                wandb.log({"loss/loss":loss}, self.env.args.epochs + 1 + epoch)

                wandb.log({"loss/profit_loss":profit_loss}, self.env.args.epochs + 1 + epoch)
                wandb.log({"loss/late_loss":late_loss}, self.env.args.epochs + 1 + epoch)

                wandb.log({"loss/profit":profit_r}, self.env.args.epochs + 1 + epoch)
                wandb.log({"loss/on_time":on_time_r}, self.env.args.epochs + 1 + epoch)

                wandb.log({"loss/mi_loss":mi_loss}, self.env.args.epochs + 1 + epoch)
                wandb.log({"loss/ma_loss":ma_loss}, self.env.args.epochs + 1 + epoch)

            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                profit, on_time_ratio, profit_min_percent, val_time = self.dm_test('val')
                # test_rec_loss, test_time = self.test('test')
                info('-' * 10)
                info(
                    f'avg_profit = {profit:.5f} on_time_ratio = {on_time_ratio:.5f} overall = {profit+on_time_ratio:.5f} val_time = {val_time:.2f}')
                info(f'profit_min_percent_10 = {profit_min_percent[0.1]:.5f} profit_min_percent_20 = {profit_min_percent[0.2]:.5f} profit_min_percent_30 = {profit_min_percent[0.3]:.5f}')
    
                
                if self.env.args.wandb:
                        wandb.log({f"eval/avg_profit":profit, 'eval/on_time_ratio':on_time_ratio}, self.env.args.epochs + 1 + epoch)
                        wandb.log({f"eval/profit_min_percent_10":profit_min_percent[0.1], \
                                   'eval/profit_min_percent_20':profit_min_percent[0.2], \
                                    'eval/profit_min_percent_30':profit_min_percent[0.3]}, self.env.args.epochs + 1 + epoch)

                
                
                # for i, accuracy in enumerate(accuracies):
                #     info(f"{feature_list.label[self.env.args.dataset][i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                #     if self.env.args.wandb:
                #         wandb.log({f"eval/{feature_list.label[self.env.args.dataset][i]}":accuracy}, epoch)

                if on_time_ratio + profit > self.best_dm_accuracy:
                    self.best_dm_accuracy =  on_time_ratio + profit
                    self.best_p = profit
                    self.best_o = on_time_ratio
                    self.best_pmp1 = profit_min_percent[0.1]
                    self.best_pmp2 = profit_min_percent[0.3]
                    self.best_pmp3 = profit_min_percent[0.3]


                    if self.env.args.wandb:
                        wandb.log({f"eval/best_on_time_ratio":on_time_ratio}, self.env.args.epochs + 1 + epoch)
                        wandb.log({f"eval/best_profit":profit},self.env.args.epochs + 1 + epoch)
                        wandb.log({f"eval/best_dm_accuracy":self.best_dm_accuracy},self.env.args.epochs + 1 + epoch)

                    info(f"best_dm_accuracy: {self.best_dm_accuracy:.5f} ")
                    # info(f"profit_best_on_time_ratio: {profit * 100:.2f}% ")

                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(self.env.args.epochs + 1 + epoch, 'dm')
                    self.best_dm_epoch = self.env.args.epochs + 1 + epoch
                    
            if self.early_stop > self.env.args.early_stop:
                break


    # ========================= v1 =========================
    # def dm_test(self, mode):

    #     chunk_size = int(self.env.args.batch_size // 1.2)
    #     self.model.eval()  # 切换到评估模式
    #     self.value_network.eval()
    #     t = time.time()

    #     if mode == 'val' or mode == 'ori':
    #         input_id = self.val_inputs
    #     else:
    #         input_id = self.test_inputs
    #     # input_id = self.train_inputs

    #     # info(len(input_id))
    #     # cost_feature = input_id[:,-len(feature_list.DataCo_cost):]
    #     # input_id = input_id[:,:-len(feature_list.DataCo_cost)]
    #     ori_input = input_id.to(self.env.device)
    #     if mode != 'ori':
    #         input_id = self.scaler.transform(input_id)
    #     input_id = torch.FloatTensor(input_id).to(self.env.device)

    #     feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
    #                        feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )

    #     # 分离 self.cost_dic_data 和 self.cost_dic_y
    #     self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
    #     self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

    #     # 创建 faiss 索引，只包含 f1, f2, f3 列
    #     index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
    #     index.add(self.cost_dic_data)  # 添加成本数据

    #     profit_sum = 0  # 记录 Gumbel-Softmax 分类为 0 的次数
    #     profit_count = 0       # 记录 Gumbel-Softmax 分类总数   
    #     time_sum = 0    # 记录 predicted_tokens 第三个结果为 0 的次数
    #     time_count = 0         # 记录 predicted_tokens 的总数
    #     state = input_id[:,:feature_dim]

    #     with torch.no_grad():
    #         # 按照 chunk_size 分块评估

    #         if mode == 'ori':
    #             decision_indices = input_id[:, feature_dim].long()  # 确保为整数类型
    #             decision_prob = F.one_hot(decision_indices, num_classes=4).float().to(self.env.device)
    #             decision_prob_value = decision_prob
    #         else:
    #             decision_prob_value = F.softmax(self.value_network(state), dim=1)  # batch x action_dim
    #             decision_prob = decision_prob_value
    #             # decision_logits = self.model.decision_process(input_id_chunk[:,:feature_dim])
    #             # decision_logits = torch.clamp(decision_logits, min=-30, max=30)  # 避免数值溢出
    #             # print("decision_logits min:", decision_logits.min().item(), "max:", decision_logits.max().item(), "mean:", decision_logits.mean().item())

    #             # combined_logits = decision_prob_value 
    #             # # combined_logits = torch.clamp(combined_logits, min=-10, max=10)
    #             # decision_prob = F.gumbel_softmax(combined_logits, tau=0.1, hard=True)


    #             # # 假设 decision_prob 的形状是 [batch_size, 4]，你需要调整它的维度来匹配 embedding
    #             # decision_prob = decision_prob.unsqueeze(2)  # 将 decision_prob 扩展为 [batch_size, 4, 1]
                

    #         # # 假设 decision_prob 是一个形状为 [batch_size, num_actions] 的张量
    #         # decision_prob_shape = decision_prob.shape

    #         # # 随机生成 action，取值范围是 [0, num_actions - 1]
    #         # batch_size = decision_prob_shape[0]
    #         # num_actions = decision_prob_shape[1]
    #         # action = torch.randint(0, num_actions, (batch_size,), device=decision_prob.device)
    #         action = decision_prob_value.argmax(dim=1).squeeze()  # shape 为 [batch_size]
    #         query_vectors = np.array([
    #             [
    #                 ori_input[i, 0].cpu().item(),
    #                 ori_input[i, len(feature_list.product_info[self.env.args.dataset])].cpu().item(),
    #                 action[i].item()
    #             ]
    #             for i in range(len(state))
    #         ], dtype='float32')

    #         _, nearest_indices = index.search(query_vectors, 1)  # 批量查询最近邻


    #         nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()  # 最近邻样本

    #         for idx, (query, nearest) in enumerate(zip(query_vectors, nearest_samples)):
    #             if np.array_equal(query, nearest):
    #                 selected_y = self.cost_dic_y[nearest_indices[idx, 0]]
    #             else:
    #                 selected_y = torch.tensor(self.avg_profit)[action[idx].cpu()]

    #             # 累加 y 值并计数
    #             profit_sum += selected_y
    #             profit_count += 1               
                    
            
    #         # 现在将 decision_prob 与 embedding 相乘，embedding 的形状是 [4, embed_dim]
    #         if mode == 'ori':
    #             selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)
    #         else:
    #             selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)
    #         predicted_tokens = self.model(input_id[:,:feature_dim], selected_embedding, ori_input[:,feature_dim+1:])
        

    #         # 统计 predicted_tokens 第三个结果为 0 的比例
    #         time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
    #         time_count += len(predicted_tokens[-1])

    #     profit = profit_sum / profit_count if profit_count > 0 else 0
    #     on_time_ratio = time_sum / time_count if time_count > 0 else 0

    #     return profit, on_time_ratio, time.time() - t

    # ========================= v1 =========================


    def dm_test(self, mode):
        self.model.eval()  # 切换到评估模式
        self.value_network.eval()
        t = time.time()

        if mode == 'val' or mode == 'ori':
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        ori_input = input_id.to(self.env.device)

        if mode != 'ori':
            input_id = self.scaler.transform(input_id)

        input_id = torch.FloatTensor(input_id).to(self.env.device)
        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                        feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset])

        # 分离 self.cost_dic_data 和 self.cost_dic_y
        self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
        self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

        # 创建 faiss 索引，只包含 f1, f2, f3 列
        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
        index.add(self.cost_dic_data)  # 添加成本数据

        profit_sum = 0  # 记录 Gumbel-Softmax 分类为 0 的次数
        profit_count = 0  # 记录 Gumbel-Softmax 分类总数
        time_sum = 0  # 记录 predicted_tokens 第三个结果为 0 的次数
        time_count = 0  # 记录 predicted_tokens 的总数
        local_profits = []  # 存储所有订单的利润

        state = input_id[:, :feature_dim]

        with torch.no_grad():
            # 按照 chunk_size 分块评估
            if mode == 'ori':
                decision_indices = input_id[:, feature_dim].long()  # 确保为整数类型
                decision_prob = F.one_hot(decision_indices, num_classes=4).float().to(self.env.device)
                decision_prob_value = decision_prob
            else:
                value_network_output = self.value_network(state)
                decision_prob_value =(F.softmax(value_network_output, dim=1) == F.softmax(value_network_output, dim=1).max(dim=1, keepdim=True).values).float()  # batch x action_dim
                decision_prob = decision_prob_value

            # 随机生成 action，取值范围是 [0, num_actions - 1]
            action = decision_prob_value.argmax(dim=1).squeeze()  # shape 为 [batch_size]
            # probability = 1
            # action = torch.tensor(np.random.choice([3, 1], size=action.shape, p=[probability, 1 - probability])).to(self.env.device)
            # decision_prob = F.one_hot(action, num_classes=4).float().to(self.env.device)

            query_vectors = np.array([
                [
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][0]].cpu().item(),
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][1]].cpu().item(),
                    action[i].item()
                ]
                for i in range(len(state))
            ], dtype='float32')

            _, nearest_indices = index.search(query_vectors, 1)  # 批量查询最近邻
            nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()  # 最近邻样本

            for idx, (query, nearest) in enumerate(zip(query_vectors, nearest_samples)):
                if np.array_equal(query, nearest):
                    selected_y = self.cost_dic_y[nearest_indices[idx, 0]]
                else:
                    selected_y = torch.tensor(self.avg_profit)[action[idx].cpu()]

                # 累加 y 值并计数
                profit_sum += selected_y
                profit_count += 1
                local_profits.append(selected_y)  # 将每个订单的利润加入列表

            # 计算统计指标
            sorted_profits = np.sort(local_profits)  # 将利润从小到大排序

            # 计算最低10%、20%、30%的利润的最小值
            thresholds = [0.1, 0.2, 0.3]
            profit_min_percent = {}
            for threshold in thresholds:
                idx = int(threshold * len(sorted_profits))
                profit_min_percent[threshold] = sorted_profits[idx]

            selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)

            predicted_tokens = self.model(input_id[:, :feature_dim], selected_embedding, ori_input[:, feature_dim + 1:])

            # 统计 predicted_tokens 第三个结果为 0 的比例
            time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count += len(predicted_tokens[-1])

        profit = profit_sum / profit_count if profit_count > 0 else 0
        on_time_ratio = time_sum / time_count if time_count > 0 else 0

        # 返回各项指标，包括新加入的统计
        return profit, on_time_ratio, profit_min_percent, time.time() - t



    def test(self, mode):

        chunk_size = int(self.env.args.batch_size // 1.5)
        self.model.eval()  # 切换到评估模式
        t = time.time()

        if mode == 'val':
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        # cost_feature = input_id[:,-len(feature_list.DataCo_cost):]
        # input_id = input_id[:,:-len(feature_list.DataCo_cost)]
        correct_preds = 0  # 每个任务的正确预测数
        total_samples = 0

        ori_input = input_id.to(self.env.device)
        input_id = self.scaler.transform(input_id)
        input_id = torch.FloatTensor(input_id).to(self.env.device)

        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )
        label_dim =  len(feature_list.label[self.env.args.dataset])

        ori_label_value_counts = {j: {} for j in range(label_dim)}

        # 初始化标签取值统计器
        label_value_counts = {j: {} for j in range(label_dim)}
        with torch.no_grad():
            # 准确度统计变量
            correct_preds = [0] * label_dim  # 每个标签的正确预测数
            total_samples = [0] * label_dim  # 每个标签的样本总数

            # 按照 chunk_size 分块评估
            for i in range(0, len(input_id), chunk_size):
                # 处理每个chunk
                input_chunk = input_id[i:i + chunk_size]
                ori_chunk = ori_input[i:i + chunk_size]

                # # 更新特征取值统计
                # for idx in range(feature_dim):
                #     for value in ori_chunk[:, idx].cpu().numpy():
                #         if value not in feature_value_counts[idx]:
                #             feature_value_counts[idx][value] = 0
                #         feature_value_counts[idx][value] += 1

                # 更新标签取值统计
                # print(f'===========ori distribution=============')
                for j in range(label_dim):
                    for value in ori_chunk[:, feature_dim + j + 1].cpu().numpy():
                        if value not in ori_label_value_counts[j]:
                            ori_label_value_counts[j][value] = 0
                        ori_label_value_counts[j][value] += 1
                    # print(f'{ori_label_value_counts[j]}')

                # 模型的前向传播
                predicted_tokens = self.model(input_chunk[:,:feature_dim], ori_chunk[:,feature_dim].long(), ori_chunk[:,feature_dim+1:])

                # 假设 ori_chunk 的最后 label_dim 列是对应的标签
                class_labels = ori_chunk[:, -label_dim:].long().to(self.env.device)

                # print(f'===========sim distribution=============')
                # 更新标签取值统计
                # for j in range(label_dim):




                # 逐个标签计算分类任务的准确率
                for j in range(label_dim):
                    predicted = torch.argmax(predicted_tokens[j], dim=1)  # 对第 j 个标签的预测
                    correct_preds[j] += (predicted == class_labels[:, j]).sum().item()  # 累加正确预测数
                    total_samples[j] += len(class_labels[:, j])  # 累加样本总数
                    for value in predicted.cpu().numpy():
                        if value not in label_value_counts[j]:
                            label_value_counts[j][value] = 0
                        label_value_counts[j][value] += 1

        # 计算每个标签的准确率
        accuracies = [correct_preds[j] / total_samples[j] for j in range(label_dim)]
        # print(f'===========ori distribution=============')
        # for j in range(label_dim):
        #     print(f'{ori_label_value_counts[j]}')

        # print(f'===========sim distribution=============')
        # for j in range(label_dim):
        #     print(f'{label_value_counts[j]}')
        return accuracies, time.time() - t



    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch, mode):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.suffix}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if mode == 'sim':
            best_epoch = self.best_epoch
        else:
            best_epoch = self.best_dm_epoch
        if current_epoch != best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.suffix}_epoch{best_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))

    def step(self, state, action):
        """
        预测下一个状态、奖励，并更新环境
        :param state: 当前订单特征
        :param action: 运输模式 (0-3)
        :return: (next_state, reward, done, info)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.env.device)
            action_tensor = torch.LongTensor([action]).to(self.env.device)

            # 使用模型预测 (profit, on_time_ratio)
            predicted_tokens = self.forward(state_tensor, action_tensor, state_tensor)
            profit = predicted_tokens[0, 0].item()
            on_time_ratio = predicted_tokens[0, 1].item()

        # 计算奖励
        reward = profit + self.env.beta * on_time_ratio

        # 更新状态（假设是单步决策问题，next_state=None）
        next_state = None  

        # 检查是否到达数据集终点
        self.env.index += 1
        done = self.env.index >= len(self.env.loader.test_inputs)

        info = {"profit": profit, "on_time": on_time_ratio}
        return next_state, reward, done, info
    
    def step(self):
        self.model.eval()  # 切换到评估模式
        self.value_network.eval()
        t = time.time()
        self.env.index = 0
        input_id = self.train_inputs

        ori_input = input_id.to(self.env.device)


        input_id = torch.FloatTensor(input_id).to(self.env.device)
        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                        feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset])

        # 分离 self.cost_dic_data 和 self.cost_dic_y
        self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
        self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

        # 创建 faiss 索引，只包含 f1, f2, f3 列
        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
        index.add(self.cost_dic_data)  # 添加成本数据

        profit_sum = 0  # 记录 Gumbel-Softmax 分类为 0 的次数
        profit_count = 0  # 记录 Gumbel-Softmax 分类总数
        time_sum = 0  # 记录 predicted_tokens 第三个结果为 0 的次数
        time_count = 0  # 记录 predicted_tokens 的总数
        local_profits = []  # 存储所有订单的利润

        state = input_id[:, :feature_dim]

        with torch.no_grad():
            value_network_output = self.value_network(state)
            decision_prob_value =(F.softmax(value_network_output, dim=1) == F.softmax(value_network_output, dim=1).max(dim=1, keepdim=True).values).float()  # batch x action_dim
            decision_prob = decision_prob_value

            # 随机生成 action，取值范围是 [0, num_actions - 1]
            action = decision_prob_value.argmax(dim=1).squeeze()  # shape 为 [batch_size]
            # probability = 1
            # action = torch.tensor(np.random.choice([3, 1], size=action.shape, p=[probability, 1 - probability])).to(self.env.device)
            # decision_prob = F.one_hot(action, num_classes=4).float().to(self.env.device)

            query_vectors = np.array([
                [
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][0]].cpu().item(),
                    ori_input[i, feature_list.retrieva_index[self.env.args.dataset][1]].cpu().item(),
                    action[i].item()
                ]
                for i in range(len(state))
            ], dtype='float32')

            _, nearest_indices = index.search(query_vectors, 1)  # 批量查询最近邻
            nearest_samples = self.cost_dic_data[nearest_indices.flatten()].cpu().numpy()  # 最近邻样本

            profit, on_time_ratio, reward = [], [], []


            for idx, (query, nearest) in enumerate(zip(query_vectors, nearest_samples)):
                if np.array_equal(query, nearest):
                    selected_y = self.cost_dic_y[nearest_indices[idx, 0]]
                else:
                    selected_y = torch.tensor(self.avg_profit)[action[idx].cpu()]

                profit.append(selected_y)



            selected_embedding = torch.sum(decision_prob.unsqueeze(2) * self.model.embedding.weight[:4, :], dim=1)

            predicted_tokens = self.model(input_id[:, :feature_dim], selected_embedding, ori_input[:, feature_dim + 1:])

            # 统计 predicted_tokens 第三个结果为 0 的比例
            time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            time_count += len(predicted_tokens[-1])


        # profit = profit_sum / profit_count if profit_count > 0 else 0
        # on_time_ratio = time_sum / time_count if time_count > 0 else 0
        # 计算奖励
        on_time_ratio = predicted_tokens[-1].argmax(dim=1).tolist()

        for i in range(len(on_time_ratio)):
            reward.append(on_time_ratio[i] + profit[i])

        # 更新状态（假设是单步决策问题，next_state=None）
        next_state = None

        done = [False * len(on_time_ratio)]
        done[-1] = True

        info = {"profit": profit, "on_time": on_time_ratio}
        return next_state, reward, done, info
    
