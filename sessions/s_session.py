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

class S_Session(object):
    def __init__(self, env, model, dataset):
        self.env = env
        self.model = model
        self.loader = DataLoader(dataset, batch_size=self.env.args.batch_size, shuffle=True)
        self.val_inputs = dataset.val_inputs
        self.test_inputs = dataset.test_inputs
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}])
        
        self.optimizer_dm = torch.optim.Adam(
            [{'params': self.model.decision_maker.parameters(), 'lr': self.env.args.dm_lr}]
        )

        self.early_stop = 0
        self.best_epoch = 0
        self.best_dm_epoch = 0
        self.total_epoch = 0
        self.best_overall_accuracy = 0
        self.best_dm_accuracy = 0
        self.cost_dic = dataset.cost_mrp
        self.test_rec_loss = 99999
        self.scaler = StandardScaler()
        
    def dm_train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1

        all_profit_loss = AverageMeter()
        all_late_loss = AverageMeter()
        all_loss = AverageMeter()

        feature_dim = len(feature_list.DataCo_Product + feature_list.DataCo_Order +\
                           feature_list.DataCo_Customer + feature_list.DataCo_Shipping )

        # 分离 self.cost_dic_data 和 self.cost_dic_y
        self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
        self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

        # 创建 faiss 索引，只包含 f1, f2, f3 列
        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
        index.add(self.cost_dic_data)  # 添加成本数据

        profit_sum = 0  # 记录 Gumbel-Softmax 分类为 0 的次数
        profit_count = 0       # 记录 Gumbel-Softmax 分类总数
        time_sum = 0    # 记录 predicted_tokens 第三个结果为 0 的次数
        time_count = 0         # 记录 predicted_tokens 的总数

    # 冻结所有参数，除了 decision_maker 的参数
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decision_maker.parameters():
            param.requires_grad = True

        for input_id in tqdm(self.loader):
            # # cost_feature = input_id[:,-len(feature_list.DataCo_cost):]
            # input_id = input_id[:,:-len(feature_list.DataCo_cost)]
            ori_input = input_id.to(self.env.device)
            input_id = self.scaler.fit_transform(input_id)
            input_id = torch.FloatTensor(input_id).to(self.env.device)

            decision_prob = self.model.decision_process(input_id[:,:feature_dim])
            decision_prob = F.gumbel_softmax(decision_prob, hard=True)
            # 假设 decision_prob 的形状是 [batch_size, 4]，你需要调整它的维度来匹配 embedding
            decision_prob = decision_prob.unsqueeze(2)  # 将 decision_prob 扩展为 [batch_size, 4, 1]

            # 现在将 decision_prob 与 embedding 相乘，embedding 的形状是 [4, embed_dim]
            selected_embedding = torch.sum(decision_prob * self.model.embedding.weight[:4, :], dim=1)
            predicted_tokens = self.model(input_id[:,:feature_dim], selected_embedding, ori_input[:,feature_dim+1:])
            
            # # 损失1：鼓励 Gumbel-Softmax 分类结果为 0
            # target_class = torch.zeros(decision_prob.size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 0
            # profit_loss = F.cross_entropy(decision_prob.squeeze(2), target_class)

            # target_class = torch.zeros(predicted_tokens[1].size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 0
            # late_loss =F.cross_entropy(predicted_tokens[1], target_class)

            decision_indices = decision_prob.argmax(dim=1).squeeze()  # shape 为 [batch_size]

            # for idx in range(decision_prob.size(0)):
            #     decision_index = decision_indices[idx].item()  # 获取当前样本的决策索引

            #     # 准备 faiss 查询向量
            #     query_vector = np.array([
            #         input_id[idx, 0].item(),
            #         input_id[idx, len(feature_list.DataCo_Product)].item(),
            #         decision_index  # 使用决策索引作为查询的第 3 个维度
            #     ], dtype='float32').reshape(1, -1)

            #     # 通过 faiss 查询找到最近的 y 值
            #     _, nearest_idx = index.search(query_vector, 1)  # 查询最近邻
            #     nearest_y = self.cost_dic_y[nearest_idx[0][0]]  # 获取最近邻的 y 值

            #     # 累加 y 值并计数
            #     profit_sum += nearest_y
            #     profit_count += 1               
                    
            
            # 现在将 decision_prob 与 embedding 相乘，embedding 的形状是 [4, embed_dim]
            selected_embedding = torch.sum(decision_prob * self.model.embedding.weight[:4, :], dim=1)
            predicted_tokens = self.model(input_id[:,:feature_dim], selected_embedding, ori_input[:,feature_dim+1:])
        

            # # 统计 predicted_tokens 第三个结果为 1 的比例
            # time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
            # time_count += len(predicted_tokens[-1])
            # 设置 decision_prob 的分类权重，类别0的优先级最高，其次是1和2

            # 原始收益
            profits = torch.tensor([23.119138475584577, 20.681653988361383, 21.211025910849173, 22.07875551481838])

            # 将收益的倒数计算为权重
            weights = 1.0 / profits

            # 归一化，使最大权重为1（可选）
            weights = weights / weights.max()

            # 将权重传入交叉熵损失
            decision_weights = weights.to(self.env.device)
            target_class_decision = torch.zeros(decision_prob.size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 0

            # 损失1：鼓励 decision_prob 更倾向于分类为第0类，带权重的交叉熵
            profit_loss = F.cross_entropy(decision_prob.squeeze(2), target_class_decision, weight=decision_weights)

            # 设置 predicted_tokens[-1] 的分类权重，鼓励分类结果为1
            late_loss_weights = torch.tensor([0.1, 1.0], device=self.env.device)  # 类别1的权重高于类别0
            target_class_predicted = torch.ones(predicted_tokens[-1].size(0), dtype=torch.long).to(self.env.device)  # 目标为类别 1

            # 损失2：鼓励 predicted_tokens[-1] 分类结果为1，带权重的交叉熵
            late_loss = F.cross_entropy(predicted_tokens[-1], target_class_predicted, weight=late_loss_weights)

            # 将损失合并
            loss = 10 * profit_loss + late_loss
            
            # if self.total_epoch < 1000:
            self.optimizer_dm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer_dm.step()
            # else:
            #     self.optimizer_decoder.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            #     self.optimizer_decoder.step()               
            # all_rec_loss.update(rec_loss, 1)
            # all_kl_loss.update(self.env.args.kl_coeff * kl_loss, 1)
            all_profit_loss.update(profit_loss, 1)
            all_late_loss.update(late_loss, 1)
            all_loss.update(loss, 1)

        return all_loss.avg, all_profit_loss.avg, all_late_loss.avg, time.time() - t

    
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1

        all_rec_loss = AverageMeter()
        all_kl_loss = AverageMeter()
        all_classification_loss = AverageMeter()

        # feature_dim = len(feature_list.DataCo_condition)

        feature_dim = len(feature_list.DataCo_Product + feature_list.DataCo_Order +\
                           feature_list.DataCo_Customer + feature_list.DataCo_Shipping )

        label_dim =  len(feature_list.DataCo_label)

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
        for epoch in range(self.env.args.ckpt_start_epoch, self.env.args.epochs):
            classification_loss, train_time = self.train_epoch()
            info('-' * 50)
            info(
                f'TRAIN:epoch = {epoch}/{self.env.args.epochs} classification_loss = {classification_loss:.5f} train_time = {train_time:.2f}')
            if self.env.args.wandb:
                wandb.log({"loss/classification_loss":classification_loss}, epoch)

            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                accuracies, val_time = self.test('val')
                # test_rec_loss, test_time = self.test('test')
                info('-' * 10)
                # info(
                #     f'TRAIN:epoch = {epoch}/{self.env.args.epochs} overall_accuracy = {overall_accuracy:.5f} val_time = {val_time:.2f}')
                for i, accuracy in enumerate(accuracies):
                    info(f"{feature_list.DataCo_label[i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                    if self.env.args.wandb:
                        wandb.log({f"eval/{feature_list.DataCo_label[i]}":accuracy}, epoch)

                if sum(accuracies) / len(accuracies) if accuracies else 0 > self.best_overall_accuracy:
                    self.best_overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                    if self.env.args.wandb:
                        wandb.log({f"eval/best_overall_accuracy":self.best_overall_accuracy}, epoch)
                    info(f"best_overall_accuracy: {self.best_overall_accuracy * 100:.2f}% ")
                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(epoch, 'sim')
                    self.best_epoch = epoch
                    
            if self.early_stop > self.env.args.early_stop:
                break




    def dm_train(self):
        for epoch in range(self.env.args.dm_epochs):
            loss, profit_loss, late_loss, train_time = self.dm_train_epoch()
            info('-' * 50)
            info(
                f'TRAIN:epoch = {epoch}/{self.env.args.dm_epochs} loss = {loss:.5f} profit_loss = {profit_loss:.5f} late_loss = {late_loss:.5f} train_time = {train_time:.2f}')
            if self.env.args.wandb:
                wandb.log({"loss/loss":loss}, self.env.args.epochs + 1 + epoch)
                wandb.log({"loss/profit_loss":profit_loss}, self.env.args.epochs + 1 + epoch)
                wandb.log({"loss/late_loss":late_loss}, self.env.args.epochs + 1 + epoch)

            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += 1
                profit, on_time_ratio, val_time = self.dm_test('val')
                # test_rec_loss, test_time = self.test('test')
                info('-' * 10)
                info(
                    f'high_profit_ratio = {profit:.5f} on_time_ratio = {on_time_ratio:.5f} val_time = {val_time:.2f}')
                if self.env.args.wandb:
                        wandb.log({f"eval/high_profit_ratio":profit, 'eval/on_time_ratio':on_time_ratio}, self.env.args.epochs + 1 + epoch)
                # for i, accuracy in enumerate(accuracies):
                #     info(f"{feature_list.DataCo_label[i]} Accuracy: {accuracy * 100:.2f}% val_time = {val_time:.2f}")
                #     if self.env.args.wandb:
                #         wandb.log({f"eval/{feature_list.DataCo_label[i]}":accuracy}, epoch)

                if on_time_ratio * 100 + profit > self.best_dm_accuracy:
                    self.best_dm_accuracy =  on_time_ratio * 100 + profit
                    if self.env.args.wandb:
                        wandb.log({f"eval/best_on_time_ratio":on_time_ratio}, self.env.args.epochs + 1 + epoch)
                        wandb.log({f"eval/profit_best_on_time_ratio":profit},self.env.args.epochs + 1 + epoch)
                    info(f"best_on_time_ratio: {self.best_dm_accuracy * 100:.2f}% ")
                    info(f"profit_best_on_time_ratio: {profit * 100:.2f}% ")

                    self.early_stop = 0
                    if self.env.args.save:
                        self.save_model(self.env.args.epochs + 1 + epoch, 'dm')
                    self.best_dm_epoch = self.env.args.epochs + 1 + epoch
                    
            if self.early_stop > self.env.args.early_stop:
                break



    def dm_test(self, mode):

        chunk_size = int(self.env.args.batch_size // 1.5)
        self.model.eval()  # 切换到评估模式
        t = time.time()

        if mode == 'val':
            input_id = self.val_inputs
        else:
            input_id = self.test_inputs

        # cost_feature = input_id[:,-len(feature_list.DataCo_cost):]
        # input_id = input_id[:,:-len(feature_list.DataCo_cost)]
        ori_input = input_id.to(self.env.device)
        input_id = self.scaler.transform(input_id)
        input_id = torch.FloatTensor(input_id).to(self.env.device)

        feature_dim = len(feature_list.DataCo_Product + feature_list.DataCo_Order +\
                           feature_list.DataCo_Customer + feature_list.DataCo_Shipping )

        # 分离 self.cost_dic_data 和 self.cost_dic_y
        self.cost_dic_data = self.cost_dic[:, :-1]  # 提取前面的 f1, f2, f3 列
        self.cost_dic_y = self.cost_dic[:, -1]  # 提取最后一列 y

        # 创建 faiss 索引，只包含 f1, f2, f3 列
        index = faiss.IndexFlatL2(self.cost_dic_data.shape[1])  # 使用 L2 距离，维度为 f1, f2, f3 列数
        index.add(self.cost_dic_data)  # 添加成本数据

        profit_sum = 0  # 记录 Gumbel-Softmax 分类为 0 的次数
        profit_count = 0       # 记录 Gumbel-Softmax 分类总数
        time_sum = 0    # 记录 predicted_tokens 第三个结果为 0 的次数
        time_count = 0         # 记录 predicted_tokens 的总数

        with torch.no_grad():
            # 按照 chunk_size 分块评估
            for i in range(0, len(input_id), chunk_size):
                ori_input_chunk = ori_input[i:i + chunk_size]
                input_id_chunk = input_id[i:i + chunk_size]

                decision_prob = self.model.decision_process(input_id_chunk[:,:feature_dim])
                decision_prob = F.gumbel_softmax(decision_prob, hard=True)
                # 假设 decision_prob 的形状是 [batch_size, 4]，你需要调整它的维度来匹配 embedding
                decision_prob = decision_prob.unsqueeze(2)  # 将 decision_prob 扩展为 [batch_size, 4, 1]
                
                decision_indices = decision_prob.argmax(dim=1).squeeze()  # shape 为 [batch_size]

                for idx in range(decision_prob.size(0)):
                    decision_index = decision_indices[idx].item()  # 获取当前样本的决策索引

                    # 准备 faiss 查询向量
                    query_vector = np.array([
                        input_id_chunk[idx, 0].item(),
                        input_id_chunk[idx, len(feature_list.DataCo_Product)].item(),
                        decision_index  # 使用决策索引作为查询的第 3 个维度
                    ], dtype='float32').reshape(1, -1)

                    # 通过 faiss 查询找到最近的 y 值
                    _, nearest_idx = index.search(query_vector, 1)  # 查询最近邻
                    nearest_y = self.cost_dic_y[nearest_idx[0][0]]  # 获取最近邻的 y 值

                    # 累加 y 值并计数
                    profit_sum += nearest_y
                    profit_count += 1               
                        
               
               # 现在将 decision_prob 与 embedding 相乘，embedding 的形状是 [4, embed_dim]
                selected_embedding = torch.sum(decision_prob * self.model.embedding.weight[:4, :], dim=1)
                predicted_tokens = self.model(input_id_chunk[:,:feature_dim], selected_embedding, ori_input_chunk[:,feature_dim+1:])
            

                # 统计 predicted_tokens 第三个结果为 0 的比例
                time_sum += predicted_tokens[-1].argmax(dim=1).sum().item()
                time_count += len(predicted_tokens[-1])

        profit = profit_sum / profit_count if profit_count > 0 else 0
        on_time_ratio = time_sum / time_count if time_count > 0 else 0

        return profit, on_time_ratio, time.time() - t




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

        feature_dim = len(feature_list.DataCo_Product + feature_list.DataCo_Order +\
                           feature_list.DataCo_Customer + feature_list.DataCo_Shipping )
        label_dim =  len(feature_list.DataCo_label)

        with torch.no_grad():
            # 准确度统计变量
            correct_preds = [0] * label_dim  # 每个标签的正确预测数
            total_samples = [0] * label_dim  # 每个标签的样本总数

            # 按照 chunk_size 分块评估
            for i in range(0, len(input_id), chunk_size):
                # 处理每个chunk
                input_chunk = input_id[i:i + chunk_size]
                ori_chunk = ori_input[i:i + chunk_size]

                # 模型的前向传播
                predicted_tokens = self.model(input_chunk[:,:feature_dim], ori_chunk[:,feature_dim].long(), ori_chunk[:,feature_dim+1:])

                # 假设 ori_chunk 的最后 label_dim 列是对应的标签
                class_labels = ori_chunk[:, -label_dim:].long().to(self.env.device)

                # 逐个标签计算分类任务的准确率
                for j in range(label_dim):
                    predicted = torch.argmax(predicted_tokens[j], dim=1)  # 对第 j 个标签的预测
                    correct_preds[j] += (predicted == class_labels[:, j]).sum().item()  # 累加正确预测数
                    total_samples[j] += len(class_labels[:, j])  # 累加样本总数

            # 计算每个标签的准确率
            accuracies = [correct_preds[j] / total_samples[j] for j in range(label_dim)]

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