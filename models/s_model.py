import sys
sys.path.append('/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain')
import random
import torch
import torch.nn as nn
from tools import feature_list

class S_SimDec(nn.Module):
    def __init__(self, env):
        super(S_SimDec, self).__init__()
        self.env = env
        self.c_num = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )
        # 处理连续变量的线性层
        # self.c_transform = nn.Linear(self.c_num, self.env.args.embed_dim)
        self.c_transform4p = nn.Linear(len(feature_list.product_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4o = nn.Linear(len(feature_list.order_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4c = nn.Linear(len(feature_list.customer_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4s = nn.Linear(len(feature_list.shipping_info[self.env.args.dataset]), self.env.args.embed_dim)

        # 池化后的线性层
        self.pooling_fc = nn.Linear(self.c_num, self.env.args.embed_dim)

        # 处理类别变量的嵌入层
        self.embedding = nn.Embedding(5, self.env.args.embed_dim)

        # # 定义多个嵌入表，每个嵌入表的大小和类别数不同
        # self.embeddings = nn.ModuleList([
        #     nn.Embedding(num_classes, self.env.args.embed_dim)  # 每个嵌入表对应一个类别数
        #     for num_classes in self.env.feature_classes
        # ])

        
        self.fc = nn.Linear(self.env.args.embed_dim, self.env.args.embed_dim)

        # LSTM Encoder 和 Decoder 的定义
        self.encoder_lstm = nn.LSTM(input_size=self.env.args.embed_dim, hidden_size=self.env.args.embed_dim, 
                                    num_layers=self.env.args.encoder_num_layers, batch_first=True, bidirectional=True)

        self.decoder_lstm = nn.LSTM(input_size=self.env.args.embed_dim, hidden_size=self.env.args.embed_dim, 
                                    num_layers=self.env.args.decoder_num_layers, batch_first=True)
        
        self.output_layer = nn.Linear(self.env.args.embed_dim, 2)


        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.env.args.embed_dim, num_classes),  # 第一层
            )
            for num_classes in self.env.feature_classes
        ])

        self.decision_maker = nn.Sequential(
                nn.Linear(self.c_num, self.c_num),
                nn.ReLU(),
                nn.Linear(self.c_num, 4)
        )

        self.to(self.env.device)
        
    def forward(self, c_input, shipping_mode, tgt):
        # 编码器部分
        # c_out = self.c_transform(c_input)
        c_out4p = self.c_transform4p(c_input[:, :len(feature_list.product_info[self.env.args.dataset])])
        c_out4o = self.c_transform4o(c_input[:, len(feature_list.product_info[self.env.args.dataset]):len(feature_list.product_info[self.env.args.dataset]) + len(feature_list.order_info[self.env.args.dataset])])
        c_out4c = self.c_transform4c(c_input[:, len(feature_list.product_info[self.env.args.dataset]) + len(feature_list.order_info[self.env.args.dataset]):len(feature_list.product_info[self.env.args.dataset]) + \
                                    len(feature_list.order_info[self.env.args.dataset]) +len(feature_list.customer_info[self.env.args.dataset])])
        c_out4s = self.c_transform4s(c_input[:, -len(feature_list.shipping_info[self.env.args.dataset]):])


        c_out = torch.cat((c_out4p.unsqueeze(1), c_out4o.unsqueeze(1)), dim=1)
        c_out = torch.cat((c_out, c_out4c.unsqueeze(1)), dim=1)
        c_out = torch.cat((c_out, c_out4s.unsqueeze(1)), dim=1)

        # 全局池化
        pooled_features = torch.mean(c_input[:, :self.c_num], dim=0, keepdim=True).repeat(c_input.shape[0], 1) # [batch_size, 1]
        pooled_features = self.pooling_fc(pooled_features)  # [batch_size, embed_dim]

        # 处理 shipping_mode 的不同输入情况
        if len(shipping_mode.shape) == 1:
            # 如果 shipping_mode 是序号，使用 embedding
            sm_embed = self.embedding(shipping_mode).unsqueeze(1)
        else:
            # 如果 shipping_mode 已经是 embedding，直接使用
            sm_embed = shipping_mode.unsqueeze(1)

        combined = torch.cat((c_out,  pooled_features.unsqueeze(1), sm_embed), dim=1) 
        # combined = torch.cat((c_out,  sm_embed), dim=1)
        combined = torch.relu(self.fc(combined))
            
        # 使用双向 LSTM 进行编码
        _, (h_n, c_n) = self.encoder_lstm(combined)

        # 取前向和后向的最后一层隐藏状态进行相加，或者只取前向部分
        h_n_forward = h_n[0:h_n.size(0):2]  # 取偶数索引的层（前向）
        h_n_backward = h_n[1:h_n.size(0):2]  # 取奇数索引的层（后向）

        # 将前向和后向的隐藏状态加在一起，形成解码器的初始隐藏状态
        h_n_combined = h_n_forward + h_n_backward
        c_n_combined = c_n[0:c_n.size(0):2] + c_n[1:c_n.size(0):2]

        decoder_hidden = (h_n_combined, c_n_combined)

        # 自回归生成的起始输入
        batch_size = c_input.shape[0]
        SOS_token = torch.full((batch_size, 1), 1, dtype=torch.long).to(self.env.device)

        generated_tokens = []
        # 逐步生成
        for t in range(tgt.size(-1)):  # 根据目标序列的长度逐步生成
            if t == 0:
                tgt_embed = self.embedding(SOS_token)
            else:
                # 使用上一步生成的 token 作为当前步输入
                tgt_embed = decoder_output
            
            decoder_output, decoder_hidden = self.decoder_lstm(tgt_embed, decoder_hidden)
            predicted_token = self.output_layer[t](decoder_output.squeeze(1))
            
            # 将生成的 token 存储到列表中，而不是堆叠
            generated_tokens.append(predicted_token)

        # 直接返回列表，不使用 stack 拼接
        return generated_tokens
        
      
    def decision_process(self, c_input):
        decision_output = self.decision_maker(c_input)
        return decision_output

