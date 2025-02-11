import sys
sys.path.append('/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain')

import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from tools.logger import info
from tools import feature_list 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

class S_Loader(torch.utils.data.Dataset):
    def __init__(self, env):
        self.env = env
        self.feature_classes = []
        if 'OOD' in self.env.args.dataset:
            self.ori_data_train = pd.read_csv(os.path.join(self.env.DATA_PATH, f'Subset_1.csv'))
            self.ori_data_test = pd.read_csv(os.path.join(self.env.DATA_PATH, f'Subset_2.csv'))
            self.ori_data = pd.concat([self.ori_data_train, self.ori_data_test], axis=0)
        else:
            self.ori_data = pd.read_csv(os.path.join(self.env.DATA_PATH, f'{self.env.args.dataset}.csv'))
        if os.path.exists(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}.csv')):
            data = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}.csv'))
            train_inputs = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_train.csv'))
            val_inputs = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_val.csv'))
            test_inputs = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_test.csv'))
            
            data = data[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
                        feature_list.shipping_info[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]
            
            train_inputs = train_inputs[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
                        feature_list.shipping_info[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]
            
            val_inputs = val_inputs[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
                        feature_list.shipping_info[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]
            
            test_inputs = test_inputs[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
                        feature_list.shipping_info[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]
            cost_m = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_m.csv'))
            cost_mr = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mr.csv'))
            cost_mp= pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mp.csv'))
            cost_mrp = pd.read_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mrp.csv'))

        else:
            cost_m = pd.read_csv(os.path.join(self.env.DATA_PATH, f'average_cost_by_mode.csv'))
            cost_mr = pd.read_csv(os.path.join(self.env.DATA_PATH, f'average_cost_by_mode_region.csv'))
            cost_mp = pd.read_csv(os.path.join(self.env.DATA_PATH, f'average_cost_by_category_mode.csv'))
            cost_mrp = pd.read_csv(os.path.join(self.env.DATA_PATH, f'shipping_cost.csv'))

            extra_data_list = [cost_m, cost_mr, cost_mp,cost_mrp]

            data = self.numerical_features_process(self.ori_data, feature_list.numerical_features[self.env.args.dataset])
            data = self.date_features_process(data, feature_list.date_features[self.env.args.dataset])
            data, extra_data_list = self.categorical_features_process(data, extra_data_list, feature_list.categorical_features[self.env.args.dataset])
            # data = data[feature_list.DataCo_condition + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]

            cost_m = extra_data_list[0]
            cost_mr = extra_data_list[1]
            cost_mp = extra_data_list[2]
            cost_mrp = extra_data_list[3]

            cost_m.to_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_m.csv'),index=None)
            cost_mr.to_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mr.csv'),index=None)
            cost_mp.to_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mp.csv'),index=None)
            cost_mrp.to_csv(os.path.join(self.env.DATA_PATH, f'processed_cost_mrp.csv'),index=None)


            data = data[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
                        feature_list.shipping_info[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]

            # data = data[feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] + feature_list.customer_info[self.env.args.dataset] + \
            #             feature_list.shipping_info[self.env.args.dataset] +  feature_list.info5[self.env.args.dataset] +  feature_list.info6[self.env.args.dataset] +\
            #                   feature_list.info7[self.env.args.dataset] +  feature_list.info8[self.env.args.dataset] + feature_list.decision[self.env.args.dataset] + feature_list.label[self.env.args.dataset]]

            if 'OOD' in self.env.args.dataset:
                train_inputs, test_inputs = train_test_split(data, test_size=0.2, random_state=42)
                val_inputs, test_inputs = train_test_split(test_inputs, test_size=0.5, random_state=42)
            else:
                train_inputs = data[:len(self.ori_data_train)]
                val_inputs = data[len(self.ori_data_train):]
                test_inputs = data[len(self.ori_data_train):]
            
            data.to_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}.csv'),index=None)
            train_inputs.to_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_train.csv'),index=None)
            val_inputs.to_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_val.csv'),index=None)
            test_inputs.to_csv(os.path.join(self.env.DATA_PATH, f'processed_{self.env.args.dataset}_test.csv'),index=None)
            # tokenizer.save_pretrained(os.path.join(self.env.DATA_PATH, f'tokenizer'))
        # 将划分后的数据保存回 self.inputs 作为训练、验证和测试集

        # self.group_features(data)

        self.return_classes_num(data, feature_list.label[self.env.args.dataset])
        self.inputs = torch.FloatTensor(data.values)
        self.train_inputs = torch.FloatTensor(train_inputs.values)
        self.val_inputs = torch.FloatTensor(val_inputs.values)
        self.test_inputs = torch.FloatTensor(test_inputs.values)
        self.cost_m = torch.FloatTensor(cost_m.values)
        self.cost_mr = torch.FloatTensor(cost_mr.values)
        self.cost_mp = torch.FloatTensor(cost_mp.values)
        self.cost_mrp = torch.FloatTensor(cost_mrp.values)


        # 假设 self.cost_mrp 是一个 2D Tensor，其中最后一列需要归一化
        last_column = self.cost_mrp[:, -1]

        # 计算中位数和四分位间距
        median = torch.median(last_column)
        q1 = torch.quantile(last_column, 0.25)
        q3 = torch.quantile(last_column, 0.75)
        iqr = q3 - q1  # 四分位间距

        # 设置 IQR 的下限和上限，用于处理极端值
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 剪裁异常值
        last_column_clipped = torch.clamp(last_column, min=lower_bound, max=upper_bound)

        # 对剪裁后的数据进行归一化（映射到 [0, 1]）
        min_val = last_column_clipped.min()
        max_val = last_column_clipped.max()

        last_column_normalized = (last_column_clipped - min_val) / (max_val - min_val + 1e-8)  # 防止除零

        # 将归一化后的列更新到 self.cost_mrp 中
        self.cost_mrp[:, -1] = last_column_normalized


        # Extract the third and fourth columns
        third_col = self.cost_mrp[:, 2]
        fourth_col = self.cost_mrp[:, 3]


        # Find the maximum value in the third column to create a fixed-size list
        num_groups = int(third_col.max().item()) + 1
        self.avg_profit = [0] * num_groups

        # Calculate the mean of the fourth column for each unique value in the third column
        for value in range(num_groups):
            mask = third_col == value
            if mask.sum() > 0:  # Check if there are rows matching this value
                avg = fourth_col[mask].mean().item()
                self.avg_profit[value] = avg



    def return_classes_num(self, data, categorical_features):
        data = data.copy()
        # 初始化一个字典来存储每个特征的类别数
        # 对每个特征单独计算类别数
        for feature in categorical_features:
            unique_classes = len(np.unique(data[feature].astype(str)))
            self.feature_classes.append(unique_classes)
        # 返回每个特征的类别数
        return None

    def numerical_features_process(self, ori_data, numerical_features):
        data = ori_data.copy()
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        return data

    def categorical_features_process(self, ori_data, extra_data_list, categorical_features):
        # 拷贝数据以避免修改原始数据
        data_ori = ori_data.copy()
        data_extra_list = [data.copy() for data in extra_data_list]  # 拷贝每个 extra_data

        # 对所有类别特征统一编码，且处理extra_data中可能不存在的特征
        for feature in categorical_features:
            encoder = LabelEncoder()

            # 检查特征是否存在于 ori_data 和每个 extra_data 中
            combined_feature_data = pd.Series(dtype=str)
            if feature in data_ori.columns:
                combined_feature_data = pd.concat([combined_feature_data, data_ori[feature].astype(str)])

            # 检查并合并每个 extra_data 中的特征
            for data in data_extra_list:
                if feature in data.columns:
                    combined_feature_data = pd.concat([combined_feature_data, data[feature].astype(str)])

            # 仅当有数据时进行编码
            if not combined_feature_data.empty:
                encoder.fit(combined_feature_data)

                # 在原数据和每个额外数据表中应用相同编码
                if feature in data_ori.columns:
                    data_ori[feature] = encoder.transform(data_ori[feature].astype(str))

                for data in data_extra_list:
                    if feature in data.columns:
                        data[feature] = encoder.transform(data[feature].astype(str))

        # 返回处理后的数据
        return data_ori, data_extra_list
    # def categorical_features_process(self, ori_data, categorical_features):
    #     data = ori_data.copy()

    #     # 对每个特征单独使用 LabelEncoder
    #     for feature in categorical_features:
    #         encoder = LabelEncoder()
    #         # 对每个特征的值进行编码
    #         data[feature] = encoder.fit_transform(data[feature].astype(str))
    #     # 返回处理后的数据和总类别数
    #     return data



    def date_features_process(self, ori_data, date_features):
        data = ori_data.copy()
        for feature in date_features:
            data[feature] = pd.to_datetime(data[feature], errors='coerce')
            data[f'{feature}_year'] = data[feature].dt.year.astype(str) + 'y'
            data[f'{feature}_month'] = data[feature].dt.month.astype(str) + 'h'
            data[f'{feature}_day'] = data[feature].dt.day.astype(str) + 'd'
            data[f'{feature}_hour'] = data[feature].dt.hour.astype(str) + 'h'
            data = data.drop(columns=[feature])
        return data


    def __getitem__(self, index):
        return self.train_inputs[index]

    def __len__(self):
        return len(self.train_inputs)
