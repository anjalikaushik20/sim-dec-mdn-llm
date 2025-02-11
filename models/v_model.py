
import torch.nn as nn
import torch.nn.functional as F
from tools import feature_list
import torch.nn.init as init

class ValueNetwork(nn.Module):
    def __init__(self, env):
        super(ValueNetwork, self).__init__()
        self.env = env
        feature_dim = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                    feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )
        self.fc1 = nn.Linear(feature_dim, 16)
        
        self.fc3 = nn.Linear(16, 4)
        self.to(self.env.device)

    def _initialize_weights(self):
        
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        return self.fc3(x)  
