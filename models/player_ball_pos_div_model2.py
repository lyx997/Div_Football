import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(Model, self).__init__()
        self.device=None
        if device:
            self.device = device

        self.arg_dict = arg_dict

        self.fc_div = nn.Linear(arg_dict["feature_dims"]["player_div"],arg_dict["div_lstm_size"])  
        
        self.norm_div = nn.LayerNorm(arg_dict["div_lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["div_lstm_size"], arg_dict["div_lstm_size"])

        self.fc_pi_z = nn.Linear(arg_dict["div_lstm_size"], arg_dict["div_num"])
        self.norm_pi_z = nn.LayerNorm(arg_dict["div_num"])
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["div_learning_rate"])
        
    def forward(self, state_dict):
        div_state = state_dict["player_div"]          
        
        div_embed = self.norm_div(F.relu(self.fc_div(div_state)))
       
        h_in = state_dict["hidden_div"]
        out, h_out = self.lstm(div_embed, h_in)
        
        z_out = self.norm_pi_z(self.fc_pi_z(out))
        prob_z = F.softmax(z_out, dim=2)
        
        return prob_z.squeeze(), h_out
