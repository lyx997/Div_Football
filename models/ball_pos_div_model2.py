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

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player_div"],arg_dict["div_lstm_size"])  
        
        self.norm_player = nn.LayerNorm(arg_dict["div_lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["div_lstm_size"], arg_dict["div_lstm_size"])

        self.fc_pi_z = nn.Linear(arg_dict["div_lstm_size"], 5)
        self.norm_pi_z = nn.LayerNorm(5)
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["div_learning_rate"])
        
    def forward(self, state_dict):
        ball_pos_state = state_dict["player_div"]          
        
        player_embed = self.norm_player(F.relu(self.fc_player(ball_pos_state)))
       
        h_in = state_dict["hidden_div"]
        out, h_out = self.lstm(player_embed, h_in)
        
        z_out = self.norm_pi_z(self.fc_pi_z(out))
        prob_z = F.softmax(z_out, dim=2)
        
        return prob_z.squeeze(), h_out
