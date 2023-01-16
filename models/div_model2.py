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

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player_div"],64)  
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"],64)
        
        self.fc_cat = nn.Linear(64*2,arg_dict["div_lstm_size"])
        
        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_cat = nn.LayerNorm(arg_dict["div_lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["div_lstm_size"], arg_dict["div_lstm_size"])

        self.fc_pi_z1 = nn.Linear(arg_dict["div_lstm_size"], 164)
        self.fc_pi_z2 = nn.Linear(164, arg_dict["div_num"])
        self.norm_pi_z1 = nn.LayerNorm(164)
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["div_learning_rate"])
        
    def forward(self, state_dict):
        player_state = state_dict["player_div"]          
        ball_state = state_dict["ball"]              
        
        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
       
        cat = torch.cat([player_embed, ball_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden_div"]
        out, h_out = self.lstm(cat, h_in)
        
        z_out = F.relu(self.norm_pi_z1(self.fc_pi_z1(out)))
        z_out = self.fc_pi_z2(z_out)
        prob_z = F.softmax(z_out, dim=2)
        
        return prob_z.squeeze(), h_out
