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
        
        self.fc_cat = nn.Linear(64*4, 256)
        
        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_cat = nn.LayerNorm(256)
        
        self.fc_pi_z1 = nn.Linear(256, 164)
        self.fc_pi_z2 = nn.Linear(164, arg_dict["div_num"])
        self.norm_pi_z1 = nn.LayerNorm(164)
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["div_learning_rate"])
        
    def forward(self, state_dict):
        player_state = state_dict["player_div"]          
        player_prime_state = state_dict["player_div_prime"]
        ball_state = state_dict["ball"]              
        ball_prime_state = state_dict["ball_prime"]              

        dim_set = player_state.size()
        dim_num = len(dim_set)
        
        player_embed = self.norm_player(self.fc_player(player_state))
        player_prime_embed = self.norm_player(self.fc_player(player_prime_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        ball_prime_embed = self.norm_ball(self.fc_ball(ball_prime_state))

        if dim_num == 2: 
            cat = torch.cat([player_embed, ball_embed, player_prime_embed, ball_prime_embed], dim=1)
        else:
            cat = torch.cat([player_embed, ball_embed, player_prime_embed, ball_prime_embed], dim=2)

        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        
        z_out = F.relu(self.norm_pi_z1(self.fc_pi_z1(cat)))
        z_out = self.fc_pi_z2(z_out)

        if dim_num == 2:
            prob_z = F.softmax(z_out, dim=1)
        else:
            prob_z = F.softmax(z_out, dim=2)
        
        return prob_z.squeeze(), []
