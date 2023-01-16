import numpy as np

SMM_WIDTH = 96
SMM_HEIGHT = 72
SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

# Normalized minimap coordinates
MINIMAP_NORM_X_MIN = -1.0
MINIMAP_NORM_X_MAX = 1.0
MINIMAP_NORM_Y_MIN = -1.0 / 2.25
MINIMAP_NORM_Y_MAX = 1.0 / 2.25

_MARKER_VALUE = 255

def mark_points(frame, points):
  """Draw dots corresponding to 'points'.

  Args:
    frame: 2-d matrix representing one SMM channel ([y, x])
    points: a list of (x, y) coordinates to be marked
  """
  for p in range(len(points) // 2):
    x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
            (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
    y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
            (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
    x = max(0, min(frame.shape[1] - 1, x))
    y = max(0, min(frame.shape[0] - 1, y))
    frame[y, x] = _MARKER_VALUE

def get_smm_layers(config):
  return SMM_LAYERS

def generate_smm(observation, config=None,
                 channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):

    frame = np.zeros((channel_dimensions[1],
                    channel_dimensions[0], len(get_smm_layers(config))),
                   dtype=np.uint8)
    o = observation
    for index, layer in enumerate(get_smm_layers(config)):
      assert layer in o
      if layer == 'active':
        if o[layer] == -1:
          continue
        mark_points(frame[ :, :, index],
                    np.array(o['left_team'][o[layer]]).reshape(-1))
      else:
        mark_points(frame[ :, :, index], np.array(o[layer]).reshape(-1))
    return frame


class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y  = 0, 0
        
    def get_feature_dims(self):
        dims = {
            'player':29,
            'ball':18,
            'left_team':7,
            'left_team_closest':7,
            'right_team':7,
            'right_team_closest':7,
            'smm': (4, 72, 96)
        }
        return dims

    def encode(self, obs):
        player_num = obs['active']
        
        player_pos_x, player_pos_y = obs['left_team'][player_num]
        player_direction = np.array(obs['left_team_direction'][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][player_num]
        is_dribbling = obs['sticky_actions'][9]
        is_sprinting = obs['sticky_actions'][8]

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs['ball_direction']
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0 
        if obs['ball_owned_team'] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs['ball_owned_team'] == 0:
            ball_owned_by_us = 1.0
        elif obs['ball_owned_team'] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0
            
        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y) 
        
        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0
        
        avail = self._get_avail(obs, ball_distance)
        player_state = np.concatenate((avail[2:], obs['left_team'][player_num], player_direction*100, [player_speed*100],
                                       player_role_onehot, [ball_far, player_tired, is_dribbling, is_sprinting]))
        
        
        ball_state = np.concatenate((np.array(obs['ball']), 
                                     np.array(ball_which_zone),
                                     np.array([ball_x_relative, ball_y_relative]),
                                     np.array(obs['ball_direction'])*20,
                                     np.array([ball_speed*20, ball_distance, ball_owned, ball_owned_by_us])))
        

        obs_left_team = np.delete(obs['left_team'], player_num, axis=0)
        obs_left_team_direction = np.delete(obs['left_team_direction'], player_num, axis=0)
        left_team_relative = obs_left_team
        left_team_distance = np.linalg.norm(left_team_relative - obs['left_team'][player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(obs['left_team_tired_factor'], player_num, axis=0).reshape(-1,1)
        left_team_state = np.concatenate((left_team_relative*2, obs_left_team_direction*100, left_team_speed*100, \
                                          left_team_distance*2, left_team_tired), axis=1)
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]
        
        
        obs_right_team = np.array(obs['right_team'])
        obs_right_team_direction = np.array(obs['right_team_direction'])
        right_team_distance = np.linalg.norm(obs_right_team - obs['left_team'][player_num], axis=1, keepdims=True)
        right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
        right_team_tired = np.array(obs['right_team_tired_factor']).reshape(-1,1)
        right_team_state = np.concatenate((obs_right_team*2, obs_right_team_direction*100, right_team_speed*100, \
                                           right_team_distance*2, right_team_tired), axis=1)
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        #smm = np.reshape(smm_obs, (4,72,96))
        smm = np.reshape(generate_smm(obs), (4, 72, 96))
        
        state_dict = {"player": player_state,
                      "ball": ball_state,
                      "left_team" : left_team_state,
                      "left_closest" : left_closest_state,
                      "right_team" : right_team_state,
                      "right_closest" : right_closest_state,
                      "avail" : avail,
                      "smm": smm}

        return state_dict
    
    def _get_avail(self, obs, ball_distance):
        avail = [1,1,1,1,1,1,1,1,1,1,1,1]
        NO_OP, MOVE, LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SPRINT, RELEASE_MOVE, \
                                                      RELEASE_SPRINT, SLIDE, DRIBBLE, RELEASE_DRIBBLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        
        if obs['ball_owned_team'] == 1: # opponents owning ball
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        elif obs['ball_owned_team'] == -1 and ball_distance > 0.03 and obs['game_mode'] == 0: # Ground ball  and far from me
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        else: # my team owning ball
            avail[SLIDE] = 0
            
        # Dealing with sticky actions
        sticky_actions = obs['sticky_actions']
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0
            
        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0
            
        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0
            
        
        # if too far, no shot
        ball_x, ball_y, _ = obs['ball']
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x<=1.0) and (-0.27<=ball_y and ball_y<=0.27):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0
            
            
        if obs['game_mode'] == 2 and ball_x < -0.7:  # Our GoalKick 
            avail = [1,0,0,0,0,0,0,0,0,0,0,0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)
        
        elif obs['game_mode'] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1,0,0,0,0,0,0,0,0,0,0,0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)
        
        elif obs['game_mode'] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1,0,0,0,0,0,0,0,0,0,0,0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)
        
    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if   (-END_X <= ball_x    and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0,0,0,0,0,0]
        elif (-END_X <= ball_x    and ball_x < -MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,1.0,0,0,0,0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,0,1.0,0,0,0]
        elif (PENALTY_X < ball_x  and ball_x <=END_X)     and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0,0,0,1.0,0,0]
        elif (MIDDLE_X < ball_x   and ball_x <=END_X)     and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,0,0,0,1.0,0]
        else:
            return [0,0,0,0,0,1.0]
        

    def _encode_role_onehot(self, role_num):
        result = [0,0,0,0,0,0,0,0,0,0]
        result[role_num] = 1.0
        return np.array(result)