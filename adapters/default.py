from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder
from gymnasium.spaces import Discrete

class Adapter:
    # ------ Static Methods ---------------------------------------
    # - Defined by simulator source https://github.com/PPierzc/ai-learns-to-sail/blob/master/tasks/channel.py
    @staticmethod
    def angle_to_state(angle):
        return int(30 * ((angle + np.pi) / (2 * np.pi) % 1))  # Discretization of the angle space
    
    @staticmethod
    def x_to_state(x):
        return int(40 * ((x + -10) / 20))  # Discretization of the x space
    
    @staticmethod
    def state_discretizer(state):
        x = float(state.split('_')[0])
        x_state = Adapter.x_to_state(x)

        angle = float(state.split('_')[1])
        angle_state = Adapter.angle_to_state(angle)

        state_out = str(x_state)+'_'+str(angle_state)
        return state_out
    # -------------------------------------------------------------

    _cached_state_idx: Dict[str, int] = dict()
    def __init__(self, setup_info:dict={}) -> None:
        # ------ State Encoder ---------------------------------------
        # Initialise encoder based on all possible env states
        # - x: -10 to 10 with N decimal precision
        # all_possible_x = [i*-1 for i in range(10*(10**setup_info['obs_precision']))]
        # for i in range(10*(10**setup_info['obs_precision'])):
        #     all_possible_x.append(i)
        # all_possible_angle = [i for i in range(30)]
        # # Need an index that preserves the identity of both the x and angle values
        # all_possible_states = []
        # for x_ind in all_possible_x:
        #     for angle_ind in all_possible_angle:
        #         index = "{n:.{d}f}".format(n=x_ind, d=setup_info['obs_precision'])+'_'+"{:0.1f}".format(angle_ind) 
        #         all_possible_states.append(index)
        # Input to pre-built possible state encoder
        # Initialize the encoder with one-hot encoding for each state
        
        num_x_states = 10*2 # -10 to 10 
        num_angle_states = 30 
        self.num_states = num_x_states*(10**setup_info['obs_precision']) * num_angle_states  # 2000 x states and 30 angle states
        
        # -------------------------------------------------------------
        # Observartion is string: "x_angle"
        # -> Then discretized and returned as string: "x_state_angle_state"
        # -> Before being numeritized to a unique id (x:-10-10*2dp * angle:0-2pi*1dp)
        self.observation_space = Discrete(2000*30)

        # elsciRL state encoder is large and not needed for tabular agents
        self.one_hot_encoder = StateEncoder(self.num_states)
        # Create a mapping from state string to unique id
        self.index_encoder: Dict[str, int] = {}
        self.encoder_idx: int = 0
        self.input_dim = 1
        
    
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Default adapter to define the state space for the agent in the correct elsciRL format."""
        #state = Adapter.state_discretizer(state)

        # Encode to Tensor for agents
        if encode:
            # if indexed use the elsciRL state encoder, othrewise single index based encoding
            if (indexed):
                # elsciRL state encoder is large and not needed for tabular agents
                # - Causes issure for training neural agents however
                state_encoded = self.one_hot_encoder.encode(state=state, legal_actions=legal_moves, episode_action_history=episode_action_history, indexed=indexed)
            else:
                if state not in self.index_encoder:
                    # If the state is not in the encoder, add it
                    state_encoded = torch.tensor([self.encoder_idx]).float()  # Use the index as the state encoded value
                    # Store the encoded state in the encoder dictionary
                    self.index_encoder[state] = state_encoded
                    self.encoder_idx += 1
                else:
                    # If the state is already in the encoder, retrieve its index
                    state_encoded = self.index_encoder[state]
        else:
            state_encoded = state

        
        return state_encoded