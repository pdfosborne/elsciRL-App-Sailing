from typing import Dict
import torch
from torch import Tensor

class Adapter:
    def __init__(self, setup_info:dict={}) -> None:  
        # Create a mapping from state string to unique id
        self.index_encoder: Dict[str, int] = {}
        self.encoder_idx: int = 0
        self.input_dim = 1
        
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Default adapter to define the state space for the agent in the correct elsciRL format."""

        # Encode to Tensor for agents
        if encode:
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