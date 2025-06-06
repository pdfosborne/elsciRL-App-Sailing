from torch import Tensor

import numpy as np
from gymnasium.spaces import Box

# Link to relevant ENCODER
from elsciRL.adapters.LLM_state_generators.text_ollama import OllamaAdapter


class Adapter:
    def __init__(self, setup_info:dict={}) -> None:        
        # Define observation space
        self.observation_space = Box(low=-1, high=1, shape=(1,384), dtype=np.float32)

        self.LLM_adapter = OllamaAdapter(
            model_name=setup_info.get('model_name', 'llama3.2'),
            base_prompt=setup_info.get('system_prompt', 'You are playing a game of Chess.'),
            context_length=2000,
            action_history_length=setup_info.get('action_history_length', 5),
            encoder=setup_info.get('encoder', 'MiniLM_L6v2')
        )

        
    def adapter(self, state: str, legal_moves:list = None, episode_action_history:list = None, encode:bool=True, indexed: bool = False) -> Tensor:     
        """ Use Language description for every student for current grid position """

        x = float(state.split('_')[0])
        angle = float(state.split('_')[1])
        
        # Horizontal position
        if (x>-1)&(x<1):
            L_x = 'in the middle'
        elif (x>-3)&(x<3):
            L_x = 'near to the center'
        elif (x>-5)&(x<5):
            L_x = 'in between the edge and the center'
        elif (x>-7)&(x<7):
            L_x = 'near to the edge'
        elif (x>=-10)&(x<=10):
            L_x = 'very close to the edge'
        else:
            L_x = 'out of bounds'

        # Side of river
        if x<0:
            L_x_side = 'on the harbor side of the river'
        elif x>0:
            L_x_side = 'on the beach side of the river'
        else:
            L_x_side = ''

        # Angle
        # - Defined in radians where 90deg = 1.57
        # - Peak velocity at  45deg = pi/4 = 0.7853...
        if angle==0:
            L_angle = 'facing directly into the wind'
        elif (angle>-0.1)&(angle<0.1):
            L_angle = 'facing into the wind'
        elif (angle>-0.5)&(angle<0.5):
            L_angle = 'close hauled with wind'
        elif (angle>-1)&(angle<1):
            L_angle = 'cutting the wind'
        else:
            L_angle = 'moving across the wind'
        # Wind side
        if angle<0:
            L_wind_side = 'on the starboard side'
        elif angle>0:
            L_wind_side = 'on the port side'
        else:
            L_wind_side = ''

        L_state = 'The boat is ' + L_x_side + ' ' + L_x + ', ' + L_angle + ' ' + L_wind_side + ', '
        L_state = L_state.replace('  ', ' ').replace(' .','.').replace(' ,',',').replace(' and,','') # Remove double spaces
        
        # Last action taken and final language state output
        if len(episode_action_history)>0:    
            last_action = episode_action_history[-1]
            # if last_action==0:
            #     L_action = 'the last action was to turn to the left slightly.'
            # elif last_action==1:
            #     L_action = 'the last action was to turn to the right slightly.'

            if (x<=0)&(last_action==0):
                L_action = 'the last action was to turn towards the harbor.'
            elif (x<0)&(last_action==1):
                L_action = 'the last action was to turn towards the center of the river.'
            elif (x>=0)&(last_action==1):
                L_action = 'the last action was to turn towards the beach.'
            elif (x>0)&(last_action==0):
                L_action = 'the last action was to turn towards the center of the river.'

            state = L_state + ' ' + L_action
        else:
            state = L_state   

        # Use the elsciRL LLM adapter to transform and encode
        state_encoded = self.LLM_adapter.adapter(
            state=state, 
            legal_moves=legal_moves, 
            episode_action_history=episode_action_history, 
            encode=encode, 
            indexed=indexed
        )

        return state_encoded
    
    
            
            