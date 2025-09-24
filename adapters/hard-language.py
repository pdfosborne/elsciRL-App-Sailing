import torch

class LanguageAdapter:
    def __init__(self, setup_info):
        # (Optional) Use a real language encoder (sentence transformer, etc.) as needed
        pass

    def adapter(self, state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        # Create a human-interpretable description
        desc = f"Ship state with cross-track error {state[0]:.2f}, course angle error {state[1]:.2f}, distance to goal {state[2]:.2f}, and yaw rate {state[3]:.2f}."
        if encode:
            # Example: simple encoding as a vector (do proper embedding if needed)
            tensor = torch.tensor(state, dtype=torch.float32)
            return tensor
        else:
            return desc
