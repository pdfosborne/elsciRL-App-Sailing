import torch

class DefaultAdapter:
    def __init__(self, setup_info):
        # No discretization needed; continuous state (4 floats)
        pass

    def adapter(self, state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        # state: [cross_track_error, course_angle_err, distance, r]
        if encode:
            tensor = torch.tensor(state, dtype=torch.float32)
            return tensor
        else:
            return state
