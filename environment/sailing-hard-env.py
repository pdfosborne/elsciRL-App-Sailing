# Ship Sailing
# - https://github.com/MarineAutonomy/Deep-Reinforcement-Learning-Based-Control-for-Ship-Navigation

import numpy as np
from scipy.integrate import solve_ivp
import kcs

class Engine:
    def __init__(self, local_setup_info: dict):
        # Store optional setup info
        self.local_setup_info = local_setup_info
        self.episode_ended = False
        self.counter = 0
        self.x_goal = 0
        self.y_goal = 0
        self.distance = 0
        self.obs_state = [1, 0, 0, 0, 0, 0, 0]

        self.train_test_flag = local_setup_info.get("train_test_flag", 0)
        self.wind_flag = local_setup_info.get("wind_flag", 0)
        self.wind_speed = local_setup_info.get("wind_speed", 0)
        self.wind_dir = local_setup_info.get("wind_dir", 0)

        self.wave_flag = local_setup_info.get("wave_flag", 0)
        self.wave_height = local_setup_info.get("wave_height", 0)
        self.wave_period = local_setup_info.get("wave_period", 0)
        self.wave_dir = local_setup_info.get("wave_dir", 0)

        # Test waypoints
        self.test_x_waypoints = local_setup_info.get("test_x_waypoints", None)
        self.test_y_waypoints = local_setup_info.get("test_y_waypoints", None)
        self.nwp = local_setup_info.get("nwp", None)
        self.initial_obs_state = local_setup_info.get("initial_obs_state", None)
        self.wp_counter = 1

        self.action_history = []
        self.obs_history = []

    def reset(self, start_obs=None):
        # Reset environment state, optionally to a specific observation
        if self.train_test_flag == 0:
            self.obs_state = [1, 0, 0, 0, 0, 0, 0]

            radius = np.random.randint(8, 28)
            random_theta = 2 * np.pi * np.random.random()

            self.x_goal = radius * np.cos(random_theta)
            self.y_goal = radius * np.sin(random_theta)

            self.episode_ended = False
            x_goal = self.x_goal
            y_goal = self.y_goal

            x_dot = 1
            y_dot = 0

            vec2 = np.array([x_goal, y_goal])
            Uvec = np.array([x_dot, y_dot])

            course_angle = np.arctan2(Uvec[1], Uvec[0])
            psi_vec2 = np.arctan2(vec2[1], vec2[0])
            course_angle_err = course_angle - psi_vec2
            course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi

            self.counter += 1
            observation = np.array([0, course_angle_err, radius, 0], dtype=np.float32)
        else:
            self.obs_state = self.initial_obs_state
            self.x_goal = self.test_x_waypoints[1]
            self.y_goal = self.test_y_waypoints[1]

            x = self.test_x_waypoints[0]
            y = self.test_y_waypoints[0]
            u = self.initial_obs_state[0]
            v = self.initial_obs_state[1]
            psi = self.initial_obs_state[5]
            x_dot = u * np.cos(psi) - v * np.sin(psi)
            y_dot = u * np.sin(psi) + v * np.cos(psi)

            vec2 = np.array([self.x_goal, self.y_goal])
            Uvec = np.array([x_dot, y_dot])

            course_angle = np.arctan2(Uvec[1], Uvec[0])
            psi_vec2 = np.arctan2(vec2[1], vec2[0])
            course_angle_err = course_angle - psi_vec2
            course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi

            dist_to_goal = np.sqrt((self.x_goal - x) ** 2 + (self.y_goal - y) ** 2)
            observation = np.array([0, course_angle_err, dist_to_goal, 0], dtype=np.float32)

        self.action_history = []
        self.obs_history = [observation.copy()]
        return observation

    def step(self, state, action_no):
        # Action set
        action_set = [-35*np.pi/180, 0, 35*np.pi/180]
        delta_c = action_set[action_no]

        tspan = (0, 0.3)
        yinit = self.obs_state

        sol = solve_ivp(lambda t, v: kcs.KCS_ode(t, v, delta_c,
                                                 wind_flag=self.wind_flag,
                                                 wind_speed=self.wind_speed,
                                                 wind_dir=self.wind_dir,
                                                 wave_flag=self.wave_flag,
                                                 wave_height=self.wave_height,
                                                 wave_period=self.wave_period,
                                                 wave_dir=self.wave_dir),
                        tspan, yinit, t_eval=tspan, dense_output=True)

        # extract solution
        u = sol.y[0][-1]
        v = sol.y[1][-1]
        r = sol.y[2][-1]
        x = sol.y[3][-1]
        y = sol.y[4][-1]
        psi_rad = sol.y[5][-1]
        psi = psi_rad % (2 * np.pi)
        delta = sol.y[6][-1]

        self.obs_state[0] = u
        self.obs_state[1] = v
        self.obs_state[2] = r
        self.obs_state[3] = x
        self.obs_state[4] = y
        self.obs_state[5] = psi
        self.obs_state[6] = delta

        x_init = 0
        y_init = 0
        x_goal = self.x_goal
        y_goal = self.y_goal

        # DISTANCE TO GOAL
        distance = ((x - x_goal)**2 + (y - y_goal)**2) ** 0.5
        self.distance = distance

        # CROSS TRACK ERROR
        vec1 = np.array([x_goal - x_init, y_goal - y_init])
        vec2 = np.array([x_goal - x, y_goal - y])
        vec1_hat = vec1 / np.linalg.norm(vec1)
        cross_track_error = np.cross(vec2, vec1_hat)

        # COURSE ANGLE ERROR
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)

        Uvec = np.array([x_dot, y_dot])
        Uvec_hat = Uvec / np.linalg.norm(Uvec)
        vec2_hat = vec2 / np.linalg.norm(vec2)

        course_angle = np.arctan2(Uvec[1], Uvec[0])
        psi_vec2 = np.arctan2(vec2[1], vec2[0])

        course_angle_err = course_angle - psi_vec2
        course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi

        # REWARDS
        R1 = 2 * np.exp(-0.08 * cross_track_error ** 2) - 1
        R2 = 1.3 * np.exp(-10 * (abs(course_angle_err))) - 0.3
        R3 = -distance * 0.25
        reward = R1 + R2 + R3

        observation = np.array([cross_track_error, course_angle_err, distance, r], dtype=np.float32)
        self.obs_history.append(observation.copy())
        self.action_history.append(action_no)

        terminated = False
        info = {}

        # DESTINATION CHECK
        if abs(distance) <= 0.5:
            reward = 100
            self.episode_ended = True
            terminated = True
            return observation, reward, terminated, info

        # TERMINATION CHECK (skipped for brevity, include for full code)
        angle_btw23 = np.arccos(np.clip(np.dot(vec2_hat, Uvec_hat), -1.0, 1.0))
        angle_btw12 = np.arccos(np.clip(np.dot(vec1_hat, vec2_hat), -1.0, 1.0))
        if angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2:
            self.episode_ended = True
            terminated = True
            return observation, reward, terminated, info

        return observation, reward, terminated, info

    def legal_move_generator(self, state=None):
        # Always 3 discrete actions
        return [0, 1, 2]

    def render(self, state=None):
        import matplotlib.pyplot as plt
        obs = self.obs_state if state is None else state
        fig, ax = plt.subplots()
        ax.set_title("Ship state (x,y)")
        ax.plot(obs[3], obs[4], marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig

    def close(self):
        pass  # nothing extra
