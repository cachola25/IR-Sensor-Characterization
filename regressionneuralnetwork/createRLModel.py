import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import gym
from gym import spaces
import os

# Define the custom environment
class RobotEnv(gym.Env):
    """Custom Environment for the Robot"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobotEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Move Forward, 1 - Turn Left, 2 - Turn Right, 3 - Stop
        self.action_space = spaces.Discrete(4)

        # Observations: [predicted_distance, start_angle, end_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, -180, -180]),
            high=np.array([np.inf, 180, 180]),
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.done = False

        # Set seed for reproducibility
        self.seed()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Randomly initialize the predicted distance and angles
        predicted_distance = np.random.uniform(20, 100)  # Distance between 20cm and 100cm
        start_angle = np.random.uniform(-90, 0)          # Start angle between -90° and 0°
        end_angle = np.random.uniform(0, 90)             # End angle between 0° and 90°
        self.state = np.array([predicted_distance, start_angle, end_angle], dtype=np.float32)
        self.done = False
        return self.state

    def step(self, action):
        """Execute one time step within the environment"""
        predicted_distance, start_angle, end_angle = self.state

        # Simulate the effect of the action
        if action == 0:  # Move Forward
            predicted_distance -= 10  # Decrease distance by 10cm
            predicted_distance = max(predicted_distance, 0)  # Distance cannot be negative
        elif action == 1:  # Turn Left
            start_angle += 15
            end_angle += 15
            start_angle = min(start_angle, 180)
            end_angle = min(end_angle, 180)
        elif action == 2:  # Turn Right
            start_angle -= 15
            end_angle -= 15
            start_angle = max(start_angle, -180)
            end_angle = max(end_angle, -180)
        elif action == 3:  # Stop
            pass  # No change in state

        # Update the state
        self.state = np.array([predicted_distance, start_angle, end_angle], dtype=np.float32)

        # Calculate reward
        reward = self.calculate_reward(action)

        # Check if episode is done
        if predicted_distance == 0:
            self.done = True

        return self.state, reward, self.done, {}

    def calculate_reward(self, action):
        """Calculate the reward for the current state"""
        predicted_distance, start_angle, end_angle = self.state

        # Reward for reducing distance
        distance_reward = -predicted_distance

        # Reward for alignment
        center_angle = (start_angle + end_angle) / 2
        alignment_reward = -abs(center_angle)

        # Penalty for unnecessary actions
        action_penalty = 0
        if action == 3:  # Stop
            action_penalty = -10  # Discourage stopping unless at target

        # Total reward
        total_reward = distance_reward + alignment_reward + action_penalty
        return total_reward

    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass  # Not implemented

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.batch_size = 32  # Batch size for training
        self.model = self._build_model()  # Build the Q-network

    def _build_model(self):
        """Build the neural network model"""
        model = models.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on the current state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state,verbose=0)
        return np.argmax(act_values[0])  # Exploit

    def replay(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # Initialize target
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            # Compute target Q-values
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = RobotEnv()
state_size = env.observation_space.shape[0]  # Should be 3
action_size = env.action_space.n  # Should be 4
agent = DQNAgent(state_size, action_size)
episodes = 500  # Number of episodes to train
max_timesteps = 100  # Max steps per episode

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(max_timesteps):
        # Decide action
        action = agent.act(state)
        # Execute action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        # Move to next state
        state = next_state
        # Train the agent
        agent.replay()
        # Check if episode is done
        if done:
            print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
            break

# Save the trained model
model_name = 'dqn_robot_model.keras'
if os.path.isfile(model_name):
    print(f"{model_name} already exists")
    ans = input("Do you want to replace it? (y/n)").lower()
    while ans not in ["y", "n"]:
        ans = input("Do you want to replace it? (y/n)").lower()
    if ans == "n":
        curr = 1
        name = f"dqn_robot_model_{curr}.keras"
        while os.path.isfile(name):
            curr += 1
            name = f"dqn_robot_model_{curr}.keras"
        model_name = name
agent.model.save(model_name)
print(f"Model saved as {model_name}")

       