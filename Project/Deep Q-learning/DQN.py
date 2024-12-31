import traci
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import deque

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 0.5
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 15
NUM_STATES = 8
NUM_ACTIONS = 4
MAX_EPOCHS = 50

# Neural Network for Deep Q-Learning
class TrafficControlNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrafficControlNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Initialize networks and replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = TrafficControlNet(NUM_STATES, NUM_ACTIONS).to(device)
target_net = TrafficControlNet(NUM_STATES, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ExperienceReplay(REPLAY_CAPACITY)

# Helper functions
def encode_state(ramp_density, highway_density, time_of_day):
    thresholds = {
        'morning': {'ramp_low': 3, 'ramp_high': 6, 'highway_low': 25},
        'midday': {'ramp_low': 2, 'ramp_high': 4, 'highway_low': 15},
        'evening': {'ramp_low': 3, 'ramp_high': 5, 'highway_low': 20},
        'night': {'ramp_low': 1, 'ramp_high': 2, 'highway_low': 5}
    }
    thresholds = thresholds[time_of_day]
    state = [
        int(ramp_density <= thresholds['ramp_low']),
        int(thresholds['ramp_low'] < ramp_density <= thresholds['ramp_high']),
        int(ramp_density > thresholds['ramp_high']),
        int(highway_density <= thresholds['highway_low']),
        int(highway_density > thresholds['highway_low']),
        int(time_of_day == 'morning'),
        int(time_of_day == 'midday'),
        int(time_of_day == 'evening')
    ]
    return np.array(state, dtype=np.float32)

def get_time_of_day(sim_time):
    if 0 <= sim_time < 1800:
        return 'morning'
    elif 1800 <= sim_time < 3600:
        return 'midday'
    elif 3600 <= sim_time < 5400:
        return 'evening'
    else:
        return 'night'

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return torch.argmax(q_values).item()

def compute_reward(highway_density, ramp_queue):
    throughput = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
    collisions = traci.simulation.getCollisions()
    emergency_brakes = sum(traci.vehicle.getEmergencyDecel(v) > 0 for v in traci.vehicle.getIDList())
    reward = throughput - 0.5 * highway_density - 2 * ramp_queue - 10 * len(collisions) - 5 * emergency_brakes
    return reward

def update_policy():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = target_net(next_states).max(1)[0]
    target_q = rewards + (1 - dones.float()) * DISCOUNT_FACTOR * next_q
    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())

# Simulation loop
epsilon = INITIAL_EPSILON
for epoch in range(MAX_EPOCHS):
    traci.start(["sumo-gui", "-c", "../Simulation-with-Trafic-Light/sumo.sumocfg", "--start", "true", "--xml-validation", "never", "--log", "log", "--quit-on-end"])
    state = encode_state(0, 0, get_time_of_day(traci.simulation.getTime()))
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        traci.trafficlight.setPhase("feux", action)
        traci.simulation.step()
        ramp_density = traci.lane.getLastStepVehicleNumber("E2_0")
        highway_density = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
        next_state = encode_state(ramp_density, highway_density, get_time_of_day(traci.simulation.getTime()))
        reward = compute_reward(highway_density, ramp_density)
        done = (traci.simulation.getMinExpectedNumber() == 0)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        update_policy()

    if epoch % TARGET_UPDATE_FREQ == 0:
        update_target_network()

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    traci.close()
    print(f"Epoch {epoch + 1}: Total Reward = {total_reward}, Epsilon = {epsilon}")

# Save the final model
torch.save(policy_net.state_dict(), "traffic_control_model.pth")
print("Model saved successfully.")