import traci
import numpy as np
import random
import matplotlib.pyplot as plt
import logging

class TrafficOptimizer:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3, exploration_decay=0.99, min_exploration=0.1, num_states=8, num_actions=4):
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay  # Decay for epsilon
        self.min_exploration = min_exploration  # Minimum epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))  # Q-table for state-action values
        self.max_highway_density = 0
        self.max_ramp_density = 0

    def encode_state(self, ramp_density, highway_density, time_of_day):
        """Encodes the state based on traffic density and time of day."""
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
        return np.argmax(state)  # Convert to a single state index

    def get_time_of_day(self, sim_time):
        """Determines the time of day based on simulation time."""
        if 0 <= sim_time < 1800:
            return 'morning'
        elif 1800 <= sim_time < 3600:
            return 'midday'
        elif 3600 <= sim_time < 5400:
            return 'evening'
        else:
            return 'night'

    def get_current_state(self):
        """Retrieves the current state of the traffic system."""
        highway_density = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
        self.max_highway_density = max(highway_density, self.max_highway_density)

        ramp_density = traci.lane.getLastStepVehicleNumber("E2_0")
        self.max_ramp_density = max(ramp_density, self.max_ramp_density)

        sim_time = traci.simulation.getTime()
        time_of_day = self.get_time_of_day(sim_time)
        return self.encode_state(ramp_density, highway_density, time_of_day)

    def select_action(self, state, evaluate=False):
        """Selects an action using an epsilon-greedy policy."""
        if not evaluate and np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)  # Exploration
        return np.argmax(self.q_table[state, :])  # Exploitation

    def compute_reward(self, highway_density, ramp_queue, ramp_time):
        """Computes the reward based on traffic conditions."""
        throughput = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
        collisions = len(traci.simulation.getCollisions())
        emergency_brakes = sum(1 for vehicle_id in traci.vehicle.getIDList() if traci.vehicle.getEmergencyDecel(vehicle_id) > 0)
        return throughput - 0.5 * highway_density - 2 * ramp_queue - 10 * collisions - 5 * emergency_brakes - 0.05 * ramp_time

    def train(self, epoch):
        """Trains the Q-learning agent for one epoch."""
        traci.start([
            "sumo-gui",
            "-c",
            "../Simulation-with-Trafic-Light/sumo.sumocfg",
            "--start",
            "true",
            "--xml-validation",
            "never",
            "--log",
            "log",
            "--quit-on-end"
        ])
        print(f"Starting training for epoch {epoch}")

        for _ in range(10):
            traci.simulation.step()

        state = self.get_current_state()
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(state)
            traci.trafficlight.setPhase("feux", action)
            traci.simulation.step()

            ramp_time = traci.simulation.getTime()
            highway_density = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
            ramp_queue = traci.lane.getLastStepVehicleNumber("E2_0")
            reward = self.compute_reward(highway_density, ramp_queue, ramp_time)
            total_reward += reward

            next_state = self.get_current_state()
            self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
            state = next_state
            done = (traci.simulation.getMinExpectedNumber() == 0)

        traci.close()
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        return total_reward

    def test_policy(self):
        """Tests the learned policy and evaluates performance."""
        traci.start([
            "sumo-gui",
            "-c",
            "../Simulation-with-Trafic-Light/sumo.sumocfg",
            "--start",
            "true",
            "--xml-validation",
            "never",
            "--log",
            "log",
            "--quit-on-end"
        ])
        print("Testing the learned policy...")

        for _ in range(10):
            traci.simulation.step()

        state = self.get_current_state()
        total_reward = 0
        ramp_time = 0
        done = False

        while not done:
            action = self.select_action(state, evaluate=True)
            traci.trafficlight.setPhase("feux", action)
            traci.simulation.step()

            ramp_time += 1
            highway_density = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
            ramp_queue = traci.lane.getLastStepVehicleNumber("E2_0")
            reward = self.compute_reward(highway_density, ramp_queue, ramp_time)
            total_reward += reward

            next_state = self.get_current_state()
            state = next_state
            done = (traci.simulation.getMinExpectedNumber() == 0)

        traci.close()
        print(f"Test Results: Total Reward = {total_reward}, Ramp Metering Time = {ramp_time}")
        return total_reward, ramp_time

# Main Execution
if __name__ == '__main__':
    traffic_optimizer = TrafficOptimizer()
    num_epochs = 4
    rewards = []

    for epoch in range(num_epochs):
        total_reward = traffic_optimizer.train(epoch)
        rewards.append(total_reward)
        print(f"Epoch {epoch + 1}: Total Reward = {total_reward}")

    # Plot Rewards
    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("Reward Trend Over Time")
    plt.show()

    # Test Policy
    traffic_optimizer.test_policy()