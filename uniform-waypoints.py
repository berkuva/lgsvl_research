#!/usr/bin/env python3
#
# Hyun Jae Cho's MS Thesis Code
#
# Partially adopted from LG Electronics, Inc and PyTorch actor_critic tutorial:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
#


import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os
import gc
import lgsvl
import math
import time
import random

import pdb

parser = argparse.ArgumentParser(description='lgsvl actor-critic')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['action_probs', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(1, 128)

        # actor's layer - chooses probability strengths for the variables
        self.action_layer1 = nn.Linear(128, 128)
        self.action_layer2 = nn.Linear(128, 1)

        # critic's layer - evaluates being in current state
        self.value_layer1 = nn.Linear(128, 128)
        self.value_layer2 = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor
        x_a = self.action_layer1(x)
        x_a = self.action_layer2(x_a)
        action_prob = F.softmax(x_a, dim=-1)
        # critic
        x_c = self.value_layer1(x)
        state_values = self.value_layer2(x_c)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()

# try:
#     model.load_state_dict(torch.load('ACmodel.pt'))
#     print("Using previously saved model weights")
# except:
#     print("Not using previous weights")
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()
torch.autograd.set_detect_anomaly(True)


class Scenario():

    def __init__(self, sim):
        self.sim = sim
        self.ego = None
        self.npc = None
        self.z_position = None
        self.rain_rate = 0
        self.fog_rate = 0
        self.wetness_rate = 0
        # self.timeofday = random.randrange(25)
        self.y_position = -3.15
        self.npc_speed = 7
        self.collided = False

    def set_environment(self):
        '''start simulator, spawn EGO and NPC'''
        ###################### simulator ######################
        if self.sim.current_scene == "BorregasAve":
            self.sim.reset()
        else:
            self.sim.load("BorregasAve")

        ###################### EGO ######################
        spawns = self.sim.get_spawn()
        state = lgsvl.AgentState()
        state.transform = spawns[0]

        self.ego = self.sim.add_agent("Lincoln2017MKZ (Apollo 5.0)", lgsvl.AgentType.EGO, state)

        # sensors = self.ego.get_sensors()
        # c = lgsvl.VehicleControl()
        # c.turn_signal_left = True
        # self.ego.apply_control(c, True)

        ###################### NPC ######################
        sx = spawns[0].position.x - 8
        sy = spawns[0].position.y
        sz = spawns[0].position.z + 100

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        state.transform.position.x = sx
        state.transform.position.z = sz

        state.transform.rotation.y = 180.0

        self.npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
        self.npc.on_waypoint_reached(self.on_waypoint)

        self.vehicles = {
            self.ego: "EGO",
            self.npc: "Sedan",
        }
        self.ego.on_collision(self.on_collision)
        self.npc.on_collision(self.on_collision)

        ###################### Traffic light is kept green ######################
        controllables = self.sim.get_controllables()
        # Pick a traffic light of intrest
        signal = controllables[2]
        # Get current controllable states

        # Create a new control policy
        control_policy = "trigger=100;green=100;yellow=0;red=0;loop"
        # Control this traffic light with a new control policy
        signal.control(control_policy)

        self.z_position = sz
        self.final_z = spawns[0].position.x
        print("Z positions: Initial {}, Final {}".format(self.z_position, self.final_z))
        #sample 30 waypoints uniformly
        self.uniform_waypoints = self.sample_uniform_waypoints(self.final_z, self.z_position, 30)

    def connect2bridge(self):
        # An EGO will not connect to a bridge unless commanded to
        print("Bridge connected:", self.ego.bridge_connected)
        # The EGO is now looking for a bridge at the specified IP and port
        self.ego.connect_bridge("127.0.0.1", 9090)
        print("Waiting for connection...")
        while not self.ego.bridge_connected:
            time.sleep(1)
        print("Bridge connected:", self.ego.bridge_connected)
        print("Initializing simulation")
        self.sim.run(3)

    def on_collision(self, agent1, agent2, contact):
        self.collided = True
        name1 = self.vehicles[agent1]
        name2 = self.vehicles[agent2] if agent2 is not None else "OBSTACLE"
        print("############{} collided with {}############".format(name1, name2))

    def on_waypoint(self, agent, index):
        pass
        # print("=======")
        # print("waypoint {} reached".format(index))

    def update_state2(self, state, i_episode):
        '''Only handle the speed'''
        action_probs, state_value = model(state)
        print("action_probs ", action_probs)
        self.npc_speed = 10 * action_probs[0].item()

        m = Categorical(action_probs)
        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        # return action_probs
        return action_probs

    def sample_uniform_waypoints(self, start, end, n):
        '''Uniformly sample n waypoints from start to end '''
        z_locations = np.random.uniform(low=start, high=end, size=n)
        z_locations.sort()
        z_locations = np.flip(z_locations)

        u_waypoints = []
        for i in range(n):
            self.y_position += 0.08
            position = lgsvl.Vector(13.81, self.y_position, z_locations[i])
            waypoint = lgsvl.DriveWaypoint(position=position,
                                       angle=lgsvl.Vector(0, 180, 0),
                                       speed=self.npc_speed)
            u_waypoints.append(waypoint)
        return u_waypoints


    def step(self, waypoint):
        '''
        move npc to new waypoint, adjust to new weather and time of day
        return reward (1/ttc) & done
        '''
        self.npc.follow([waypoint])

        # sim.weather = lgsvl.WeatherState(rain=self.rain_rate,
        #                                  fog=self.fog_rate,
        #                                  wetness=self.wetness_rate)
        # sim.set_time_of_day(self.timeofday)
        if self.collided:
            reward = 100
            done = True
        else:
            reward = 1 / self.calculate_ttc()
            done = False
        return reward, done

    def calculate_ttc(self):
        '''calculate the time to collision between EGO and NPC'''
        dist = abs(math.sqrt((self.npc.state.position.x - self.ego.state.position.x) ** 2 + \
                             (self.npc.state.position.y - self.ego.state.position.y) ** 2 + \
                             (self.npc.state.position.z - self.ego.state.position.z) ** 2))

        relative_speed = self.npc_speed - self.ego.state.speed

        ttc = abs(np.round(dist / relative_speed, 3))
        return ttc

    def finish_episode(self):
        """
        Training code. Calcultes actor and critic loss, reward, and performs backprop.
        """
        # calculate the true value using rewards returned from the environment
        R = 0 #discounted reward
        saved_actions = model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (action_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-action_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]
        # torch.save(model.state_dict(), 'ACmodel.pth')


    def run_simulator(self, waypoint):
        '''
        run and render simulator until NPC reaches its next waypoint or collides with EGO
        '''
        pdb.set_trace()
        while abs(waypoint.position.z-self.npc.state.position.z) > 1 and self.collided==False:
            self.sim.run(1)

    def main(self, i_episode):
        global running_reward

        # run episodes
        print("i_episode {}".format(i_episode))

        # start simulator, set up environment, connect to bridge
        self.set_environment()
        self.connect2bridge()
        print("Finished setting the environment. Connected to bridge.")
        if i_episode == 1:
            input("Set waypoint through UI. Enter to continue.")

        state = torch.FloatTensor([self.npc_speed])

        # reset episode reward
        ep_reward = 0
        action_probs = None

        done = False
        iteration = 1
        while True:
            print("Iteration# ", iteration)
            # select action from policy
            action_probs = self.update_state2(state, i_episode)
            state = torch.FloatTensor([self.npc_speed])

            waypoint = self.uniform_waypoints[iteration]
            reward, done = self.step(waypoint)

            self.run_simulator(waypoint)

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

            iteration += 1

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        self.finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            with open("ac-log.txt", "a+") as logfile:
                logfile.write("   {} \t\t  {:.2f}\t\t\t{:.2f}\t\t{}\t{}\n".format(
                                                                                i_episode,
                                                                                ep_reward,
                                                                                running_reward,
                                                                                action_probs.tolist(),                                                                                # self.timeofday,
                                                                                self.collided))

sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

if __name__ == '__main__':
    # remove logfile if it already exists
    try:
        os.remove("ac-log.txt")
    except:
        pass
    # Titles of logging messages
    with open("ac-log.txt", "a+") as logfile:
        logfile.write("Episode\tEpisode cum reward\tAvg running reward\tact_prb\tcollided\n")
    # Initial reward
    running_reward = 10
    num_episodes = 50
    episode_counter = 1
    while episode_counter <= num_episodes:
        scenario = Scenario(sim)
        scenario.main(episode_counter)
        print("Finished episode {}\n\n".format(episode_counter))
        episode_counter += 1
        gc.collect()
