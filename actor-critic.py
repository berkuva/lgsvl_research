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
        self.affine1 = nn.Linear(5, 128)

        # actor's layer - chooses probability strengths for z_position, npc_speed, rain, fog, wetness, time of day
        self.action_layer1 = nn.Linear(128, 128)
        self.action_layer2 = nn.Linear(128, 5)

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

        # pdb.set_trace()

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
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
        # print("\n# Current control policy:")
        # print(signal.control_policy)
        # Create a new control policy
        control_policy = "trigger=100;green=100;yellow=0;red=0;loop"
        # Control this traffic light with a new control policy
        signal.control(control_policy)

        self.z_position = sz
        # print("initial z_position: {}".format(self.z_position))

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

    def update_state(self, state, i_episode):
        '''update self.z_position and self.npc_speed using the model'''
        action_probs, state_value = model(state)
        # print("action_probs {}".format(action_probs))

        z_pos_strength = action_probs[0].item()
        speed_strength = action_probs[1].item()
        rain_strength = action_probs[2].item()
        fog_strength = action_probs[3].item()
        wetness_strength = action_probs[4].item()
        # timeofday_strength = action_probs[5].item()

        if i_episode < 5:
            self.z_position += z_pos_strength
            self.npc_speed += speed_strength
        else:
            self.z_position = z_pos_strength*10
            self.npc_speed = speed_strength*10

        self.rain_rate = round(rain_strength, 2)
        self.fog_rate = min(0.5, round(fog_strength, 2))
        self.wetness_rate = round(wetness_strength, 2)
        # self.timeofday = int(timeofday_strength)

        # print("new z_position {}, npc_speed {}".format(self.z_position, self.npc_speed))
        
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(action_probs)
        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        model.saved_actions.append(SavedAction(m.log_prob(action_probs), state_value))
        # return action_probs
        return action_probs

    def sample_waypoint(self):
        '''sample a waypoint that depends on self.z_position'''
        self.y_position += 0.2

        position = lgsvl.Vector(13.81, self.y_position, self.z_position)

        waypoint = lgsvl.DriveWaypoint(position=position,
                                       angle=lgsvl.Vector(0, 180, 0),
                                       speed=self.npc_speed)
        return waypoint

    def step(self, waypoint):
        '''
        move npc to new waypoint, adjust to new weather and time of day
        return reward (1/ttc) & done
        '''
        self.npc.follow([waypoint])
        sim.weather = lgsvl.WeatherState(rain=self.rain_rate,
                                         fog=self.fog_rate,
                                         wetness=self.wetness_rate)
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

        # print("NPC speed {}".format(self.npc_speed))
        # print("EGO speed {}".format(self.ego.state.speed))

        relative_speed = self.npc_speed - self.ego.state.speed

        ttc = abs(np.round(dist / relative_speed, 3))
        # print("distance: {}, relative speed {}".format(dist, relative_speed))
        # print("TTC {}".format(ttc))
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
        # pdb.set_trace()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]

    def run_simulator(self):
        # runtime = 10
        # i = 0
        # while i < 10:
        #     if not self.collided:
        #         # print("The NPC is currently at {}".format(self.npc.state.position))
        #         self.sim.run(1)
        #         i += 1
        #     else:
        #         break
        sim.run(1)

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

        state = torch.FloatTensor([ self.z_position,
                                    self.npc_speed,
                                    self.rain_rate,
                                    self.fog_rate,
                                    self.wetness_rate])
                                    # self.timeofday])

        # reset episode reward
        ep_reward = 0
        action_probs = None
        # run 10 times for each episode
        for t in range(1, 12):
            print("\niteration {}".format(t))
            # print("state {}".format(state))

            # select action from policy
            action_probs = self.update_state(state, i_episode)
            state = torch.FloatTensor([ self.z_position,
                                        self.npc_speed,
                                        self.rain_rate,
                                        self.fog_rate,
                                        self.wetness_rate])
                                        # self.timeofday])

            waypoint = self.sample_waypoint()
            # print("waypoint {}, at speed {}".format(waypoint.position,
                                                    # waypoint.speed))

            # take the action
            reward, done = self.step(waypoint)
            # print("reward {}, done {}".format(reward, done))

            self.run_simulator()

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

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
