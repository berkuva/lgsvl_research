import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os
import lgsvl
import random
import time

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_false',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


class Scenario():
    def __init__(self):
        '''start simulator, spawn EGO and NPC'''
        ###################### simulator ######################
        self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
        if self.sim.current_scene == "BorregasAve":
            self.sim.reset()
        else:
            self.sim.load("BorregasAve")

        ###################### EGO ######################
        spawns = self.sim.get_spawn()
        state = lgsvl.AgentState()
        state.transform = spawns[0]
        self.ego = self.sim.add_agent("Lincoln2017MKZ (Apollo 5.0)", lgsvl.AgentType.EGO, state)

        ###################### NPC ######################
        sx = spawns[0].position.x - 8
        sy = spawns[0].position.y
        sz = spawns[0].position.z + 150

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        state.transform.position.x = sx
        state.transform.position.z = sz

        state.transform.rotation.y = 180.0
        self.npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

        self.waypoints = []

        self.y_positions = []
        self.z_positions = []

        ###################### Traffic light is kept green ######################
        controllables = sim.get_controllables()
        # Pick a traffic light of intrest
        signal = controllables[2]
        # Get current controllable states
        print("\n# Current control policy:")
        print(signal.control_policy)
        # Create a new control policy
        control_policy = "trigger=100;green=100;yellow=0;red=0;loop"
        # Control this traffic light with a new control policy
        signal.control(control_policy)

        self.distance = 0

        self.R = 0
        self.saved_actions = model.saved_actions
        self.policy_losses = []  # list to save actor (policy) loss
        self.value_losses = []  # list to save critic (value) loss
        self.returns = []  # list to save the true values

    def on_waypoint(self, agent, index):
        print("=======")
        print("waypoint {} reached".format(index))

    def sample_waypoint(self, timestep):
        x_pos = 13.81

        if timestep>0:
            y_pos = self.y_positions[timestep-1] - 0.15
            z_pos = self.z_positions[timestep-1] - random.randint(5, 10)
        else:
            y_pos = -3.15
            z_pos = 63.50

        self.y_positions.append(y_pos)
        self.z_positions.append[z_pos]

        wp = lgsvl.Vector(x_pos, y_pos, z_pos)
        self.waypoints.append(wp)
        return wp


    def select_action(self):
        EGO_position = np.array([self.ego.state.position.x,
                                 self.ego.state.position.y,
                                 self.ego.state.position.z])
        NPC_position = np.array([self.npc.state.position.x,
                                 self.npc.state.position.y,
                                 self.npc.state.position.z])
        fog = lgsvl.WeatherState()

    def calculate_ttc(self):
        '''calculate the time to collision between EGO and NPC'''
        approaching = False
        dist_changed = 0

        dist = abs(math.sqrt((self.npc.state.position.x - self.ego.state.position.x) ** 2 + \
                             (self.npc.state.position.y - self.ego.state.position.y) ** 2 + \
                             (self.npc.state.position.z - self.ego.state.position.z) ** 2))
        if self.distance == 0:
            self.distance = dist
            approaching = None
        else:
            dist_changed = self.distance - dist
            self.distance = dist
            approaching = dist_changed > 0

        relative_speed = self.npc.state.speed - self.ego.state.speed

        self.ttc = np.round(self.distance / relative_speed, 3)

        if approaching == None:
            ttc_log = "timestep = 0"
        elif approaching:
            ttc_log = self.npc.uid.split("(Clone)")[0] + " " + str(self.ttc) + " seconds"
        else:
            if self.distance != 0:
                ttc_log = self.npc.uid.split("(Clone)")[0] + " is moving away from EGO"
            else:
                ttc_log = self.npc.uid.split("(Clone)")[0] + " " + str(self.ttc) + " seconds"

        print(ttc_log)

    def step(self):
        '''
        Input: action to take
            (which have already took effect by setting weather, time of day)
        Return: resulting state
                reward earned
                finished in terminal state or not
                others

        In this function, we run the simulator with the set actions,
        and calculate the reward
        '''
        ###################### connect to bridge; run simulation ######################
        # An EGO will not connect to a bridge unless commanded to
        print("Bridge connected:", self.ego.bridge_connected)
        # The EGO is now looking for a bridge at the specified IP and port
        self.ego.connect_bridge("127.0.0.1", 9090)
        print("Waiting for connection...")
        while not self.ego.bridge_connected:
            time.sleep(1)
        print("Bridge connected:", self.ego.bridge_connected)
        print("Initializing simulation")
        sim.run(1)
        input("Press Enter to run")

        runtime = 30  # seconds
        current_time = 0

        while current_time < runtime:
            self.sim.run(1)
            self.calculate_ttc()
            self.finish_episode()

    def finish_episode(self):
        """
        Training code. Calcultes actor and critic loss, reward, and performs backprop.
        """
        # calculate the true value using rewards returned from the environment
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

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

    def main():
        running_reward = 10

        # number of times to run scenario
        for i_episode in range(1):

            # reset environment and episode reward
            simulator, ego, npc = reset_environment()

            ep_reward = 0

            # select action from policy
            sim.weather, time_of_day = select_action(simulator)
            action = [sim.weather, time_of_day]

            # take the action
            state, reward, done, _ = step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            finish_episode()

            # log results
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

            # check if crash happens
            if running_reward > env.spec.reward_threshold:
                print("Crashed! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break


if __name__ == '__main__':
    scenario = Scenario()
    scenario.main()
