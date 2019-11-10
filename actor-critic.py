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

parser = argparse.ArgumentParser(description='lgsvl actor-critic')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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
        self.affine1 = nn.Linear(2, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)#probability of moving in z direction, probability of speeding 

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

    def reset_environment(self):
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
        self.ego.on_collision(self.on_collision)

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
        self.npc.on_waypoint_reached(self.on_waypoint)
        self.npc.on_collision(self.on_collision)
        
        self.y_positions = []

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

        self.z_position = sz
        self.npc_speed = 7

        self.timestep=0

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

    def on_collision(self, contact):
        print("EGO and NPC collided!")
        
    def on_waypoint(self, agent, index):
        print("=======")
        print("waypoint {} reached".format(index))

    def sample_waypoint(self):
        '''sample a waypoint that depends on self.z_position'''
        if self.timestep>0:
            y_pos = self.y_positions[self.timestep-1] - 0.15
        else:
            y_pos = -3.15
        self.y_positions.append(y_pos)
        self.timestep += 1

        position = lgsvl.Vector(13.81, y_pos, self.z_position)
        angle = lgsvl.Vector(0, 180, 0)
        
        waypoint = lgsvl.DriveWaypoint(wp, angle, self.npc_speed)

        return waypoint

    def select_action(self, state):
        '''update self.z_position and self.npc_speed using the model'''
        action_probs, state_value = model(state)
        action_probs[0] *= -10 #new self.z_position = -5 (action_probs[0]==0.5. 0.5*-10=-5)
        action_probs[1] *= 14 #speed 7 on average (action_probs[1]==0.5. 0.5*14=7)

        self.z_position += action_probs[0].item() #expected value=-5
        self.speed = action_probs[1].item() #expected value=7

        #save to action buffer
        model.saved_actions.append(SavedAction(action_probs, state_value))
        return action_probs

    def step(self, action_probs, waypoint):
        '''
        1. move npc to new waypoint,
        2. reward (1/ttc) & done
        '''
        self.npc.follow(waypoint)
        time2collision = ttc()
        reward = 1/time2collision
        done = self.timestep>=30
        return reward, done
        
    def calculate_ttc(self):
        '''calculate the time to collision between EGO and NPC'''
        dist = abs(math.sqrt((self.npc.state.position.x - self.ego.state.position.x) ** 2 + \
                             (self.npc.state.position.y - self.ego.state.position.y) ** 2 + \
                             (self.npc.state.position.z - self.ego.state.position.z) ** 2))

        relative_speed = self.npc.state.speed - self.ego.state.speed

        ttc = np.round(dist / relative_speed, 3)
        return ttc

    def finish_episode(self):
        """
        Training code. Calcultes actor and critic loss, reward, and performs backprop.
        """
        # calculate the true value using rewards returned from the environment
        R = 0
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

    def run_simulator(self):
        self.sim.run(1)
        input("Enter to resume")

    def main():
        running_reward = 10

        # number of times to run scenario
        for i_episode in range(1):

            # start simulator, set up environment, connect to bridge
            self.reset_environment()
            self.connect2bridge()

            # reset episode reward
            ep_reward = 0

            # run 10 times for each episode
            for t in range(1, 10):
                state = torch.FloatTensor([self.z_position, self.npc_speed])

                # select action from policy
                action_probs = select_action(state) #action=tensor(z distance to move, speed)
                waypoint = sample_waypoint()

                # take the action
                reward, done = step(action_probs, waypoint)

                self.run_simulator()

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
            if :
                print("Crashed!")
                break


if __name__ == '__main__':
    scenario = Scenario()
    scenario.main()
