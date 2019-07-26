import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from GenerateScene import GenerateScene


import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='actor-critic for Apollo')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

namedtuple('SavedAction', ['log_prob', 'value'])


NUM_NODES = 128
NUM_PARAMS = 3 # position, rotation, speed
NUM_ACTIONS = 3 # steering, throttle, braking

NUM_NODES = NUM_NODES + NUM_PARAMS

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(7, NUM_NODES)

        # actor's layer
        self.action_head = nn.Linear(NUM_NODES, NUM_ACTIONS)

        # critic's layer
        self.value_head = nn.Linear(NUM_NODES, 1)

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
        # pdb.set_trace()
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

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

# Returns position, rotation, speed of EGO at start
def reset_sim(scene):
    scene.generate_EGO()
    scene.generate_POV()
    return scene.get_EGO_state(scene.ego.state)


def on_collision(agent1, agent2, contact):
    raise evaluator.TestException("Colision between EGO and POV")

def save_camera_image(timestep):
    for s in sensors:
        if s.name == "Main Camera":
            s.save(IMAGE_PATH + "/waypoints/" + str(timestep) + "main-camera.jpg", quality=75)
            # s.save(IMAGE_PATH + "/training/" + str(timestep) + "main-camera.jpg", quality=75)
            print("image saved")
            break

def log(state_tuple, control_tuple):
    print(time.time())

    position, rotation, speed = state_tuple
    
    print("Position : ", position)
    print("Rotation ", rotation)
    print("Speed ", speed)

    steering, throttle, braking, turn_signal_right = control_tuple

    print("Steering ", steering)
    print("Throttle ", throttle)
    print("Braking ", braking)
    print("Right signal", turn_signal_right, "\n")

# Generate an action in simulator
def step(action, ego, POV, sim, POVWaypoints):
    print("Stepping..")

    if action == 0:
        set_ego_state()
        pass

    elif action == 1:
        set_ego_state()
        pass

    elif action == 2:
        set_ego_state()
        pass

    ego.on_collision(on_collision)
    POV.on_collision(on_collision)

    t0 = time.time()
    POV.follow(POVWaypoints)

    timestep = 0

    egoCurrentState = ego.state

    sim.run(1)

    save_camera_image(timestep)
    pos, rot, spd = get_EGO_state(egoCurrentState)
    ste, thr, bra, tsr = get_EGO_control(lgsvl.VehicleControl())

    log((pos, rot, spd), (ste, thr, bra, tsr))


    # if time.time() - t0 > TIME_LIMIT:
    #     break

    return state, reward, done, info



def main():
    running_reward = -10

    # run inifinitely many episodes
    for i_episode in count(1):

        scene = GenerateScene()

        # reset environment and episode reward
        
        state = reset_sim(scene)
        position = state[0]
        rotation = state[1]
        speed = state[2]

        state = np.array((position.x, position.y, position.z, rotation.x, rotation.y, rotation.z, speed))
        # pdb.set_trace()
        POVWaypoints = scene.POVWaypoints

        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

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

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()











