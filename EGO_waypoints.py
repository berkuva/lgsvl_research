#!/usr/bin/env python3
#
# Copyright (c) 2019 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import os
import lgsvl
import random
import time
import math
import numpy as np

###################### simulator ######################
sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
if sim.current_scene == "BorregasAve":
  sim.reset()
else:
  sim.load("BorregasAve")

###################### EGO ######################
spawns = sim.get_spawn()
state = lgsvl.AgentState()
state.transform = spawns[0]
ego = sim.add_agent("Lincoln2017MKZ (Apollo 5.0)", lgsvl.AgentType.EGO, state)


###################### NPC ######################
# NPC's initial position
sx = spawns[0].position.x - 8
sy = spawns[0].position.y
sz = spawns[0].position.z + 150

state = lgsvl.AgentState()
state.transform = spawns[0]
state.transform.position.x = sx
state.transform.position.z = sz

state.transform.rotation.y = 180.0
npc = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)


###################### collision callback ######################
vehicles = {
  ego: "EGO",
  npc: "Sedan",
}

def on_collision(agent1, agent2, contact):
  name1 = vehicles[agent1]
  name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
  print("{} collided with {}".format(name1, name2))

ego.on_collision(on_collision)
npc.on_collision(on_collision)


###################### Time to Collision - reward fn in RL #####################
distance = 0
def ttc(): #calculate the time to collision between EGO and NPC
    global distance
    approaching = False
    dist_changed = 0
    

    dist = abs(math.sqrt( (npc.state.position.x - ego.state.position.x)**2 + \
                          (npc.state.position.y - ego.state.position.y)**2 + \
                          (npc.state.position.z - ego.state.position.z)**2))
    if distance == 0:
        distance = dist
        approaching = None
    else:
        dist_changed = distance - dist
        distance = dist
        approaching = dist_changed > 0
    

    relative_speed = npc.state.speed - ego.state.speed
    
    if approaching == None:
        ttc = "timestep = 0"
    elif approaching:
        ttc = npc.uid.split("(Clone)")[0] + " " + str(np.round(distance / relative_speed, 3)) + " seconds"
    else:
        if distance != 0:
          ttc = npc.uid.split("(Clone)")[0] + " is moving away from EGO"
        else:
          ttc = npc.uid.split("(Clone)")[0] + " " + str(np.round(distance / relative_speed, 3)) + " seconds"

    print(ttc)
    return distance, ttc



###################### NPC waypoints ######################
waypoints = [ \
    lgsvl.DriveWaypoint(position=lgsvl.Vector(13.81, -3.15, 63.50),
              angle=lgsvl.Vector(0, 180, 0),
              speed=10),

    lgsvl.DriveWaypoint(position=lgsvl.Vector(13.81, -2.2, 17),
              angle=lgsvl.Vector(0, 180, 0),
              speed=10),

    lgsvl.DriveWaypoint(position=lgsvl.Vector(13.81, -1.12, -46.505),
              angle=lgsvl.Vector(0, 180, 0),
              speed=10),
]


def on_waypoint(agent, index):
  print("=======")
  print("waypoint {} reached".format(index))

npc.on_waypoint_reached(on_waypoint)
npc.follow(waypoints)


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


###################### connect to bridge; run simulation ######################
# An EGO will not connect to a bridge unless commanded to
print("Bridge connected:", ego.bridge_connected)
# The EGO is now looking for a bridge at the specified IP and port
ego.connect_bridge("127.0.0.1", 9090)
print("Waiting for connection...")
while not ego.bridge_connected:
  time.sleep(1)
print("Bridge connected:", ego.bridge_connected)
print("Initializing simulation")
sim.run(1)
input("Press Enter to run")
sim.run(1)
ttc()
sim.run(1)
ttc()
input("Press Enter to change weather")
sim.weather = lgsvl.WeatherState(rain=0.9, fog=0.3, wetness=0.1)
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
sim.run(1)
ttc()
