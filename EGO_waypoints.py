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
a = sim.add_agent("Lincoln2017MKZ (Apollo 5.0)", lgsvl.AgentType.EGO, state)


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
  a: "EGO",
  npc: "Sedan",
}

def on_collision(agent1, agent2, contact):
  name1 = vehicles[agent1]
  name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
  print("{} collided with {}".format(name1, name2))

a.on_collision(on_collision)
npc.on_collision(on_collision)


###################### NPC waypoints ######################
# Each waypoint is a position vector paired with the speed that the NPC will drive to it
# waypoints = []
# x_max = 2
# z_delta = 2

# layer_mask = 0
# layer_mask |= 1 << 0 # 0 is the layer for the road (default)

# for i in range(20):
#   px = 0
#   pz = (i + 1) * z_delta

#   # Raycast the points onto the ground because BorregasAve is not flat
#   hit = sim.raycast(origin=lgsvl.Vector(sx + px, sy, sz - pz),
#   					direction=lgsvl.Vector(0, -1, 0),
#   					layer_mask=layer_mask) 
#   # NPC will wait for 0 second at each waypoint
#   wp = lgsvl.DriveWaypoint(position=hit.point, speed=7)
#   # print(wp.position)
#   waypoints.append(wp)

# print(a.state.position)
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
print("Bridge connected:", a.bridge_connected)
# The EGO is now looking for a bridge at the specified IP and port
a.connect_bridge("127.0.0.1", 9090)
print("Waiting for connection...")
while not a.bridge_connected:
  time.sleep(1)
print("Bridge connected:", a.bridge_connected)
print("Initializing simulation")
sim.run(1)
input("Press Enter to run")
sim.run(30)
