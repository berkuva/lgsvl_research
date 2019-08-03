import os
import lgsvl
import sys
import time
import evaluator

# Vehicles: DeliveryTruck, Hatchback, Jeep, Sedan, SchoolBus, SUV
import math

import pdb


################ EGO ##################
EGO_start_default = (218, 9.9, 4.3)

print("Press Enter to set the value to default.")

EGO_start_x = input("Enter EGO's starting x position: ")
if EGO_start_x == "":
	EGO_start_x = EGO_start_default[0]
	print("EGO's x set to default ({})".format(EGO_start_x))

EGO_start_y = input("Enter EGO's starting y position: ")
if EGO_start_y == "":
	EGO_start_y = EGO_start_default[1]
	print("EGO's x set to default ({})".format(EGO_start_y))

EGO_start_z = input("Enter EGO's starting z position: ")
if EGO_start_z == "":
	EGO_start_z = EGO_start_default[2]
	print("EGO's z set to default ({})".format(EGO_start_z))

EGO_start = lgsvl.Vector(float(EGO_start_x), float(EGO_start_y), float(EGO_start_z))


print("Running EOV_S_25_20 ...")

MAX_SPEED = 11.111 # (40 km/h, 25 mph)
INITIAL_HEADWAY = 30
SPEED_VARIANCE = 4
TIME_LIMIT = 30
TIME_DELAY = 3


sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
if sim.current_scene == "SanFrancisco":
	sim.reset()
else:
	sim.load("SanFrancisco")

################ EGO ##################

# spawn EGO in the 2nd to right lane
egoState = lgsvl.AgentState()
# A point close to the desired lane was found in Editor.
# This method returns the position and orientation of the closest lane to the point.
egoState.transform = sim.map_point_on_lane(lgsvl.Vector(EGO_start.x, EGO_start.y, EGO_start.z))
ego = sim.add_agent("XE_Rigged-apollo_3_5", lgsvl.AgentType.EGO, egoState)

# enable sensors required for Apollo 3.5
sensors = ego.get_sensors()
for s in sensors:
	if s.name in ['velodyne', 'Main Camera', 'Telephoto Camera', 'GPS', 'IMU']:
		s.enabled = True

ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)

################ POV ##################

# Encroaching Vehicle in opposite lane
POVState = lgsvl.AgentState()
POVState.transform.position = lgsvl.Vector(EGO_start.x - 4.5 - INITIAL_HEADWAY,
											EGO_start.y,
											2.207)
POVState.transform.rotation = lgsvl.Vector(0, 90, 0)
POV = sim.add_agent("Jeep", lgsvl.AgentType.NPC, POVState)

POV_final_position = lgsvl.Vector(EGO_start.x, EGO_start.y, 2.207)

POVWaypoints = []
POVWaypoints.append(lgsvl.DriveWaypoint(POV_final_position, MAX_SPEED))

################ NPC 1 (behind EGO) ##################
NPC_1_state = lgsvl.AgentState()
NPC_1_state.transform.position = lgsvl.Vector(EGO_start.x + INITIAL_HEADWAY,
											EGO_start.y,
											EGO_start.z)
NPC_1_state.transform.rotation = lgsvl.Vector(0, -90, 0)

NPC1 = sim.add_agent("SchoolBus", lgsvl.AgentType.NPC, NPC_1_state)
# Prevents EGO from moving backwards
NPC1_final_position = EGO_start

NPC1Waypoints = []
NPC1Waypoints.append(lgsvl.DriveWaypoint(NPC1_final_position, MAX_SPEED))

################ NPC 2 (right of EGO) ##################
NPC_2_state = lgsvl.AgentState()
NPC_2_state.transform.position = lgsvl.Vector(EGO_start.x,
											EGO_start.y,
											8.1)
NPC_2_state.transform.rotation = lgsvl.Vector(0, -90, 0)

NPC2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, NPC_2_state)
# Prevents EGO from changing lane to right
NPC2_final_position = lgsvl.Vector(EGO_start.x - INITIAL_HEADWAY,
									EGO_start.y,
									8.1)
NPC2Waypoints = []
NPC2Waypoints.append(lgsvl.DriveWaypoint(NPC2_final_position, MAX_SPEED))


def on_collision(agent1, agent2, contact):
	raise evaluator.TestException("Colision between {} and {}".format(agent1, agent2))

ego.on_collision(on_collision)
POV.on_collision(on_collision)
NPC1.on_collision(on_collision)
NPC2.on_collision(on_collision)

try:
	t0 = time.time()
	sim.run(TIME_DELAY)
	POV.follow(POVWaypoints)
	NPC1.follow(NPC1Waypoints)
	NPC2.follow(NPC2Waypoints)

	# Speed check for ego and POV
	
	# check for the time to collision
	npcs = [POV, NPC1, NPC2]
	npc_dists = [None] * len(npcs)
	approaching = [False] * len(npcs)
	ttcs = [None] * len(npcs)
	counter = 0

	while True:
		ego_pos = ego.state.position

		for i, npc in enumerate(npcs):

			dist = abs(math.sqrt( (ego.state.position.x - npc.state.position.x)**2 + \
							  (ego.state.position.y - npc.state.position.y)**2 + \
							  (ego.state.position.z - npc.state.position.z)**2))

			if npc_dists[i] == None:
				npc_dists[i] = dist
			else:
				dist_changed = npc_dists[i] - dist
				npc_dists[i] = dist
				approaching[i] = dist_changed > 0
		
		if counter > 0:
			for i, npc in enumerate(npcs):
				relative_speed = npc.state.speed - ego.state.speed
				if approaching[i]:
					ttcs[i] = npc.uid.split("(Clone)")[0] + " (npc#" + str(i) + ") " + str(npc_dists[i] / relative_speed)
				else:
					ttcs[i] = npc.uid.split("(Clone)")[0] + " (npc#" + str(i) + ") moving away from EGO"
			print("TTC: ", ttcs)

		egoCurrentState = ego.state
		if egoCurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
			raise evaluator.TestException("Ego speed exceeded limit, {} > {} m/s".format(egoCurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

		POVCurrentState = POV.state
		if POVCurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
			raise evaluator.TestException("POV1 speed exceeded limit, {} > {} m/s".format(POVCurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

		NPC_1_CurrentState = NPC1.state
		if NPC_1_CurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
			raise evaluator.TestException("NPC1 speed exceeded limit, {} > {} m/s".format(NPC_1_CurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

		NPC_2_CurrentState = NPC2.state
		if NPC_2_CurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
			raise evaluator.TestException("NPC2 speed exceeded limit, {} > {} m/s".format(NPC_2_CurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

		sim.run(0.5)

		if time.time() - t0 > TIME_LIMIT:
			break
		
		counter += 1

except evaluator.TestException as e:
	print("FAILED: " + repr(e))
	exit()

print("Program Terminated")