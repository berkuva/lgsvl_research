import os
import lgsvl
import sys
import time
import math
import evaluator
import numpy as np

MAX_SPEED = 10
INITIAL_HEADWAY = 90
SPEED_VARIANCE = 10
TIME_DELAY = 2
TIME_LIMIT = 25+TIME_DELAY

class GenerateScene:
    def __init__(self, EGO_start = None):
        print("Generating Scene..")

        self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

        if self.sim.current_scene == "SanFrancisco":
            self.sim.reset()
        else:
            self.sim.load("SanFrancisco")

        if EGO_start == None:
            self.EGO_start = lgsvl.Vector(-3.0, 11.9, 8.1)
        else:
            self.EGO_start = EGO_start

    def generate_EGO(self):
        egoState = lgsvl.AgentState()
        egoState.transform = self.sim.map_point_on_lane(self.EGO_start)
        # egoState.velocity = lgsvl.Vector(-5, 0, 0)

        self.ego = self.sim.add_agent("XE_Rigged-apollo_3_5", lgsvl.AgentType.EGO, egoState)

        # enable sensors required for Apollo 3.5
        sensors = self.ego.get_sensors()

        for s in sensors:
            if s.name in ['velodyne', 'Main Camera', 'Telephoto Camera', 'GPS', 'IMU']:
                s.enabled = True

        self.ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)


    def generate_NPCs(self):
        # Encroaching Vehicle in opposite lane
        npcState = lgsvl.AgentState()

        x = self.EGO_start.x
        y = self.EGO_start.y
        z = self.EGO_start.z

        ##### NPC1 #####
        npcState.transform = self.sim.map_point_on_lane(lgsvl.Vector(x+7, y, z))
        self.npc1 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, npcState)

        npc1Speed = 5

        n1x = self.npc1.state.position.x
        n1y = self.npc1.state.position.y
        n1z = self.npc1.state.position.z

        self.npc1Waypoints = []
        self.npc1Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n1x-12, n1y, n1z+0.5), npc1Speed))
        self.npc1Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n1x-14, n1y, n1z+2.5), npc1Speed))
        self.npc1Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n1x-16, n1y, n1z+6), npc1Speed))
        self.npc1Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n1x-18, n1y, n1z+9), npc1Speed))
        self.npc1Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n1x-20, n1y, n1z+13), npc1Speed))


        ##### NPC2 #####
        npcState.transform = self.sim.map_point_on_lane(lgsvl.Vector(x-INITIAL_HEADWAY, y, 1.0))
        npcState.transform.rotation = lgsvl.Vector(0, 90, 0)
        self.npc2 = self.sim.add_agent("Jeep", lgsvl.AgentType.NPC, npcState)

        npc2Speed = 12

        n2x = self.npc2.state.position.x
        n2y = self.npc2.state.position.y
        n2z = self.npc2.state.position.z

        self.npc2Waypoints = []
        self.npc2Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n2x+50, n2y, n2z), npc2Speed-5)) # ready for left turn
        self.npc2Waypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(n2x+65, n2y, n2z+30), npc2Speed))
        
        # pt = self.EGO_start
        # pt.x += 5
        # self.npc2Waypoints.append(lgsvl.DriveWaypoint(pt, npc2Speed))


    def on_collision(self, agent1, agent2, contact):
        raise evaluator.TestException("Colision between {} and {}".format(agent1, agent2))

    def ttc(self): #calculate the time to collision between the NPCs
        approaching = False
        dist_changed = False
        
        dist = abs(math.sqrt( (self.npc1.state.position.x - self.npc2.state.position.x)**2 + \
                              (self.npc1.state.position.y - self.npc2.state.position.y)**2 + \
                              (self.npc1.state.position.z - self.npc2.state.position.z)**2))
        if distance == None:
            distance = dist
            approaching = None
        else:
            dist_changed = distance - dist
            distance = dist
            approaching = dist_changed > 0
        

        relative_speed = self.npc2.state.speed - self.npc1.state.speed
        
        if approaching == None:
            ttc = "timestep = 0"
        elif approaching:
            ttc = self.npc2.uid.split("(Clone)")[0] + " " + str(np.round(distance / relative_speed, 3)) + " seconds"
        else:
            ttc = self.npc2.uid.split("(Clone)")[0] + " is moving away from NPC"
        print(ttc)
        return distance, ttc

    def run(self):
        self.ego.on_collision(self.on_collision)
        self.npc1.on_collision(self.on_collision)
        self.npc2.on_collision(self.on_collision)

        try:
            t0 = time.time()
            self.sim.run(TIME_DELAY)
            # self.npc1.follow(self.npc1Waypoints)
            self.npc2.follow(self.npc2Waypoints)

            distance = None
            ttc = None

            while True:
                distance, ttc = self.ttc(distance, ttc)

                egoCurrentState = self.ego.state
                if egoCurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
                    raise evaluator.TestException("EGO speed exceeded limit, {} > {} m/s".format(egoCurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

                NPC1CurrentState = self.npc1.state
                if NPC1CurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
                    raise evaluator.TestException("NPC1 speed exceeded limit, {} > {} m/s".format(NPC1CurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

                NPC2CurrentState = self.npc2.state
                if NPC2CurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
                    raise evaluator.TestException("NPC2 speed exceeded limit, {} > {} m/s".format(NPC2CurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

                self.sim.run(0.5)

                if time.time() - t0 > TIME_LIMIT:
                    break

        except evaluator.TestException as e:
            print("FAILED: " + repr(e))
            exit()

        print("Program Terminated")

if __name__ == "__main__":
    scene = GenerateScene()
    scene.generate_EGO()
    scene.generate_NPCs()
    scene.run()
