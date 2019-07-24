import os
import lgsvl
import sys
import time
import math
import evaluator


MAX_SPEED = 10
INITIAL_HEADWAY = 90
SPEED_VARIANCE = 4
TIME_LIMIT = 14
TIME_DELAY = 3
IMAGE_PATH = "/home/derek/Desktop/simulator/Api/examples/NHTSA-sample-tests/Encroaching-Oncoming-Vehicles/"


class GenerateScene:
    def __init__(self, EGO_start = None):
        print("Generating Scene..")

        self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

        if self.sim.current_scene == "SanFrancisco":
            self.sim.reset()
        else:
            self.sim.load("SanFrancisco")

        if EGO_start == None:
            # spawn EGO at right lane for a right turn
            self.EGO_start = lgsvl.Vector(-3, 9.9, 8.1)
        else:
            self.EGO_start = EGO_start

    def generate_EGO(self):
        egoState = lgsvl.AgentState()
        egoState.transform = self.sim.map_point_on_lane(self.EGO_start)
        egoState.velocity = lgsvl.Vector(-5, 0, 0)

        egoControl = lgsvl.VehicleControl()
        egoControl.turn_signal_right = True

        self.ego = self.sim.add_agent("XE_Rigged-apollo_3_5", lgsvl.AgentType.EGO, egoState)
        self.ego.apply_control(egoControl)

        # enable sensors required for Apollo 3.5
        self.sensors = self.ego.get_sensors()

        for s in self.sensors:
            if s.name in ['velodyne', 'Main Camera', 'Telephoto Camera', 'GPS', 'IMU']:
                s.enabled = True

        self.ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)


    def generate_POV(self):
        # Encroaching Vehicle in opposite lane
        POVState = lgsvl.AgentState()
        POVState.transform.position = lgsvl.Vector(self.EGO_start.x  - INITIAL_HEADWAY,
        											self.EGO_start.y,
        											1.0)
        POVState.transform.rotation = lgsvl.Vector(0, 90, 0)
        # Vehicles: DeliveryTruck, Hatchback, Jeep, Sedan, SchoolBus, SUV
        self.POV = self.sim.add_agent("Jeep", lgsvl.AgentType.NPC, POVState)

        POVControl = lgsvl.NPCControl()
        POVControl.turn_signal_left = True
        self.POV.apply_control(POVControl)

        npcSpeed = 12
        nx = self.POV.state.position.x
        ny = self.POV.state.position.y
        nz = self.POV.state.position.z

        self.POVWaypoints = []
        self.POVWaypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(nx+35, ny, nz), npcSpeed))
        self.POVWaypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(nx+50, ny, nz), npcSpeed-5)) # ready for left turn
        self.POVWaypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(nx+62, ny, nz+5), npcSpeed-5))
        self.POVWaypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(nx+65, ny, nz+10), npcSpeed-4))
        self.POVWaypoints.append(lgsvl.DriveWaypoint(lgsvl.Vector(nx+65, ny, nz+20), npcSpeed))


    # def on_collision(self, agent1, agent2, contact):
    #     raise evaluator.TestException("Colision between EGO and POV")

    def save_camera_image(self, timestep):
        for s in self.sensors:
            if s.name == "Main Camera":
                # s.save(IMAGE_PATH + "/waypoints/" + str(timestep) + "main-camera.jpg", quality=75)
                s.save(IMAGE_PATH + "/training/" + str(timestep) + "main-camera.jpg", quality=75)
                print("image saved")
                break

    def run(self):
        # self.ego.on_collision(self.on_collision)
        # self.POV.on_collision(self.on_collision)

        try:
            t0 = time.time()
            self.sim.run(TIME_DELAY)
            self.POV.follow(self.POVWaypoints)

        	# Speed check for ego and POV
            timestep = 0
            while True:
                egoCurrentState = self.ego.state
                if egoCurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
                    raise evaluator.TestException("Ego speed exceeded limit, {} > {} m/s".format(egoCurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

                POVCurrentState = self.POV.state
                if POVCurrentState.speed > MAX_SPEED + SPEED_VARIANCE:
                    raise evaluator.TestException("POV1 speed exceeded limit, {} > {} m/s".format(POVCurrentState.speed, MAX_SPEED + SPEED_VARIANCE))

                self.sim.run(1)
                self.save_camera_image(timestep)

                if time.time() - t0 > TIME_LIMIT:
                    break

                timestep += 1
        except evaluator.TestException as e:
            print("FAILED: " + repr(e))
            exit()

        print("Program Terminated")

if __name__ == "__main__":
    scene = GenerateScene()
    scene.generate_EGO()
    scene.generate_POV()
    scene.run()