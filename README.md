# Towards Automated Safety Coverage and Testing for Autonomous Vehicles (2019)

This repository contains a computer science Master's degree research I conducted at the University of Virginia, titled "Towards Automated Safety Coverage and Testing for Autonomous Vehicles". I was advised by [Prof. Madhur Behl](https://engineering.virginia.edu/faculty/madhur-behl).

It uses reinforcement learning (actor critic in particular) to try to generate edge case driving scenarions in which a state-of-the-art self-driving vehicles publicly available to date--[Baidu's Apollo](http://apollo.auto/)--will not drive safely.

For scene generation and the integration of Apollo's autonomous driving modules, such as perception, planning and control, we use the [LGSVL simulator](https://www.lgsvlsimulator.com/), a photo realistic autonomous driving simulator.

Among many driving parameters, such as road geometry (number of lanes, positions of traffic lights/traffic signs), status of traffic lights, weather (rain, fog, wetness), time of day, movements of non-autonomous vehicles (NPC vehicles) and pedestrians, our RL agent's action space includes weather, time of day, and movements of non-autonomous vehicles (NPC vehicles).

We generate two such edge cases. In the first edge case scenario, the RL agent controls the parameters above. As a result of this, we found that Apollo creates a dense fog (fog rate 100%), invalidating Apollo's perception module. Apollo was not able to recognize traffic lights and was not safely drive to its destination. In the second scenario, we hoped to find a scenario in which an NPC vehicle and Apollo collided. To achieve this goal, we uniformly sampled a set of waypoints for the NPC vehicle to follow, and the RL agent's only action space was to determine the velocities at which the vehicle should be traveling at each waypoint to maximize the chances of collision with Apollo. We discovered that since we did not restrict the velocities to be positive, the NPC vehicle was able to "trick" the Apollo vehicle and drive backwards (negative velocity) by stopping at a waypoint until Apollo was right behind it.

actor-critic.py and actor-critic2.py contain the code for the first and second edge case scenarios respectively. For more detailed explanations, please visit my [website](https://hyunjaecho94.github.io/) and see my project and/or slides.
