# Background & Introduction
In simple robot exploration the robot’s main focus is to navigate through an unexplored environment, detecting and avoiding obstacles, while simultaneously creating a map. This map does not, in simple robot exploration, identify the obstacles or declare any semantic meaning to them. For a search and rescue robot exploring unknown disaster environments, the semantic meaning of an object is necessary to be able to prioritize where to explore first. This does not just mean identifying people, but also being able to prioritize exploration around larger objects of interest, such as cars, where it is more likely to find other smaller key objects, such as people, that might be harder to find.

# Problem description
This project aims to implement the semantic exploration of a Clearpath Husky A200 in a simulated, static environment. The robot uses object detection and localisation to detect objects of interest with a camera, and remember the object locations relative to an occupancy grid. The robot uses the semantic values of the objects to determine exploration priority. The goal is to evaluate the performance of the semantic exploration in comparison to simple exploration, where no semantic values in the robot’s environment are used.

# Video of experiment
https://www.youtube.com/watch?v=10vzFfcSj4Y
