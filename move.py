import numpy as np
from model import Manipulator_X
from gpsmine import ROS_connection
import pickle


model = Manipulator_X(T=10)
ros_agent = ROS_connection(model)
trj = model.set_x_init()
ros_agent.move_arm_once(trj)
learned_policy = pickle.load(open('policy.pkl','rb'))
for t in range(model.T):
	trj = np.asarray(trj)
	next_policy = learned_policy[t]
	trj = model.plant_dyn(trj,next_policy,0,0)
	trj = ros_agent.move_arm_once(trj)

print 'get end effector location'
final =  model.getPosition(trj)
print(final)
print 'get target position'
print model.fin_position
