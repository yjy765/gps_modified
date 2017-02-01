import roslib
import sys
import rospy
from std_msgs.msg import Float64
import numpy as np
import PyKDL
import random as rd
from sensor_msgs.msg import JointState
from model import Manipulator_X
import tensorflow as tf
from scipy.stats import multivariate_normal as mul_normal
import math

class ROS_connection():
	def __init__(self):
		rospy.init_node('tutorial_x_control')
		self.r = rospy.Rate(1)
		self.pub = []
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint1_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint2_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint3_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint4_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint5_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint6_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint7_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/grip_joint_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/grip_joint_sub_position/command', Float64, queue_size=1))
		self.sub_once = rospy.Subscriber('/robotis_manipulator_x/joint_states', JointState, self.callback)

		self.q_init = PyKDL.JntArray(7)
		self.q_status = PyKDL.JntArray(7)
                self.flag_callback = 1
                self.move_arm_once(self.q_init)
		return

	def callback(self, data):
                """
		self.q_init[0] = data.position[2]
		self.q_init[1] = data.position[3]
		self.q_init[2] = data.position[4]
		self.q_init[3] = data.position[5]
		self.q_init[4] = data.position[6]
		self.q_init[5] = data.position[7]
		self.q_init[5] = data.position[8]	
		self.sub_once.unregister()
                """			
		self.q_status[0] = data.position[2]
		self.q_status[1] = data.position[3]
		self.q_status[2] = data.position[4]
		self.q_status[3] = data.position[5]
		self.q_status[4] = data.position[6]
		self.q_status[5] = data.position[7]
		self.q_status[6] = data.position[8]
                if self.flag_callback == 1:
                        print "callback"	
	        	#print([self.q_status])
                        self.flag_callback = 0
		return

	def move_arm(self, T, trj):
		for i in range(T):
			for j in range(0,7):
				self.pub[j].publish(trj[i][j])			
			self.r.sleep()
			print("ros signal sent")
        	        #print(trj[i])
		return self.q_status

        def move_arm_once(self, target):
                self.flag_callback = 1		
                for j in range(0,7):
			self.pub[j].publish(target[j])			
                #while self.flag_callback == 0:		
                        #self.r.sleep()
		self.r.sleep()
		#wait_until(self.flag_callback == 0)
		print("ros signal sent_once")
     	        #print(target)
		return self.q_status

def test():
	model = Manipulator_X(T=10, weight=[1.0,5.0],verbose=False)
	model.build_ilqr_problem()

	N_sample = 100  #number of total sample
	N_sample_max = 200
	N_sample_init = 10 #number of sample from each ddp solution
	N_ddp_sol = 10  #number of ddp solution
	N_sol = 11
        policy_num = 0
	sample_num = 0 #last index of sample
	flag_rand = 0 #randomly pict sample

	policy_set = [None]*N_sol    #set of solution ddp & nn
	sample_set = [None]*N_sample	#set of samples from each solution

	#print "test 1 ",model.q_init

	#line 1&2 Generate DDP solutions and build sample set
        print "process 1 Generate DDP solutions"
	for i in range(N_ddp_sol):
		print "iLQR number ",i
		model.solve_ilqr_problem()
		res_temp = model.res
		res_temp['type'] = 'DDP'
		policy_set[i] = res_temp
		#model.generate_dest()
                policy_num += 1
		for j in range(N_sample_init):
			trajectory_temp = []
			action_temp = []
                        imp_q_temp = []
                        cost_temp = []
			trj = model.set_x_init()
			x_i = policy_set[i]['x_array_opt']
			u_i = policy_set[i]['u_array_opt'] 
			K_i = policy_set[i]['K_array_opt']
                        k_i = policy_set[i]['k_array_opt']
			Cov = policy_set[i]['Q_array_opt']
			for t in range(model.T):
                                next_action = np.random.multivariate_normal(u_i[t]+k_i[t]+K_i[t].dot(trj-x_i[t]),Cov[t])
                                l,_ = np.linalg.eig(Cov[t])
                                #print(np.real(l))
                                #assert(np.sum(l>0) == len(l))
				imp_q_temp.append(1.0)
				trajectory_temp.append(trj)
				action_temp.append(next_action)
				cost_temp.append(model.instaneous_cost(trj,next_action,t,0))
				trj = model.plant_dyn(trj,next_action,t,0)  #dynamics
                                
			sample_temp = {
                        'trajectory':trajectory_temp,
                        'action':action_temp,
                        #'pb':pb_temp,
                        'imp_q':imp_q_temp,
                        'cost':cost_temp,
                        'index':policy_num
                        }
                        sample_set[sample_num] = sample_temp
                        sample_num += 1

      
	#line 3 Initialize theta
	#calculate importance probability
	for i in range(sample_num):
                imp_q_temp = []
		trj = sample_set[i]['trajectory']
		next_action = sample_set[i]['action']
                prev_q_temp = np.zeros(policy_num)
		for t in range(model.T):
                        imp_q_temp.append(np.zeros(policy_num))
			for j in range(policy_num):
				x_i = policy_set[j]['x_array_opt']
				u_i = policy_set[j]['u_array_opt'] 
				K_i = policy_set[j]['K_array_opt']
				Cov = policy_set[j]['Q_array_opt']
                                k_i = policy_set[j]['k_array_opt']
				imp_q_temp[t][j] = prev_q_temp[j] +  mul_normal.logpdf(next_action[t], u_i[t]+k_i[t]+K_i[t].dot(trj[t]-x_i[t]),Cov[t])
                        prev_q_temp = imp_q_temp[t]
                        imp_q_temp[t] = np.mean(np.exp(imp_q_temp[t]))
		sample_set[i]['imp_q'] = imp_q_temp
	return sample_set


def establish_network(x,W1,b1,W2,b2):
	h = tf.nn.tanh(tf.matmul(x,W1))+b1
	y = tf.matmul(h,W2)+b2
        return y

def optimizer(loss):
        global_step = tf.Variable(0,trainable= False)
        learningRate = tf.train.exponential_decay(1.,global_step,100,0.99)
	train = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
        return train

if __name__ == '__main__':

	model = Manipulator_X(T=10, weight=[1.0,5.0],verbose=False)
	model.build_ilqr_problem()

        policy_var = 1e-14
	N_sample = 100  #number of total sample
	N_sample_max = 200
	N_sample_init = 10 #number of sample from each ddp solution
	N_ddp_sol = 10  #number of ddp solution
	N_sol = 11
        policy_num = 0
	sample_num = 0 #last index of sample
	flag_rand = 0 #randomly pict sample

	policy_set = [None]*N_sol    #set of solution ddp & nn
	sample_set = [None]*N_sample	#set of samples from each solution

	#print "test 1 ",model.q_init

	#line 1&2 Generate DDP solutions and build sample set
        print "process 1 Generate DDP solutions"
	for i in range(N_ddp_sol):
		print "iLQR number ",i
		model.solve_ilqr_problem()
		res_temp = model.res
		res_temp['type'] = 'DDP'
		policy_set[i] = res_temp
		#model.generate_dest()
                policy_num += 1
		for j in range(N_sample_init):
			trajectory_temp = []
			action_temp = []
                        imp_q_temp = []
                        cost_temp = []
			trj = model.set_x_init()
			x_i = policy_set[i]['x_array_opt']
			u_i = policy_set[i]['u_array_opt'] 
			K_i = policy_set[i]['K_array_opt']
                        k_i = policy_set[i]['k_array_opt']
			Cov = policy_set[i]['Q_array_opt']
			for t in range(model.T):
                                next_action = np.random.multivariate_normal(u_i[t]+k_i[t]+K_i[t].dot(trj-x_i[t]),Cov[t])
                                l,_ = np.linalg.eig(Cov[t])
                                #print(np.real(l))
                                #assert(np.sum(l>0) == len(l))
				imp_q_temp.append(1.0)
				trajectory_temp.append(trj)
				action_temp.append(next_action)
				cost_temp.append(model.instaneous_cost(trj,next_action,t,0))
				trj = model.plant_dyn(trj,next_action,t,0)  #dynamics
                                
			sample_temp = {
                        'trajectory':trajectory_temp,
                        'action':action_temp,
                        #'pb':pb_temp,
                        'imp_q':imp_q_temp,
                        'cost':cost_temp,
                        'index':policy_num
                        }
                        sample_set[sample_num] = sample_temp
                        sample_num += 1

      
	#line 3 Initialize theta
	#calculate importance probability
	for i in range(sample_num):
                imp_q_temp = []
		trj = sample_set[i]['trajectory']
		next_action = sample_set[i]['action']
		for t in range(model.T):
                        q_temp = 0
			for j in range(policy_num):
				x_i = policy_set[j]['x_array_opt']
				u_i = policy_set[j]['u_array_opt'] 
				K_i = policy_set[j]['K_array_opt']
				Cov = policy_set[j]['Q_array_opt']
                                k_i = policy_set[j]['k_array_opt']
				q_temp += mul_normal.pdf(next_action[t], u_i[t]+k_i[t]+K_i[t].dot(trj[t]-x_i[t]),Cov[t])
			q_temp /= policy_num
			if len(imp_q_temp) == 0:
        	                imp_q_temp.append(q_temp)
                        else:
                                imp_q_temp.append(imp_q_temp[-1]*q_temp)
		sample_set[i]['imp_q'] = imp_q_temp

	#train parameter theta
        print "process 2 Initialize parameter"
	N_hidden_node = 50
	x_best = tf.placeholder(tf.float32,[None, model.nj])
        x_new1 = tf.placeholder(tf.float32,[None,model.nj])
        x_new2 = tf.placeholder(tf.float32,[None,model.nj])

        W1_best = tf.Variable(tf.truncated_normal([model.nj,N_hidden_node],stddev=1.0/math.sqrt(float(model.nj)),name="W1_best"))
	b1_best = tf.Variable(tf.zeros([N_hidden_node]), name="b1_best")
	W2_best = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(N_hidden_node))), name="W2_best")
	b2_best = tf.Variable(tf.zeros([model.nj]), name="b2_best")

	W1_new1 = tf.Variable(tf.truncated_normal([model.nj, N_hidden_node], stddev=1.0/math.sqrt(float(model.nj))), name="W1_new1")
	b1_new1 = tf.Variable(tf.zeros([ N_hidden_node]), name="b1_new1")
	W2_new1 = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(N_hidden_node))), name="W2_new1")
	b2_new1 = tf.Variable(tf.zeros([model.nj]), name="b2_new1")

	W1_new2 = tf.Variable(tf.truncated_normal([model.nj, N_hidden_node], stddev=1.0/math.sqrt(float(model.nj))), name="W1_new2")
	b1_new2 = tf.Variable(tf.zeros([ N_hidden_node]), name="b1_new2")
	W2_new2 = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(N_hidden_node))), name="W2_new2")
	b2_new2 = tf.Variable(tf.zeros([model.nj]), name="b2_new2")

	y_best = establish_network(x_best,W1_best,b1_best,W2_best,b2_best)
        y_new1 = establish_network(x_new1,W1_new1,b1_new1,W2_new1,b2_new1)
        y_new2 = establish_network(x_new2,W1_new2,b1_new2,W2_new2,b2_new2)

	label_best = tf.placeholder(tf.float32,[None, model.nj])
        label_new1 = tf.placeholder(tf.float32,[None, model.nj])
        label_new2 = tf.placeholder(tf.float32,[None, model.nj])

	loss_best = tf.reduce_mean(tf.square(y_best-label_best))  # policy network - gaussian probability : log likelihood is equivalent to squared loss

        train_best = optimizer(loss_best)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	x = []
	y = []
	for i in range(sample_num):
		for t in range(model.T):
			x.append(sample_set[i]['trajectory'][t])
			y.append(sample_set[i]['action'][t])
        x = np.asarray(x,dtype='float32')
        y = np.asarray(y,dtype='float32')
        print(x.shape)
        print(y.shape)
        for step in range(10000):
        	_,loss_value = sess.run([train_best,loss_best], feed_dict={x_best:x, label_best:y})
 		if step % 1000 == 0:
			print(loss_value)
	#print sess.run(W2)
	#print sess.run(b2)
        
        policy_num += 1


	#line 4 Build initial sample set S
        print "process 3 Build initial sample set"
	for j in range(N_sample_init):
		trajectory_temp = []
		action_temp = []
                cost_temp = []
		trj = model.set_x_init()
		for i in range(model.T):
			next_policy_mean = sess.run(y_best, feed_dict={x_best:np.expand_dims(trj,axis=0)})
			next_policy = np.random.multivariate_normal(next_policy_mean[0],policy_var*np.eye(model.nj))
        		trajectory_temp.append(trj)
			action_temp.append(next_policy)

			cost_temp.append(model.instaneous_cost(trj,next_policy,i,0))
                        trj = model.plant_dyn(trj,next_policy,0,0)

		sample_temp = {
                'trajectory':trajectory_temp,
                'action':action_temp,
                'cost':cost_temp,
                'index':policy_num
                }
                sample_set.append(sample_temp)
                sample_num += 1
      
	print ('process 4 Initialize finished')

	print "test 3 ", model.q_init
        print "sample_num : ", sample_num
        print "policy num : ", policy_num


	#line 5 GPS start!!!!! wow! LOL!
        print "process 5 GPS start"
        K = 40
        w_reg = 1e-4
        cost_prev = -1
        
        ros_agent = ROS_connection()	
        
	save_result = open("log/gps_result.txt",'a')
	save_result.write("GPS_start\n")

	#####FOR TEST######
	#calculate final cost
        print "###TEST Execute###"
        trj = model.set_x_init()
        ros_agent.move_arm_once(trj)
        cost_final = 0
        cost_temp = []
	for i in range(model.T):
                print "GPS test ",i
                next_policy = sess.run(y_best, feed_dict={x_best:np.expand_dims(trj,axis=0)})
		print(next_policy)
                trj= model.plant_dyn(trj,next_policy[0],0,0)
                trj_temp = ros_agent.move_arm_once(trj)
		for j in range(model.nj):
			trj[j] = trj_temp[j]
		cost_final += model.instaneous_cost(trj,next_policy[0],i,0)
		cost_temp.append(cost_final)

        #print(cost_temp)
        
	for k in range(K):
                print "GPS iter ",k
		save_result.write("GPS_iter %d\n"%k)
                cost_next = 0
                cost_temp = []
                trj = model.set_x_init()
                trajectory_temp = [np.zeros(model.nj)]
                policy_temp = []


                #line 6 Choose current sample set
		print "Choose sample set"
		ss_x = []
		ss_y = []
		gps_sample_set = []
		for i in range(sample_num):
			for t in range(model.T):
				ss_x.append(sample_set[i]['trajectory'][t])
				ss_y.append(sample_set[i]['control'][t])
			
		ss_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node,diag_stdev=np.ones(model.nj).astype(np.float32))
		ss_imp_serial = sess.run(ss_dist.pdf(y_temp), feed_dict={x_temp:ss_x, y_temp:ss_y})
		ss_imp = []
		for i in range(sample_num):
			ss_temp = 0
			ss_pb = 1
			for t in range(model.T):
				ss_pb *= ss_imp_serial[i*model.T+t]
				ss_temp += ss_pb
			ss_imp.append(ss_temp)

		sample_num_gps = min(flag_end_sample, N_sample_max)
		if sample_num > N_sample_max and flag_rand == 0:		
			ss_imp_temp = ss_imp
			ss_imp_temp.sort()
			ss_imp_limit = ss_imp_temp[-N_sample_max]
			for i in range(sample_num):
				if ss_imp[i] >= ss_imp_limit:
                			gps_sample_set.append(sample_dict[i])
		elif flag_rand == 1:
			for i in range(sample_num_gps):
				gps_sample_set.append(sample_dict[rd.randrange(0,sample_num)])
				flag_rand = 0
		else:
			for i in range(sample_num):
				gps_sample_set.append(sample_dict[i])


		#line 7 Optimize theta
		#calculate importance probability
		ss_x = []
		ss_y = []
		for i in range(sample_num_gps):
			for t in range(model.T):
				ss_x.append(gps_sample_set[i]['trajectory'][t])
				ss_y.append(gps_sample_set[i]['control'][t])
		ss_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node,diag_stdev=np.ones(model.nj).astype(np.float32))
		ss_imp_serial = sess.run(ss_dist.pdf(y_temp), feed_dict={x_temp:ss_x, y_temp:ss_y})
		ss_imp = []
		for i in range(sample_num_gps):
			ss_temp = 0
			ss_pb = 1
			for t in range(model.T):
				ss_pb *= ss_imp_serial[i*model.T+t]
				ss_temp += ss_pb
				ss_imp.append(ss_temp)
		
		for i in range(sample_num_gps):
	                imp_q_temp = []
			trj = gps_sample_set[i]['trajectory']
			next_policy = gps_sample_set[i]['control']
			for t in range(model.T):
	                        pb_temp = 0
				for j in range(policy_num-1):
					x_i = policy_set[j]['x_array_opt']
					u_i = policy_set[j]['u_array_opt'] 
					K_i = policy_set[j]['K_array_opt']
					Cov = policy_set[j]['Q_array_opt']
					pb_temp += mul_normal.pdf(next_policy[t], u_i[t]+K_i[t].dot(trj[t]-x_i[t]),Cov[t])
				pb_temp += ss_imp[i*model.T+t]
				pb_temp /= policy_num
				if len(imp_q_temp) == 0:
	        	                imp_q_temp.append(pb_temp)
	                        else:
	                                imp_q_temp.append(imp_q_temp[-1]*pb_temp)
			sample_set[i]['imp_q'] = imp_q_temp

		#optimize parameter theta
                print "Optimize parameter"

                #z_t = np.zeros(model.T)
                i_rate_temp = np.ones(sample_num)
                i_rate_sel = np.zeros(policy_num)

		opt = tf.train.GradientDescentOptimizer(0.5)

		#calculate gradient
		"""
		print "Compute gradients of policy"
                #v, g = opt.compute_gradients(y_node_new, [W1_new, W2_new])
		print "Compute gradients of z_t and J"
		loss_grad = 0
		z = []
		J = []
		dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node_new,diag_stdev=np.ones(model.nj).astype(np.float32))
		i_temp = dist.pdf(y_temp)
		for i in range(model.T):
	        	exp_temp = 0
        	        z_t = 0
			print "time ",i
			for j in range(sample_num):
				z_t += i_temp[i*sample_num+j]/sample_set[j]['imp_q'][i]
				exp_temp += i_temp[i*sample_num+j]/sample_set[j]['imp_q'][i]*sample_dict[j]['cost'][i]
			z.append(z_t)		
        	        J.append(exp_temp/z_t + w_reg*tf.log(z_t))

		print "Compute gradients of loss function"
		for t in range(model.T):
			print "time ",t
			for j in range(sample_num):
				exp_temp = 0
				for i in range(t,model.T):
					exp_temp += i_temp[i*sample_num+j]/sample_set[j]['imp_q'][i]/z[i]*(sample_dict[j]['cost'][i]-J[i]+w_reg)
				for i in range(model.nj):
					g, v = opt.compute_gradients(y_node_new[t*sample_num+j][i], [W1_new, W2_new])
					loss_grad += g*(y_temp[t*sample_num+j][i]-y_node_new[t*flag_end_sample+j][i])*exp_temp
		"""
		loss_new = 0
		
                #define loss function
		dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node_new,diag_stdev=np.ones(model.nj).astype(np.float32))
		i_temp = dist.pdf(y_temp)
		print "Define loss function"
		for i in range (model.T):
                        exp_temp = 0
                        z_t = 0
			print "time ",i
			for j in range(sample_num_gps):
				z_t += i_temp[i*sample_num_gps+j]/gps_sample_set[j]['imp_q'][i]
				exp_temp += i_temp[i*sample_num_gps+j]/gps_sample_set[j]['imp_q'][i]*gps_sample_dict[j]['cost'][i]
                        loss_new += exp_temp/z_t + w_reg*tf.log(z_t)
		
		#print "Apply gradients"		
		#train = opt.apply_gradients(loss_grad)
		train = opt.minimize(loss_new)
		
                #execute Optimization
		x = []
		y = []
		for t in range(model.T):
			for i in range(sample_num_gps):
				x.append(gps_sample_set[i]['trajectory'][t])
				y.append(gps_sample_set[i]['control'][t])
		#method 1
		print "Pre-training method 1"
		W1_new = W1
		b1_new = b1
		W2_new = W2
		b2_new = b2
		#method 2
		#print "Pre-training method 2"
		#pre_train = tf.train.GradientDescentOptimizer(0.5).minimize(tf.reduce_mean(tf.square(y_temp-y_node_new)))
		#sess.run(pre_train, feed_dict={x_temp:x, y_temp:y})

		print "Start training"
		sess.run(train, feed_dict={x_temp:x, y_temp:y})
		print sess.run(W1_new)
		print sess.run(b1_new)
	

                #line 8 Append samples to current sample set S
                print "Append new samples"
                for j in range(N_sample_init):
			trajectory_temp = []
			control_temp = []
                        imp_q_temp = []
                        cost_temp = []
        		trj = [np.zeros(model.nj)]
        		#print "init ", trj
        		for i in range(model.T):
        			next_policy_mean = sess.run(y_eval, feed_dict={x_eval:trj})
        			next_policy = np.random.multivariate_normal(next_policy_mean[0],np.eye(model.nj))
				imp_q_temp.append(1.0)
                        
                		trajectory_temp.append(trj[0])
        			control_temp.append(next_policy)
        			cost_temp.append(model.instaneous_cost(trj[0],next_policy,i,0))
				trj[0] += next_policy #dynamics

        			#print "time ", i, "result", trj[0]
        
        		sample_temp = {
                        'trajectory':trajectory_temp,
                        'control':control_temp,
                        'imp_q':imp_q_temp,
                        'cost':cost_temp,
                        'index':policy_num
                        }
                        sample_set[sample_num] = sample_temp
                        sample_num += 1


                #line 10 Estimate the costs of prev and next parameters
                print "Execute and estimate the cost"
                #execute and calculate cost
		ros_agent.move_arm_once(np.zeros(model.nj))
                trj = [np.zeros(model.nj)]
                cost_temp = []
                cost_next = 0
                for i in range(model.T):
                        next_policy = sess.run(y_eval_new, feed_dict={x_eval_new:trj})
                        trj[0] += next_policy[0]
                        trj_temp = ros_agent.move_arm_once(trj[0])
			for j in range(model.nj):
				trj[0][j] = trj_temp[j]
                        cost_next += model.instaneous_cost(trj[0],next_policy[0],i,0)
		        cost_temp.append(cost_next)

		save_result.write("cost_prev ")
		save_result.write(str(cost_prev))
		save_result.write("  cost_next ")
		save_result.write(str(cost_next))
		save_result.write("\n")
                
                #line 11 Compare and change sample
                if cost_prev == -1 or cost_prev > cost_next:
                        cost_prev = cost_next
                        W1 = W1_new
                        W2 = W2_new
			b1 = b1_new
			b2 = b2_new
			if w_reg > 1e-6:
	                        w_reg/=10
                        print "Update the parameter"
			print sess.run(W1)
			print sess.run(b1)
                else:
			if w_reg < 1e-2:
	                        w_reg*=10
                        print "Keep the parameter"
			print w_reg
			print "cost_prev ",cost_prev
			print "cost_next ",cost_next
			print sess.run(W1)
			print sess.run(W1_new)
			flag_rand = 1


        print "GPS finished"
	save_result.close()

	#calculate final cost
        print "###Final Execute###"
	ros_agent.move_arm_once(np.zeros(model.nj))
        trj = [np.zeros(model.nj)]
        cost_final = 0
        cost_temp = []
        for i in range(model.T):
                next_policy = sess.run(y_eval, feed_dict={x_eval:trj})
                trj[0] += next_policy[0]
                trj_temp = ros_agent.move_arm_once(trj[0])
		for j in range(model.nj):
			trj[0][j] = trj_temp[j]
		cost_final += model.instaneous_cost(trj[0],next_policy[0],i,0)
		cost_temp.append(cost_final)        
	
        #ros_agent.move_arm(model.T, sample_trajectory[sample_num-1])
	
	print "final position"
	print model.fin_position
	print "result position"
	print model.getPosition(trj[0])

