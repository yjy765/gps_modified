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
import copy

class ROS_connection():
	def __init__(self,model):
		self.model = model
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
                        #print "callback"	
	        	#print([self.q_status])
                        self.flag_callback = 0
		return

	def move_arm(self, T, trj):
		for i in range(T):
			for j in range(0,7):
				self.pub[j].publish(trj[i][j])			
			self.r.sleep()
			#print("ros signal sent")
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
		#print("ros signal sent_once")
     	        #print(target)
		return self.model.JntToNp(self.q_status)


def establish_network(x,W1,b1,W2,b2):
	h = tf.nn.tanh(tf.matmul(x,W1))+b1
	y = tf.matmul(h,W2)+b2
        return y

def optimizer(loss,sess=None):
        global_step = tf.Variable(0,trainable= False)
        learningRate = tf.train.exponential_decay(2.,global_step,100,0.95)
	train = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
	if not sess == None:
		sess.run(global_step.initializer)
        return train

def get_q_prob(policy_set,trj,action,ddp_num):
        log_prob = 0
	prob = 0
	for i in range(ddp_num):
		x_i = policy_set[i]['x_array_opt']
		u_i = policy_set[i]['u_array_opt'] 
		K_i = policy_set[i]['K_array_opt']
	        k_i = policy_set[i]['k_array_opt']
		Cov = policy_set[i]['Q_array_opt']
		for t in range(model.T):
			log_prob +=mul_normal.logpdf(action[t],u_i[t]+k_i[t]+K_i[t].dot(trj[t]-x_i[t]),Cov[t])
		prob += math.exp(log_prob)/ddp_num
	return prob

def get_q_best_prob(policy_set,pi_prob,trj,action,ddp_num,sample_num,T):
	log_policy_prob = []
 	for i in range(ddp_num+1):
		if i < ddp_num :
			log_policy_prob.append([])
			x_i = policy_set[i]['x_array_opt']
			u_i = policy_set[i]['u_array_opt']
			K_i = policy_set[i]['K_array_opt']
			k_i = policy_set[i]['k_array_opt']
			Cov = policy_set[i]['Q_array_opt']
			for j in range(sample_num):
				log_policy_prob[i].append([])
				for t in range(model.T):
					log_policy_prob[i][j].append([])
					if t==0:
						log_policy_prob[i][j][t] = mul_normal.logpdf(action[j*T+t],u_i[t]+k_i[t]+K_i[t].dot(trj[j*T+t]-x_i[t]),Cov[t])
					else:
						log_policy_prob[i][j][t] = log_policy_prob[i][j][t-1] + mul_normal.logpdf(action[j*T+t],u_i[t]+k_i[t]+K_i[t].dot(trj[j*T+t]-x_i[t]),Cov[t])
		else:
			log_policy_prob.append([])
			for j in range(sample_num):
				log_policy_prob[i].append([])
				for t in range(model.T):
					if t==0:
						log_policy_prob[i][j].append(pi_prob[j*T+t])
					else:
						log_policy_prob[i][j].append(log_policy_prob[i][j][t-1] + pi_prob[j*T+t])
	log_prob = []
	for j in range(sample_num):
		log_prob.append([])
		for t in range(model.T):
			prob = 0
			for i in range(ddp_num+1):
				prob += math.exp(log_policy_prob[i][j][t]) / (ddp_num+1)
			if t == 0:
				log_prob[j].append(math.log(prob))
			else:
				log_prob[j].append(math.log(prob) + log_prob[j][t-1])
	return log_prob

	

if __name__ == '__main__':

	model = Manipulator_X(T=10, weight=[3.0,0.5],verbose=False)
	model.build_ilqr_problem()

	eps = 1e-100
        policy_var = 1e-2
	ddp_sample = 150  #number of total sample
	sample_num_max = 200
	N_sample_init = 10 #number of sample from each ddp solution
	N_ddp_sol = 19  #number of ddp solution
	N_sol = 20
        K_iter = 30
        w_reg = 1e-4
        cost_prev = -1
        policy_num = 0
	sample_num = 0 #last index of sample
	flag_rand = 0 #randomly pict sample

	policy_set = [None]*N_sol    #set of solution ddp & nn
	sample_set = [None]*ddp_sample	#set of samples from each solution


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
				print(l)
                                assert(np.sum(l>0) == len(l))
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
	x_best = tf.placeholder(tf.float64,[None, model.nj],name="x_best")
        x_new1 = tf.placeholder(tf.float64,[None,model.nj],name="x_new1")
        x_new2 = tf.placeholder(tf.float64,[None,model.nj],name="x_new2")

        W1_best = tf.Variable(tf.truncated_normal([model.nj,N_hidden_node],stddev=1.0/math.sqrt(float(model.nj)),name="W1_best",dtype=tf.float64))
	b1_best = tf.Variable(tf.zeros([N_hidden_node],dtype=tf.float64), name="b1_best")
	W2_best = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(N_hidden_node)),dtype=tf.float64), name="W2_best")
	b2_best = tf.Variable(tf.zeros([model.nj],dtype=tf.float64), name="b2_best")

	W1_temp = tf.Variable(tf.truncated_normal([model.nj, N_hidden_node], stddev=1.0/math.sqrt(float(model.nj)),dtype=tf.float64),name="W1_temp")
	b1_temp = tf.Variable(tf.zeros([N_hidden_node],dtype=tf.float64),name="b1_temp")
	W2_temp = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(model.nj)),dtype=tf.float64), name="W2_temp")
	b2_temp = tf.Variable(tf.zeros([model.nj],dtype=tf.float64), name="b2_temp")

	W1_new1 = tf.Variable(tf.truncated_normal([model.nj, N_hidden_node], stddev=1.0/math.sqrt(float(model.nj)),dtype=tf.float64), name="W1_new1")
	b1_new1 = tf.Variable(tf.zeros([ N_hidden_node],dtype=tf.float64), name="b1_new1")
	W2_new1 = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj], stddev=1.0/math.sqrt(float(N_hidden_node)),dtype=tf.float64), name="W2_new1")
	b2_new1 = tf.Variable(tf.zeros([model.nj],dtype=tf.float64), name="b2_new1")

	copy_W1_new1 = W1_new1.assign(W1_best)
	copy_b1_new1 = b1_new1.assign(b1_best)
	copy_W2_new1 = W2_new1.assign(W2_best)
	copy_b2_new1 = b2_new1.assign(b2_best)

	copy_W1_temp = W1_temp.assign(W1_new1)
	copy_b1_temp = b1_temp.assign(b1_new1)
	copy_W2_temp = W2_temp.assign(W2_new1)
	copy_b2_temp = b2_temp.assign(b2_new1)

	copy_W1_best = W1_best.assign(W1_temp)
	copy_b1_best = b1_best.assign(b1_temp)
	copy_W2_best = W2_best.assign(W2_temp)
	copy_b2_best = b2_best.assign(b2_temp)

	W1_new2 = tf.Variable(tf.truncated_normal([model.nj, N_hidden_node], stddev=1.0/math.sqrt(float(model.nj)),dtype=tf.float64), name="W1_new2")
	b1_new2 = tf.Variable(tf.zeros([ N_hidden_node],dtype=tf.float64), name="b1_new2")
	W2_new2 = tf.Variable(tf.truncated_normal([N_hidden_node, model.nj],dtype=tf.float64, stddev=1.0/math.sqrt(float(N_hidden_node))), name="W2_new2")
	b2_new2 = tf.Variable(tf.zeros([model.nj],dtype=tf.float64), name="b2_new2")


	y_best = establish_network(x_best,W1_best,b1_best,W2_best,b2_best)
        y_new1 = establish_network(x_new1,W1_new1,b1_new1,W2_new1,b2_new1)
        y_new2 = establish_network(x_new2,W1_new2,b1_new2,W2_new2,b2_new2)

	label_best = tf.placeholder(tf.float64,[None, model.nj],name='label_best')
        label_new1 = tf.placeholder(tf.float64,[None, model.nj],name='label_new1')
        label_new2 = tf.placeholder(tf.float64,[None, model.nj],name='label_new2')

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
        x = np.asarray(x,dtype='float64')
        y = np.asarray(y,dtype='float64')
        print(x.shape)
        print(y.shape)
        for step in range(4000):
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
        
        ros_agent = ROS_connection(model)	
        
	save_result = open("log/gps_result.txt",'w')
	save_result.write("GPS_start\n")

      	pi_best = tf.contrib.distributions.MultivariateNormalDiag(mu=y_best,diag_stdev=policy_var*np.ones(model.nj).astype(np.float64))
	pi_new1_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_new1,diag_stdev=policy_var*np.ones(model.nj).astype(np.float64))

	for k in range(K_iter):
		print "+=======================================================+"
                print "GPS iter ",k
		save_result.write("GPS_iter %d\n"%k)
                cost_next = 0
                cost_temp = []
                trj = model.set_x_init()
                trajectory_temp = [np.zeros(model.nj)]
                policy_temp = []


                #line 6 Choose current sample set
		print "Choose sample set"
		sample_x = []
		sample_y = []
		gps_sample_set = []
		if sample_num <= sample_num_max:
			gps_sample_set = copy.deepcopy(sample_set)
		else:
			print "Chosse sub set"
        		for i in range(sample_num):
				for t in range(model.T):
					sample_x.append(sample_set[i]['trajectory'][t])
					sample_y.append(sample_set[i]['action'][t])	
			pi_prob = sess.run(pi_best.log_pdf(label_best), feed_dict={x_best:sample_x, label_best:sample_y})
			sample_imp = []
			for i in range(ddp_sample,sample_num):
				sample_pi_prob = 0
				for t in range(model.T):
					sample_pi_prob += pi_prob[(i-ddp_sample)*model.T+t]
				sample_q = get_q_prob(policy_set,sample_set[i]['trajectory'],sample_set[i]['action'],N_ddp_sol)
				#print(sample_pi_prob)
				#print(math.log(sample_q))
				sample_imp.append(sample_pi_prob - math.log(sample_q))

			if flag_rand == 0:
				for i in range(ddp_sample):
					gps_sample_set.append(sample_set[i])
				sample_imp_temp = sample_imp
				sample_imp_temp.sort()
				sample_imp_limit = sample_imp_temp[-(sample_num_max-ddp_sample)]
				for i in range(sample_num-ddp_sample):
					if sample_imp[i] >= sample_imp_limit:
	                			gps_sample_set.append(sample_set[i+ddp_sample])
			elif flag_rand == 1:
				for i in range(gps_sample_num):
					gps_sample_set.append(sample_set[rd.randrange(0,sample_num)])

		gps_sample_num = len(gps_sample_set)


		#line 7 Optimize theta

		# make loss and gradients explicitly

		'''
		# calculate importance probability for every iteration for optimizing theta
		#calculate importance probability
		for iter_opt in range(1000):
			sample_x = []
			sample_y = []
			for i in range(gps_sample_num):
				for t in range(model.T):
					sample_x.append(gps_sample_set[i]['trajectory'][t])
					sample_y.append(gps_sample_set[i]['action'][t])
			pi_new1_prob = sess.run(pi_new1_dist.log_pdf(label_new1), feed_dict={x_new1:sample_x, label_new1:sample_y})
			pi_best_prob = sess.run(pi_best.log_pdf(label_best), feed_dict={x_best:sample_x, label_best:sample_y})

			q_best_prob = get_q_best_prob(policy_set,pi_best_prob,sample_x,sample_y,N_ddp_sol,gps_sample_num,model.T)
			sample_log_imp = []
			for i in range(gps_sample_num):
				sample_log_imp.append([])
				for t in range(model.T):
					if t == 0:
						sample_log_imp[i].append(pi_new1_prob[i*model.T+t]-q_best_prob[i][t])
					else:
						sample_log_imp[i].append(sample_log_imp[i][t-1] + pi_new1_prob[i*model.T+t]-q_best_prob[i][t])

			#calculate Z_t first and then J_t
			Z_t = []
			for t in range(model.T):
				for i in range(gps_sample_num):
					if i == 0:
						Z_t.append(math.exp(sample_log_imp[i][t]))
					else:
						Z_t[t] += math.exp(sample_log_imp[i][t])
			J_t = []
			for t in range(model.T):
				for i in range(gps_sample_num):
					if i == 0:
						J_t.append(math.exp(sample_log_imp[i][t] + math.log(gps_sample_set[i]['cost'][t])))
					else:
						J_t[t] += math.exp(sample_log_imp[i][t] + math.log(gps_sample_set[i]['cost'][t]))
				J_t[t] /= (Z_t[t]+eps)



                #z_t = np.zeros(model.T)
                i_rate_temp = np.ones(sample_num)
                i_rate_sel = np.zeros(policy_num)

		'''

		
                #define loss function
		loss_new1 = 0
		sample_x = []
		sample_y = []
		min_cost = -1e20
		min_cost_idx = 0
		for i in range(gps_sample_num):
			for t in range(model.T):
				sample_x.append(gps_sample_set[i]['trajectory'][t])
				sample_y.append(gps_sample_set[i]['action'][t])
			if min_cost > gps_sample_set[i]['cost']:
				min_cost = gps_sample_set[i]['cost']
				min_cost_idx = i
				
		pi_best_prob = sess.run(pi_best.log_pdf(label_best), feed_dict={x_best:sample_x, label_best:sample_y})
		q_best_prob = get_q_best_prob(policy_set,pi_best_prob,sample_x,sample_y,N_ddp_sol,gps_sample_num,model.T)
		gps_sample_cost = []
		for j in range(gps_sample_num):
			gps_sample_cost.append([])
			for t in range(model.T):
				gps_sample_cost[j].append(gps_sample_set[j]['cost'][t])
		gps_sample_cost = np.asarray(gps_sample_cost)
		print "Define loss function"
		log_pi_new1 = tf.reshape(pi_new1_dist.log_pdf(label_new1),[gps_sample_num,model.T])
		pi_new1_temp = tf.cumsum(log_pi_new1,axis=1)
		z_t_sample = tf.exp(pi_new1_temp - q_best_prob)
		r_t_sample = tf.multiply(tf.exp(pi_new1_temp - q_best_prob),gps_sample_cost)
		z_t = tf.reduce_sum(z_t_sample,axis=0)
		r_t = tf.reduce_sum(r_t_sample,axis=0)
		loss_new1 = tf.reduce_sum( tf.div(r_t,z_t+eps) + w_reg*tf.log(z_t+eps))
		
		#train_new1 = optimizer(loss_new1,sess)	
		train_new1 = tf.train.GradientDescentOptimizer(0.00001).minimize(loss_new1)
                #execute Optimization

		#method 1
		print "training start from current best policy"
		sess.run([copy_W1_new1,copy_b1_new1,copy_W2_new1,copy_b2_new1])
		feed_dict = {x_new1:sample_x,x_best:sample_x,label_new1:sample_y,label_best:sample_y}

		#method 2
		#print "training start from the policy maximizing the highest reward"
		#loss_new2 = tf.reduce_mean(tf.square(y_new2 - label_new2))
		#train_new2 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_new2)
		#feed_dict = {x_new2:gps_sample_set[min_cost_idx]['trajectory'],label_new2:gps_sample_set[min_cost_idx]['action']}
		#for step in range(4000):
		#	_,loss_value = sess.run([train_new2,loss_new2],feed_dict=feed_dict)
		#	if step % 1000 == 0:
		#		print(loss_value)	
		#sess.run(pre_train, feed_dict={x_temp:x, y_temp:y})
		print "Start training"
		for step in range(3000):
			_,loss_value,z_t_value = sess.run([train_new1,loss_new1,z_t],feed_dict={x_new1:sample_x,x_best:sample_x,label_new1:sample_y,label_best:sample_y})
			if step % 200 == 0:
				print(loss_value)
				#print(z_t_value)
		

                #line 8 Append samples to current sample set S
                print "Append new samples"
                for j in range(N_sample_init):
			trajectory_temp = []
			action_temp = []
                        cost_temp = []
        		trj = model.set_x_init()
        		for i in range(model.T):
        			next_policy_mean = sess.run(y_new1, feed_dict={x_new1:np.expand_dims(trj,axis=0)})
        			next_policy = np.random.multivariate_normal(next_policy_mean[0],policy_var*np.eye(model.nj))                   
                		trajectory_temp.append(trj)
        			action_temp.append(next_policy)
        			cost_temp.append(model.instaneous_cost(trj,next_policy,i,0))
				trj = model.plant_dyn(trj,next_policy,0,0)

        			#print "time ", i, "result", trj[0]
        
        		sample_temp = {
                        'trajectory':trajectory_temp,
                        'action':action_temp,
                        'cost':cost_temp,
                        'index':policy_num
                        }
                        sample_set.append(sample_temp)
			gps_sample_set.append(sample_temp)
			gps_sample_num += 1
                        sample_num += 1
		policy_num += 1

                #line 10 Estimate the costs of prev and next parameters
                print "Execute and estimate the cost"
                #estimate the values of the trained_network
			
		sample_x = []
		sample_y = []
		for i in range(gps_sample_num):
			for t in range(model.T):
				sample_x.append(gps_sample_set[i]['trajectory'][t])
				sample_y.append(gps_sample_set[i]['action'][t])
		pi_best_prob_eval = sess.run(pi_best.log_pdf(label_best), feed_dict={x_best:sample_x, label_best:sample_y})
		q_best_prob_eval = get_q_best_prob(policy_set,pi_best_prob_eval,sample_x,sample_y,N_ddp_sol,gps_sample_num,model.T)
		gps_sample_cost = []
		for j in range(gps_sample_num):
			gps_sample_cost.append([])
			for t in range(model.T):
				gps_sample_cost[j].append(gps_sample_set[j]['cost'][t])
		gps_sample_cost = np.asarray(gps_sample_cost)
		print "Define eval network"
		pi_new1_temp_eval = tf.cumsum(tf.reshape(pi_new1_dist.log_pdf(label_new1),[gps_sample_num,model.T]),axis=1)
		z_t_sample_eval = tf.exp(pi_new1_temp_eval - q_best_prob_eval)
		r_t_sample_eval = tf.multiply(tf.exp(pi_new1_temp_eval - q_best_prob_eval),gps_sample_cost)
		z_t_eval = tf.reduce_sum(z_t_sample_eval,axis=0) + eps
		r_t_eval = tf.reduce_sum(r_t_sample_eval,axis=0)
		loss_new1_eval = tf.reduce_sum( tf.div(r_t_eval,z_t_eval) + w_reg*tf.log(z_t_eval))

		feed_dict = {x_new1:sample_x,x_best:sample_x,label_new1:sample_y,label_best:sample_y}
		current_cost = sess.run(loss_new1_eval,feed_dict=feed_dict)

		sess.run([copy_W1_temp,copy_b1_temp,copy_W2_temp,copy_b2_temp])
		sess.run([copy_W1_new1,copy_b1_new1,copy_W2_new1,copy_b2_new1])
		best_cost = sess.run(loss_new1_eval,feed_dict=feed_dict)

		save_result.write("cost_best ")
		save_result.write(str(best_cost))
		save_result.write("  cost_trained ")
		save_result.write(str(current_cost))
		save_result.write("\n")
                
                #line 11 Compare and change sample
                if best_cost > current_cost:
                        sess.run([copy_W1_best,copy_b1_best,copy_W2_best,copy_b2_best])
			if w_reg > 1e-6:
	                        w_reg/=10
                        print "Update the parameter!!!!"
			print "cost_best", best_cost
			print "cost_trained",current_cost
                else:
			if w_reg < 1e-2:
	                        w_reg*=10
                        print "Keep the parameter"
			print w_reg
			print "cost_best", best_cost
			print "cost_trained",current_cost
			#flag_rand = 1  sample from pi_best


        print "GPS finished"
	save_result.close()

	#calculate final cost
        print "###Final Execute###"
        trj = model.set_x_init()
	ros_agent.move_arm_once(trj)
        cost_final = 0
        cost_temp = []
	learned_policy = []
        for t in range(model.T):
		trj = np.asarray(trj)
		print(type(trj))
		print(trj.shape)
		print(t)
                next_policy = sess.run(y_best, feed_dict={x_best:np.expand_dims(trj,axis=0)})[0]
		learned_policy.append(next_policy)
                trj = model.plant_dyn(trj,next_policy,0,0)
                trj = ros_agent.move_arm_once(trj)
		print('return trj ',trj)
		cost_final += model.instaneous_cost(trj,next_policy,t,0)
		cost_temp.append(cost_final)        
	
        #ros_agent.move_arm(model.T, sample_trajectory[sample_num-1])

	import pickle
	with open('policy.pkl','wb') as f:
		pickle.dump(learned_policy,f,pickle.HIGHEST_PROTOCOL)	
	print "cost"
	print cost_temp	
	print "final position"
	print model.fin_position
	print "result position"
	print model.getPosition(trj)

