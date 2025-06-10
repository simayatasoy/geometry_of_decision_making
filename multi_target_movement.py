import numpy as np
from random import randrange
import math
import utils
import copy
import live_plot


target = np.array([[100,100],[100,-100]])

target = np.array([[-10,10]])
#target = np.array([[10, 10], [10, -10], [10, 0]]) 
bump_log = []
#theta_target = np.arctan2(target_y, target_x) % (2 * np.pi)

class RA():
    def __init__(self):
        self.N_s = 100 #total number of neurons
        self.v = 0.5
        self.h_0 = 0.25
        self.h_b = 0.01
        self.v_0 = 10
        self.h_i = None
        self.beta = 400
        self.indx = None
        self.std_dev = 2*np.pi/self.N_s
        #self.alpha = np.mod(np.arange(self.N_s) * self.std_dev, 2 * np.pi)
        #self.alpha = np.mod(np.pi/2 + np.arange(self.N_s) * self.std_dev, 2 * np.pi)
        #self.alpha = np.arange(self.N_s) * (2 * np.pi / self.N_s)
        self.alpha = np.linspace(-np.pi, np.pi, self.N_s, endpoint=False)

        self.sigma = np.random.choice([1,0], size=self.N_s)  
        #self.sigma = np.array([1] * (self.N_s // 2) + [1] * (self.N_s - self.N_s // 2))
        self.H = np.zeros(2)
        self.allocentric = True
        self.pos = np.zeros(2)
        self.heading = 0
        #np.random.shuffle(self.sigma)


ring = RA()
t_0 = 200
activity_log = np.zeros((t_0, ring.N_s))
target_angle_log = np.zeros((t_0))
h_i_log = np.zeros((t_0, ring.N_s))
trajectory = np.zeros((t_0, 2))

for k in range (t_0):
    diff = target - ring.pos
    diff_x = diff[:,0]
    diff_y = diff[:,1]
    theta_world = np.arctan2(diff_y,diff_x)
    

    if ring.allocentric:
        theta_target = theta_world
        target_angle_log[k] = theta_target
    else:
        #theta_target = theta_world-ring.heading
        theta_target = (theta_world-ring.heading) % (2 * np.pi) 

    utils.external_stimuli_mt(ring, theta_target)
    for t in range (50*ring.N_s):
        ring.H = np.zeros(2)
        ring.indx = np.random.randint(0,ring.N_s)
        
        utils.hamiltonian(ring)

        delta_H = ring.H[1] - ring.H[0]
        if delta_H < 0:
            ring.sigma[ring.indx] = 1 - ring.sigma[ring.indx]
        #else:
        #    prob = np.exp(-ring.beta*delta_H)
        #    if np.random.rand() < prob:
        #        ring.sigma[ring.indx] = 1 - ring.sigma[ring.indx]
                
    active_indices = np.where(ring.sigma == 1)[0]
    if len(active_indices) > 0:
        bump_angle = np.angle(np.sum(np.exp(1j * ring.alpha[ring.sigma == 1])))
        print(bump_angle,k)
        bump_log.append(bump_angle)
        if not ring.allocentric:
            ring.heading += bump_angle
            ring.heading = (ring.heading) % (2 * np.pi) 

            dx = ring.v_0 * np.cos(ring.heading)/ring.N_s
            dy = ring.v_0 * np.sin(ring.heading)/ring.N_s
            ring.pos += np.array([dx, dy])

        if ring.allocentric:
            #i_max = np.argmax(ring.h_i)
            #angle = ring.alpha[i_max]
            dx = ring.v_0 * np.cos(bump_angle)/ring.N_s
            dy = ring.v_0 * np.sin(bump_angle)/ring.N_s
            ring.pos += np.array([dx, dy])
        
    #dist = math.sqrt((target_x-ring.pos[0])**2 + (target_y-ring.pos[1])**2)
    #if dist <= 15:
    #    break
    print(len(active_indices))
    activity_log[k] = ring.sigma.copy()
    trajectory[k] = ring.pos
    h_i_log[k] = ring.h_i.copy()



#print(target_angle_log)

utils.plot_neuron_activity_ba(activity_log,ring,bump_log,target_angle_log)
utils.plot_trajectory(trajectory,target,ring)
utils.plot_neuron_activity_w_bump_angle(activity_log, ring, bump_log,target_angle_log)
live_plot.plot_trajectory_live(trajectory,target,ring,bump_log,target_angle_log)
live_plot.plot_hi_live(h_i_log, ring,target_angle_log,bump_log)