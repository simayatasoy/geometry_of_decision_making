import numpy as np
from random import randrange
import math
import utils


target_x = 20
target_y = 30
#theta_target = np.arctan2(target_y, target_x) % (2 * np.pi)

class RA():
    def __init__(self):
        self.N_s = 100 # total number of neurons
        self.v = 1
        self.h_0 = 0.0025
        self.h_b = 0
        self.v_0 = 10
        self.h_i = None
        self.beta = 20
        self.indx = None
        self.std_dev = 2*np.pi/self.N_s
        #self.alpha = np.mod(np.arange(self.N_s) * self.std_dev, 2 * np.pi)
        #self.alpha = np.mod(np.pi/2 + np.arange(self.N_s) * self.std_dev, 2 * np.pi)
        self.alpha = np.arange(self.N_s) * (2 * np.pi / self.N_s)
        #self.sigma = np.random.choice([1,-1], size=self.N_s)  
        self.sigma = np.array([1] * (self.N_s // 2) + [-1] * (self.N_s - self.N_s // 2))
        self.H = np.zeros(2)
        self.allocentric = True
        self.pos = np.zeros(2)
        self.heading = 0
        np.random.shuffle(self.sigma)


ring = RA()
t_0 = 2000
activity_log = np.zeros((t_0, ring.N_s))


trajectory = np.zeros((t_0, 2))

for k in range (t_0):
    theta_target = np.arctan2(target_y - ring.pos[1], target_x-ring.pos[0]) % (2 * np.pi)
    for t in range (50*ring.N_s):
        ring.H = np.zeros(2)
        ring.indx = np.random.randint(0,ring.N_s)
        utils.external_stimuli(ring, theta_target)
        utils.hamiltonian(ring)

        delta_H = ring.H[1] - ring.H[0]
        if delta_H < 0:
            ring.sigma[ring.indx] *= -1
        else:
            prob = np.exp(-ring.beta*delta_H)
            if np.random.rand() < prob:
                ring.sigma[ring.indx] *= -1
                
    active_indices = np.where(ring.sigma == 1)[0]
    if len(active_indices) > 0:
        bump_angle = np.angle(np.sum(np.exp(1j * ring.alpha[active_indices])))
        if not ring.allocentric:
            ring.heading = bump_angle
        dx = ring.v_0 * np.cos(bump_angle)/ring.N_s
        dy = ring.v_0 * np.sin(bump_angle)/ring.N_s
        ring.pos += np.array([dx, dy])
 

    activity_log[k] = (ring.sigma + 1) / 2
    trajectory[k] = ring.pos
  

utils.plot_neuron_activity(activity_log)
utils.plot_trajectory(trajectory,target_x,target_y)

    
    

    









