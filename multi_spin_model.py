import numpy as np
from random import randrange
import math
import utils
import live_plot


target = np.array([[0, 50]])
#target = np.array([[10, 10], [10, -10], [10, 0]]) 
bump_log = []
#theta_target = np.arctan2(target_y, target_x) % (2 * np.pi)

class RA():
    def __init__(self):
        self.N_s = 100 # total number of neurons
        self.M = 10 #spins in each neuron
        self.v = 0.5
        self.h_0 = 0
        self.h_b = 0
        self.v_0 = 10
        self.h_i = None
        self.beta = 400
        self.indx = None
        self.subindx = None
        self.std_dev = 2*np.pi/self.N_s
        #self.alpha = np.arange(self.N_s) * (2 * np.pi / self.N_s)
        self.alpha = np.linspace(-np.pi, np.pi, self.N_s, endpoint=False)
        self.sigma = np.random.choice([1, -1], size=(self.N_s, self.M))
        self.H = np.zeros(2)
        self.allocentric = True
        self.pos = np.zeros(2)
        self.heading = 0
        #np.random.shuffle(self.sigma)


ring = RA()
t_0 = 500
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
    
    utils.external_stimuli_ms(ring, theta_target)

    for t in range (50*ring.N_s):
        ring.H = np.zeros(2)
        ring.indx = np.random.randint(0,ring.N_s)
        ring.subindx = np.random.randint(0,ring.M)
        
        utils.hamiltonian_multi_spin(ring)

        delta_H = ring.H[1] - ring.H[0]
        if delta_H < 0:
            ring.sigma[ring.indx, ring.subindx] *= -1
        else:
            prob = np.exp(-ring.beta*delta_H)
            if np.random.rand() < prob:
                ring.sigma[ring.indx,ring.subindx] *= -1
                
    #active_indices = np.where(ring.sigma == 1)[0]
    sigma_bar = np.mean(ring.sigma, axis=1)  
    bump_angle = np.angle(np.sum(sigma_bar * np.exp(1j * ring.alpha)))
    print(bump_angle,k)
    bump_log.append(bump_angle)
    if ring.allocentric:
        dx = ring.v_0 * np.cos(bump_angle)/ring.N_s
        dy = ring.v_0 * np.sin(bump_angle)/ring.N_s
        #ring.pos += np.array([dx, dy])

    #if len(active_indices) > 0:
    #    bump_angle = np.angle(np.sum(np.exp(1j * ring.alpha[ring.sigma == 1])))
    #    print(bump_angle,k)
    #    bump_log.append(bump_angle)
    #    if not ring.allocentric:
    #        ring.heading += bump_angle
    #        ring.heading = (ring.heading) % (2 * np.pi) 
    #        dx = ring.v_0 * np.cos(ring.heading)/ring.N_s
    #        dy = ring.v_0 * np.sin(ring.heading)/ring.N_s
    #        ring.pos += np.array([dx, dy])
#
    #    if ring.allocentric:
    #        dx = ring.v_0 * np.cos(bump_angle)/ring.N_s
    #        dy = ring.v_0 * np.sin(bump_angle)/ring.N_s
    #        ring.pos += np.array([dx, dy])
    #    
    #dist = math.sqrt((target_x-ring.pos[0])**2 + (target_y-ring.pos[1])**2)
    #if dist <= 15:
    #    break

    activity_log[k] = (sigma_bar + 1) / 2
    trajectory[k] = ring.pos

  

utils.plot_neuron_activity_ba(activity_log,ring,bump_log,target_angle_log)
utils.plot_trajectory(trajectory,target,ring)
utils.plot_neuron_activity_w_bump_angle(activity_log, ring, bump_log,target_angle_log)
live_plot.plot_trajectory_live(trajectory,target,ring,bump_log,target_angle_log)