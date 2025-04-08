import numpy as np
from random import randrange
import math
import utils

import matplotlib.pyplot as plt

theta_target = math.pi/4 
timestep = 800
done = False


class RA():
    def __init__(self):
        self.N_s = 100 # total number of neurons
        self.v = 1
        self.h_0 = 0
        self.h_b = 0
        self.v_0 = 10
        self.h_i = None
        self.beta = 1
        self.indx = None
        self.std_dev = 2*math.pi/self.N_s
        self.alpha = np.mod(np.arange(self.N_s) * self.std_dev, 2 * np.pi)
        self.sigma = np.random.choice([1,-1], size=self.N_s)  
        self.H = np.zeros([2,1])

ring = RA()
t_0 = 50
activity_log = np.zeros((t_0, ring.N_s))


for k in range (t_0):
    for t in range (ring.N_s):
        ring.H = np.zeros([2,1])
        ring.indx = np.random.randint(0,ring.N_s,size=1)
        utils.external_stimuli(ring, theta_target)
        utils.hamiltonian(ring)
        #activity_log[t] = (ring.sigma + 1) / 2

        delta_H = ring.H[1] - ring.H[0]
        if delta_H < 0:
            ring.sigma[ring.indx] *= -1
        else:
            prob = np.exp(-ring.beta*delta_H)
            if np.random.rand() < prob:
                ring.sigma[ring.indx] *= -1
    activity_log[k] = (ring.sigma + 1) / 2
  


plt.figure(figsize=(8, 4))
plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower')
plt.xlabel("Time step")
plt.ylabel("Neuron index")
plt.title("Ring attractor activity over time")
plt.colorbar(label='Activity (1 = active)')
plt.tight_layout()
plt.show()

    
    

    









