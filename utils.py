import numpy as np
from random import randrange
import math



def synaptic_connectivity(alpha_i,alpha_j,v):
    return math.cos(math.pi*(abs(alpha_i-alpha_j)/math.pi)**v)

def external_stimuli(ra, theta_target):
    angle_diff = ra.alpha[ra.indx] - theta_target
    if angle_diff > math.pi:
        angle_diff -= 2*math.pi
    elif angle_diff > -math.pi:
        angle_diff+=2*math.pi

    h_i = ra.h_0/math.sqrt(2*math.pi*ra.std_dev)*math.exp(-(angle_diff)**2/(2*ra.std_dev**2))
    ra.h_i = h_i

def hamiltonian_old(ra):
    ra_dyn_sum_0 = 0
    ra_dyn_sum_1 = 0
    for j in range (ra.N_s):
        if j == ra.indx:
            continue
        else:
            ra_dyn_sum_0 += math.cos(math.pi*(abs(ra.alpha[ra.indx]-ra.alpha[j])/math.pi)**ra.v)*ra.sigma[ra.indx]*ra.sigma[j] 
            ra_dyn_sum_1 += math.cos(math.pi*(abs(ra.alpha[ra.indx]-ra.alpha[j])/math.pi)**ra.v)*ra.sigma[ra.indx]*-1*ra.sigma[j] 
    
    inp_0 = ra.h_i * ra.sigma[ra.indx] 
    leak_0 = ra.h_b*ra.sigma[ra.indx]
    inp_1 = ra.h_i * ra.sigma[ra.indx]*-1
    leak_1 = ra.h_b*ra.sigma[ra.indx]*-1

    hamil_0 = ra_dyn_sum_0/(ra.N_s-1) + inp_0 + leak_0
    hamil_1 = ra_dyn_sum_1/(ra.N_s-1) + inp_1 + leak_1

    ra.H.append(-1*hamil_0)
    ra.H.append(-1*hamil_1)

import numpy as np

def hamiltonian(ra):
    i = ra.indx
    alpha_i = ra.alpha[i]

    angle_diffs = np.abs(alpha_i - ra.alpha)

    kernel = np.cos(np.pi * (angle_diffs / np.pi) ** ra.v)
    kernel[i] = 0  #no self interaction

    interaction_sum = np.dot(kernel, ra.sigma)

    spin_flip = np.array([ra.sigma[i], -ra.sigma[i]])

    interaction_energy = (interaction_sum * spin_flip) / (ra.N_s - 1)

    input_energy = ra.h_i * spin_flip
    leak_energy = ra.h_b * spin_flip

    ra.H = -1 * (interaction_energy + input_energy + leak_energy)


    




   
