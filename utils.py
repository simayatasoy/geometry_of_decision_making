import numpy as np
from random import randrange
import math
import matplotlib.pyplot as plt

def external_stimuli(ra, theta_target):
    #angle_diff = (ra.alpha[ra.indx] - theta_target + math.pi)%(2*math.pi)-math.pi
    if ra.allocentric:
        angle_diff = (ra.alpha[ra.indx] - theta_target)
        true_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
    else:
        #egocentric_target = (theta_target-ra.heading+np.pi) % (2*np.pi) - np.pi
        #mapped_alpha = (ra.alpha[ra.indx]+ra.heading+np.pi) % (2*np.pi) - np.pi
        true_diff = (ra.alpha[ra.indx] - theta_target + np.pi) % (2*np.pi) - np.pi
        #true_diff = (ra.alpha[ra.indx] - theta_target) % (2*np.pi)
    h_i = ra.h_0/math.sqrt(2*math.pi*ra.std_dev**2)*math.exp(-(true_diff)**2/(2*ra.std_dev**2))
    ra.h_i = h_i


def hamiltonian(ra):

    i = ra.indx
    alpha_i = ra.alpha[i]

    angle_diffs = np.abs(alpha_i - ra.alpha)
    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ra.v))
    kernel[i] = 0  #no self interaction

    interaction_sum = np.dot(kernel, ra.sigma)

    spin_flip = np.array([ra.sigma[i], -ra.sigma[i]])

    interaction_energy = (interaction_sum * spin_flip) / (ra.N_s - 1)

    input_energy = ra.h_i * spin_flip
    leak_energy = ra.h_b * spin_flip

    ra.H = -1 * (interaction_energy + input_energy + leak_energy)

def plot_neuron_activity(activity_log):

    plt.figure(figsize=(8, 4))
    plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower')
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title("Ring attractor activity over time")
    plt.colorbar(label='Activity (1 = active)')
    plt.tight_layout()
    plt.show()

def plot_trajectory(trajectory,target_x,target_y,ra):
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue')
    plt.scatter([target_x], [target_y], color='orange', label='Target')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='End')
    plt.axis('equal')
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    plt.gca().set_aspect('equal', adjustable='box')

    if ra.allocentric:
        plt.title("Agent Trajectory (Allocentric)")
    else:
        plt.title("Agent Trajectory (Egocentric)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




   
