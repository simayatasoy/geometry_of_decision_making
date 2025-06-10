import numpy as np
from random import randrange
import math
import matplotlib.pyplot as plt
import os
import copy

def external_stimuli_mt(ra, theta_target):
    h_i = np.zeros_like(ra.alpha)
    for theta in theta_target:
        angle_diff = (ra.alpha - theta + np.pi) % (2 * np.pi) - np.pi
        h_i += ra.h_0/np.sqrt(2*np.pi*ra.std_dev**2) * np.exp(- (angle_diff ** 2) / (2 * ra.std_dev ** 2))
    ra.h_i = h_i


def external_stimuli_updated(ra, theta_target):
    h_i = np.zeros_like(ra.alpha)
    #for theta in theta_target:
    angle_diff = (ra.alpha - theta_target + np.pi) % (2 * np.pi) - np.pi
    h_i = ra.h_0/np.sqrt(2*np.pi*ra.std_dev**2) * np.exp(- (angle_diff ** 2) / (2 * ra.std_dev ** 2))
    ra.h_i = h_i
    #print(ra.h_i)
    #ra.h_i /= len(theta_target)




def external_stimuli(ra, theta_target):
    #h_i = 0
    angle_diff = (ra.alpha[ra.indx] - theta_target + math.pi)%(2*math.pi)-math.pi
    if ra.allocentric:
        angle_diff = (ra.alpha[ra.indx] - theta_target)
        true_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
    else:
        true_diff = (ra.alpha[ra.indx] - theta_target + np.pi) % (2*np.pi) - np.pi
    #
    #h_i = np.zeros_like(true_diff[0])
    #for diff in true_diff:
    h_i = ra.h_0/np.sqrt(2*math.pi*ra.std_dev**2) * np.exp(- (true_diff ** 2) / (2 * ra.std_dev ** 2))

    #h_i = np.zeros(ra.N_s)
    #for theta in theta_target:
    #    # compute angle difference for all neurons
    #    true_diff = (ra.alpha - theta + np.pi) % (2 * np.pi) - np.pi
    #    h_i += ra.h_0/np.sqrt(2*math.pi*ra.std_dev**2) * np.exp(- (true_diff ** 2) / (2 * ra.std_dev ** 2))
#
    ra.h_i = h_i
    
    #h_i = (ra.h_0/np.sqrt(2*math.pi*ra.std_dev**2)*np.exp(-(true_diff[0])**2/(2*ra.std_dev**2)) + ra.h_0/np.sqrt(2*math.pi*ra.std_dev**2)*np.exp(-(true_diff[1])**2/(2*ra.std_dev**2)))/2

def hamiltonian_updated(ra):
   
    i = ra.indx
    alpha_i = ra.alpha[i]

    angle_diffs = np.abs(alpha_i - ra.alpha) 
    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ra.v))
    kernel[i] = 0  # no self interaction

    interaction_sum = np.dot(kernel, ra.sigma)
    spin_flip = np.array([ra.sigma[i], -ra.sigma[i]])

    interaction_energy = (interaction_sum * spin_flip) / (ra.N_s - 1)
    #input_energy = ra.h_i[i] * spin_flip
    input_energy = ra.h_i[i] * spin_flip
    leak_energy = ra.h_b * spin_flip

    ra.H = -1 * (interaction_energy + input_energy - leak_energy)

def hamiltonian_w_one_random_spin(ra):
   
    i = ra.indx
    choices = np.delete(np.arange(ra.N_s), i)
    #j = np.random.randint(0,ra.N_s)
    j = np.random.choice(choices)
    alpha_i = ra.alpha[i]

    angle_diffs = np.abs(alpha_i - ra.alpha[j])

    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ra.v))
    print(kernel)

    interaction_sum = kernel*ra.sigma[i]*ra.sigma[j]
    spin_flip = np.array([ra.sigma[i], -ra.sigma[i]])

    interaction_energy = (interaction_sum * spin_flip)
    #input_energy = ra.h_i[i] * spin_flip
    input_energy = ra.h_i[i] * spin_flip
    leak_energy = ra.h_b * spin_flip

    ra.H = -1 * (interaction_energy + input_energy - leak_energy)


def hamiltonian(ra):
    kernel = None
    interaction_energy = None

    i = ra.indx
    alpha_i = copy.deepcopy(ra.alpha[i])

    angle_diffs = np.abs((copy.deepcopy(ra.alpha) - alpha_i + np.pi) % (2 * np.pi) - np.pi)

    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ra.v))
    kernel[i] = 0  # no self interaction
    #print(kernel)

    interaction_sum = np.dot(kernel, ra.sigma)
    #print('interaction sum: ', interaction_sum)
    #print(np.shape(kernel), np.shape(ra.sigma))
    spin_flip = np.array([ra.sigma[i], 1-ra.sigma[i]])
   
    #if (ra.h_i[i]>0.1):
    #print("Spin ",i," (", alpha_i,"): ", interaction_sum, " + ", ra.h_i[i])

    interaction_energy = (interaction_sum * spin_flip) / (ra.N_s - 1)
    #print('interaction energy: ', interaction_energy)
    #print(interaction_energy)
    input_energy = ra.h_i[i] * spin_flip
    #input_energy = ra.h_i * spin_flip
    leak_energy = ra.h_b * spin_flip

    ra.H = -1 * (interaction_energy + input_energy - leak_energy)
    #print(ra.H)
    #print('interaction sum: ', interaction_sum, 'spin flip: ', spin_flip,'interaction energy: ', interaction_energy, 'H: ', ra.H)
   

def hamiltonian_multi_spin(ra):
    i = ra.indx
    m = ra.subindx

    sigma_bar = np.mean(ra.sigma, axis=1)

    alpha_i = ra.alpha[i]
    angle_diffs = np.abs(alpha_i-ra.alpha)
    J = np.cos(np.pi*(angle_diffs/np.pi)**ra.v)
    J[i] = 0 

    interaction_sum = np.dot(J,sigma_bar)/ra.N_s
    spin_flip = np.array([ra.sigma[i, m], -ra.sigma[i, m]])
    interaction_energy = interaction_sum * spin_flip

    input_energy = 0#ra.h_i[i]*spin_flip
    leak_energy = ra.h_b * spin_flip
    ra.H = -1 * (interaction_energy + input_energy + leak_energy)


def get_unique_filename(base_path, suffix=".png"):
    if not os.path.exists(base_path+suffix ):
        return base_path + suffix

    version = 2
    while True:
        new_path = f"{base_path}_v{version}{suffix}"
        if not os.path.exists(new_path):
            return new_path
        version += 1


def plot_neuron_activity(activity_log,ra):
    if ra.allocentric:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/allocentric"
    else:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/egocentric"

    png_path = os.path.join(save_dir, "beta_{}".format(ra.beta))
    plt.figure(figsize=(8, 4))
    plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower')
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title("Ring attractor activity over time, beta={}".format(ra.beta))
    plt.colorbar(label='Activity (1 = active)')
    plt.tight_layout()
    file_path = get_unique_filename(png_path) 
    #plt.savefig(file_path, dpi=900,bbox_inches='tight')
    plt.show()

def plot_trajectory(trajectory,target,ra):
    T = trajectory.shape[0]
    colors = np.linspace(0, 1, T)   
    if ra.allocentric:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/allocentric"
    else:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/egocentric"

    png_path = os.path.join(save_dir, "beta_{}_trajectory".format(ra.beta))
    plt.figure(figsize=(6, 6))
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, cmap='viridis', s=2)
    #plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue')
    #plt.scatter(trajectory[:, 0], trajectory[:, 1], color='blue')
    plt.scatter(target[:,0], target[:,1], color='orange', label='Target')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='End')
    plt.axis('equal')
    #plt.xlim(-15, 15)
    #plt.ylim(-15, 15)
    plt.autoscale()

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

    file_path = get_unique_filename(png_path.replace(".png", "") )
    #plt.savefig(file_path, dpi=900,bbox_inches='tight')
    plt.show()




def plot_neuron_activity_w_bump_angle(activity_log, ra, bump_log,target_angle_log):

    if ra.allocentric:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/allocentric"
    else:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/egocentric"

    png_path = os.path.join(save_dir, "beta_{}".format(ra.beta))
    plt.figure(figsize=(8, 4))
    plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower')
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title("Ring attractor activity over time, beta={}".format(ra.beta))
    plt.colorbar(label='Activity (1 = active)')

    t_0 = activity_log.shape[0]
    for t in range(0, t_0, 50):
        bump_deg = np.degrees(bump_log[t]) % 360
        plt.text(t, 5, f"{bump_deg:.0f}°", rotation=90, color='white',
                 fontsize=8, ha='center', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))


    plt.tight_layout()
    file_path = get_unique_filename(png_path) 
    #plt.savefig(file_path, dpi=900, bbox_inches='tight')
    plt.show()

def plot_neuron_activity_ba(activity_log, ra, bump_log, target_angle_log):

    if ra.allocentric:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/allocentric"
    else:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/egocentric"

    png_path = os.path.join(save_dir, "beta_{}".format(ra.beta))
    plt.figure(figsize=(8, 4))

    # Y-axis: angles in degrees
    #angles_deg = np.degrees(ra.alpha) % 360
    angles_deg = np.degrees(ra.alpha) 
    extent = [0, activity_log.shape[0], angles_deg[0], angles_deg[-1]]

    plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower',
               extent=extent)


    bump_deg = np.degrees(bump_log) #% 360
    target_deg = np.degrees(target_angle_log) #%360
    
    #angles_deg = np.degrees(ra.alpha)
    #extent = [0, activity_log.shape[0], angles_deg[0], angles_deg[-1]]
#
    #plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower', extent=extent)
#
    ## Bump line: convert bump_log (radians) to degrees in [-180, 180)
    #bump_deg = np.degrees(bump_log)
    #bump_deg = (bump_deg + 180) % 360 - 180  # wrap to [-180, 180)

    plt.plot(np.arange(len(bump_deg)), bump_deg, color='cyan', linewidth=1.5, label='Bump angle')
    plt.plot(np.arange(len(target_deg)), target_deg, color='brown', linewidth=1.5, label='Target angle')

    plt.xlabel("Time step")
    plt.ylabel("Preferred direction (°)")
    plt.title("Ring attractor activity over time, β={}".format(ra.beta))
    plt.colorbar(label='Activity (1 = active)')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    file_path = get_unique_filename(png_path)
    # plt.savefig(file_path, dpi=900, bbox_inches='tight')
    plt.show()


def plot_neuron_activity_ba_mt(activity_log, ra, bump_log, target_angle_log):

    if ra.allocentric:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/allocentric"
    else:
        save_dir = "/Users/simayatasoy/Desktop/results/ring_attractor/egocentric"

    png_path = os.path.join(save_dir, "beta_{}".format(ra.beta))
    plt.figure(figsize=(8, 4))

    # Y-axis: angles in degrees
    #angles_deg = np.degrees(ra.alpha) % 360
    angles_deg = np.degrees(ra.alpha) 
    extent = [0, activity_log.shape[0], angles_deg[0], angles_deg[-1]]

    plt.imshow(activity_log.T, aspect='auto', cmap='gray', origin='lower',
               extent=extent)


    bump_deg = np.degrees(bump_log) #% 360
    
    target_deg = np.degrees(target_angle_log )

    for i in range(target_deg.shape[1]):  
        plt.plot(
            np.arange(target_deg.shape[0]), 
            target_deg[:, i], 
            linewidth=1.5, 
            label=f'Target {i+1} angle'
        )

    plt.plot(np.arange(len(bump_deg)), bump_deg, color='cyan', linewidth=1.5, label='Bump angle')

    plt.xlabel("Time step")
    plt.ylabel("Preferred direction (°)")
    plt.title("Ring attractor activity over time, β={}".format(ra.beta))
    plt.colorbar(label='Activity (1 = active)')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    file_path = get_unique_filename(png_path)
    # plt.savefig(file_path, dpi=900, bbox_inches='tight')
    plt.show()
