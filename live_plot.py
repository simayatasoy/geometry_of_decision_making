import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_trajectory_live(trajectory, target, ra, bump_log, target_angle_log):
    T = trajectory.shape[0]
    colors = np.linspace(0, 1, T)

    fig, ax = plt.subplots(figsize=(6, 6))


    ax.scatter(target[:, 0], target[:, 1], color='orange', label='Target')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    margin = 5
    x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Agent Trajectory with Live Direction Arrow")
    ax.legend()


    path, = ax.plot([], [], color='blue', linewidth=1)

    dot, = ax.plot([], [], 'ro')


    arrow = ax.quiver([], [], [], [], color='red', scale=20, width=0.01)
    target_arrow = ax.quiver([], [], [], [], color='orange', scale=20, width=0.01)

    ref_arrow_origin = [x_max - margin * 2, y_max - margin * 2]
    ref_arrow = ax.quiver(*ref_arrow_origin, 1, 0, color='purple', scale=10, width=0.015)
    ax.text(ref_arrow_origin[0], ref_arrow_origin[1] + 1.5, 'Bump angle', color='purple', fontsize=8)


    def init():
        path.set_data([], [])
        dot.set_data([], [])
        arrow.set_UVC([], [])
        ref_arrow.set_UVC(np.cos(0), np.sin(0))
        target_arrow.set_UVC([], [])
        return path, dot, arrow, ref_arrow, target_arrow

    def update(i):

        path.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])


        x, y = trajectory[i]
        dot.set_data(x, y)


        angle = bump_log[i]
        dx = np.cos(angle)
        dy = np.sin(angle)

        target_angle = target_angle_log[i]
        target_dx = np.cos(target_angle)
        target_dy = np.sin(target_angle)


        arrow.set_offsets([x, y])
        arrow.set_UVC(dx, dy)
        ref_arrow.set_UVC(np.cos(angle), np.sin(angle))

        target_arrow.set_offsets([x, y])
        target_arrow.set_UVC(target_dx, target_dy)



        return path, dot, arrow, ref_arrow, target_arrow

    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=30)
    #ani.save("animation.mp4", writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()

def plot_hi_live(h_i_log, ra,target_angle_log,bump_log):
    T, N_s = h_i_log.shape
    alpha_deg = np.degrees(ra.alpha) 
    target_angles = np.degrees(target_angle_log) 
    bump_angles = np.degrees(bump_log)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_container = ax.bar(alpha_deg, h_i_log[0], width=360 / N_s, color='skyblue')
    target_line = ax.axvline(x=0, color='orange', linestyle='--', label='Target angle')
    bump_line = ax.axvline(x=0, color='green', linestyle='-', label='Bump angle')

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, np.max(h_i_log) * 1.1)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("h_i (External Input)")
    ax.set_title("External Input h_i vs. Angle Over Time")

    def update(frame):
        for rect, h in zip(bar_container, h_i_log[frame]):
            rect.set_height(h)
        target_line.set_xdata(target_angles[frame])
        bump_line.set_xdata(bump_angles[frame])
        ax.set_title(f"External Input h_i (t={frame})")
        time.sleep(0.2)
        return list(bar_container) + [target_line] + [bump_line]

    ani = FuncAnimation(fig, update, frames=T, blit=False, interval=50)
    plt.tight_layout()
    plt.show()

def plot_hi_live(h_i_log, ra,target_angle_log,bump_log):
    T, N_s = h_i_log.shape
    alpha_deg = np.degrees(ra.alpha) 
    target_angles = np.degrees(target_angle_log) 
    bump_angles = np.degrees(bump_log)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_container = ax.bar(alpha_deg, h_i_log[0], width=360 / N_s, color='skyblue')
    target_line = ax.axvline(x=0, color='orange', linestyle='--', label='Target angle')
    bump_line = ax.axvline(x=0, color='green', linestyle='-', label='Bump angle')

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, np.max(h_i_log) * 1.1)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("h_i (External Input)")
    ax.set_title("External Input h_i vs. Angle Over Time")

    def update(frame):
        for rect, h in zip(bar_container, h_i_log[frame]):
            rect.set_height(h)
        target_line.set_xdata(target_angles[frame])
        bump_line.set_xdata(bump_angles[frame])
        ax.set_title(f"External Input h_i (t={frame})")
        time.sleep(0.2)
        return list(bar_container) + [target_line] + [bump_line]

    ani = FuncAnimation(fig, update, frames=T, blit=False, interval=50)
    plt.tight_layout()
    plt.show()

def plot_hi_live_mt(h_i_log, ra, target_angle_log, bump_log):
    T, N_s = h_i_log.shape
    num_targets = target_angle_log.shape[1]

    alpha_deg = np.degrees(ra.alpha) 
    target_angles = np.degrees(target_angle_log)
    bump_angles = np.degrees(bump_log)

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_container = ax.bar(alpha_deg, h_i_log[0], width=360 / N_s, color='skyblue')

    # Create one line per target
    target_lines = [
        ax.axvline(x=0, color='orange', linestyle='--', label=f'Target {i+1}')
        for i in range(num_targets)
    ]
    bump_line = ax.axvline(x=0, color='green', linestyle='-', label='Bump angle')

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, np.max(h_i_log) * 1.1)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("h_i (External Input)")
    ax.set_title("External Input h_i vs. Angle Over Time")

    # Only add one legend entry per label
    handles = target_lines[:1] + [bump_line]
    ax.legend(handles=handles)

    def update(frame):
        for rect, h in zip(bar_container, h_i_log[frame]):
            rect.set_height(h)
        for i, line in enumerate(target_lines):
            line.set_xdata(target_angles[frame, i])
        bump_line.set_xdata(bump_angles[frame])
        ax.set_title(f"External Input h_i (t={frame})")
        time.sleep(0.2)
        return list(bar_container) + target_lines + [bump_line]

    ani = FuncAnimation(fig, update, frames=T, blit=False, interval=50)
    plt.tight_layout()
    plt.show()
