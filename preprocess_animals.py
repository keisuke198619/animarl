import os, sys, glob, argparse, random, copy, math
import pickle
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib

import multiprocessing as mp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from utils.utils_chase import *
matplotlib.use('TkAgg')

# create argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--numProcess", type=int, default=16, help="No. CPUs used, default: 16"
)
# parser.add_argument('--n_actions', type=int, default=13, help='No. of actions, default: 13')
parser.add_argument(
    "--reward", type=str, default="touch", help="reward type: touch (default), ..."
)
# setting
parser.add_argument(
    "--val_devide", type=int, default=5, help="split ratio for validation, default: 3"
)
# parser.add_argument('--test_cond', type=str, default=None, help='test condition')
parser.add_argument(
    "--opponent", action="store_true", help="model opponent first, default: No"
)
parser.add_argument(
    "--data_path", type=str, default="../AniMARL_data/", help="data path"
)
parser.add_argument(
    "--result_path",
    type=str,
    default="../AniMARL_results/preprocessed/",
    help="result path",
)
parser.add_argument(
    "--episode_sec", type=int, default=8, help="episode seconds (s) for bats"
)
parser.add_argument(
    "--env", type=str, default="agent", help="environment, default: agent"
)
parser.add_argument(
    "--option", type=str, default=None, help="option (default: None)"
)

parser.add_argument(
    "--trainSize", type=int, default=-1, help="Size of the training sample, default: -1 (all)"
)

parser.add_argument('--check_hist', action='store_true')

args = parser.parse_args()
numProcess = args.numProcess
os.environ["OMP_NUM_THREADS"] = str(numProcess)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def plot_trajectory(pos,j,indiv,no_seq):
    plt.figure(figsize=(10, 6))
    for agent in range(pos.shape[1]):
        plt.plot(pos[:, agent, 0], pos[:, agent, 1], label=f'Agent {agent+1}')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of Agents')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig("./figures/fly/" + f'trajectory_{j}_{indiv}_{no_seq}.png')
    plt.close()

def create_trajectory_video(pos, vel, j, indiv, no_seq, threshold, fps=30):
    fig, ax = plt.subplots(figsize=(10, 6))

    def init():
        ax.clear()
        ax.set_xlim(np.min(pos[:, :, 0]), np.max(pos[:, :, 0]))
        ax.set_ylim(np.min(pos[:, :, 1]), np.max(pos[:, :, 1]))
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Trajectory of Agents')
        ax.grid(True)
        return []

    def update(frame):
        ax.clear()
        velocities = []
        for agent in range(pos.shape[1]):
            ax.plot(pos[:frame, agent, 0], pos[:frame, agent, 1], label=f'Agent {agent}', linewidth=1)
            if vel.shape[0] > frame:
                norm_vel = np.linalg.norm(vel[frame, agent])
                velocities.append(norm_vel)
                for past_frame in range(frame + 1):
                    if np.linalg.norm(vel[past_frame, agent]) >= threshold:
                        ax.plot(pos[past_frame, agent, 0], pos[past_frame, agent, 1], 'o', markersize=8)
                        ax.text(pos[past_frame, agent, 0], pos[past_frame, agent, 1], f'{np.linalg.norm(vel[past_frame, agent]):.2f}', fontsize=12, color='red')

        ax.legend()
        ax.grid(True)
        title = f'Trajectory of Agents - File: {j}_{indiv}_{no_seq} - Frame: {frame}\n'
        if vel.shape[0] >= frame:
            title += 'Velocities: ' + ', '.join([f'{v:.2f}' for v in velocities])
        ax.set_title(title)
        return []

    ani = animation.FuncAnimation(fig, update, frames=pos.shape[0], init_func=init, blit=True, repeat=False)
    
    ani.save(f"./figures/newt_video/trajectory_{j}_{indiv}_{no_seq}.mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()

def plot_velocity(vel, args, title='velocity'):
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(vel[:, 0], color="blue")
    plt.xlabel("frames")
    plt.ylabel(title+"_x")
    plt.legend()
    if len(vel[1]) > 1:
        plt.subplot(2, 1, 2)
        plt.plot(vel[:, 1], color="blue")
        plt.xlabel("frames")
        plt.ylabel(title+"_y")
        plt.legend()
        plt.tight_layout()
    plt.savefig("./figures/" + args.env + "_" + title + ".png", format="png")


def plot_histogram(vel, args, title='velocity', bins=20):
    normalize = True
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.title(title+" histogram")
    plt.ylabel("frequnecy")
    for n in range(args.n_all_agents):
        plt.subplot(2, 2, n+1)
        if args.n_all_agents == 1: 
            vel_ = vel
        elif len(vel) == args.n_all_agents:
            vel_ = vel[n]
        else:
            vel_ = vel[:,n]
        if normalize:
            hist, bins = np.histogram(vel_, bins=bins)
            widths = np.diff(bins)
            hist = hist / hist.sum()
            plt.bar(bins[:-1], hist, widths, edgecolor="black")
        else:
            plt.hist(vel_, bins=bins, edgecolor='black', density=True)
    
        plt.xlabel(title)

    plt.savefig("./figures/" + args.env + "_"+title+"_histogram.png", format="png")


def show_filter(pos, pos_filt, theta, theta_filt, args):
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pos[:, 0], color="blue")
    plt.plot(pos_filt[:, 0], color="red", linestyle="--")
    plt.xlabel("time (s)")
    plt.ylabel("pos_x")
    plt.legend()
    if theta is not None:
        plt.subplot(2, 1, 2)
        plt.plot(theta, color="blue")
        plt.plot(theta_filt, color="red", linestyle="--")
        plt.xlabel("time (s)")
        plt.ylabel("theta")
        plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/" + args.env + "_filter.png", format="png")

def get_dist(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    dist = np.sqrt(np.sum(np.square(pos_rel),1))
    return dist

def get_sub_vec(abs_pos_own, abs_pos_adv, abs_vec, args):

    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(pos_rel[:,1], pos_rel[:,0])
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) # inverse
    vec_post = []
    for t in range(abs_vec.shape[0]):
        vec_post.append(np.dot(rot[:,:,t], abs_vec[t,:]))
    return np.array(vec_post), theta

def get_sub_vec_3d(abs_pos_own, abs_pos_adv, abs_vec, args):
    """
    Returns:
        sub_acc (np.ndarray): Transformed vectors in 3D space (Nx3).
    """
    # Calculate the relative position between the target and the own agent
    pos_rel = abs_pos_adv - abs_pos_own
    
    # Normalize the relative position vector to obtain the unit vector
    pos_rel_norm = np.linalg.norm(pos_rel, axis=1, keepdims=True)
    unit_pos_rel = pos_rel / pos_rel_norm  # Unit vector of relative position

    # Define the base direction vector (x-axis direction)
    base_vec = np.array([1, 0, 0])

    vec_post = []
    for t in range(abs_vec.shape[0]):
        # Calculate the rotation axis as the cross product between unit_pos_rel and base_vec
        rot_axis = np.cross(unit_pos_rel[t], base_vec)
        axis_norm = np.linalg.norm(rot_axis)  # Calculate the norm of the rotation axis
        
        # If the rotation axis is a zero vector (already aligned), use the identity matrix
        if axis_norm == 0:
            rot_mat = np.eye(3)  # Identity matrix (no rotation needed)
        else:
            # Normalize the rotation axis
            rot_axis /= axis_norm
            
            # Calculate the rotation angle using the dot product between unit_pos_rel and base_vec
            angle = np.arccos(np.clip(np.dot(unit_pos_rel[t], base_vec), -1.0, 1.0))
            
            # Compute the rotation matrix using Rodrigues' rotation formula
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                          [rot_axis[2], 0, -rot_axis[0]],
                          [-rot_axis[1], rot_axis[0], 0]])
            
            # Calculate the final rotation matrix
            rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # Apply the rotation matrix to the absolute vector
        transformed_vec = np.dot(rot_mat, abs_vec[t, :])
        vec_post.append(transformed_vec)

    # Return the transformed vectors, with placeholders for phi and theta
    return np.array(vec_post)

def discrete_direction(pos_1, pos_2, vec, args, no_angles=12):
    if args.dim == 2:
        vec_projected, theta = get_sub_vec(pos_1, pos_2, vec, args)
        try: angles = np.arctan2(vec_projected[:,1], vec_projected[:,0])
        except: import pdb; pdb.set_trace()
    elif args.dim == 3:
        # test data
        # pos_1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) 
        # pos_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 
        # vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        vec_projected = get_sub_vec_3d(pos_1, pos_2, vec, args)
        # for test data, all: [1, 0, 0]

    if no_angles == 12 and args.dim == 2:
        # relative coordinates 
        # ang = action * -np.pi / 6 # 0:top (toward target), 3:right, 6:bottom, 9:left
        width = np.pi/12
        discrete_angles, directions = [],[]

        for t, angle in enumerate(angles):
            if angle > -width and angle <= width: # top
                discrete_angle = 0; phi = 0
            elif angle > -np.pi/6-width and angle <= -np.pi/6+width:  
                discrete_angle = 1; phi = -np.pi/6
            elif angle > -np.pi/3-width and angle <= -np.pi/3+width: 
                discrete_angle = 2; phi = -np.pi/3
            elif angle > -np.pi/2-width and angle <= -np.pi/2+width: # right
                discrete_angle = 3; phi = -np.pi/2
            elif angle > -4*np.pi/6-width and angle <= -4*np.pi/6+width: 
                discrete_angle = 4; phi = -4*np.pi/6
            elif angle > -5*np.pi/6-width and angle <= -5*np.pi/6+width: 
                discrete_angle = 5; phi = -5*np.pi/6
            elif angle <= -np.pi+width or angle > np.pi-width: # bottom
                discrete_angle = 6; phi = np.pi
            elif angle >= 5*np.pi/6-width and angle < 5*np.pi/6+width: #  
                discrete_angle = 7; phi = 5*np.pi/6
            elif angle >= 4*np.pi/6-width and angle < 4*np.pi/6+width: # 
                discrete_angle = 8; phi = 4*np.pi/6
            elif angle >= np.pi/2-width and angle < np.pi/2+width: # left
                discrete_angle = 9; phi = np.pi/2
            elif angle >= np.pi/3-width and angle < np.pi/3+width: # 
                discrete_angle = 10; phi = np.pi/3
            elif angle >= np.pi/6-width and angle < np.pi/6+width: # 
                discrete_angle = 11; phi = np.pi/6

            if np.linalg.norm(vec[t]) < args.acc_threshold:
                discrete_angle = 12
            # if np.isnan(angle): 
            #    discrete_angle = np.nan

            discrete_angles.append(discrete_angle)

            xy = np.array([np.cos(phi), np.sin(phi)])

            # pos_rel = pos_2 - pos_1
            # theta = np.arctan2(pos_rel[t,1], pos_rel[t,0])
            theta_ = theta[t]
            rot = np.array([[np.cos(theta_), -np.sin(theta_)], [np.sin(theta_), np.cos(theta_)]]) # forward
            xy_ = np.dot(rot, xy)
            directions.append(xy_)

    elif no_angles == 14 and args.dim == 3:
        # Define the primary directions as unit vectors in Cartesian coordinates
        discrete_angles, directions = [], []

        # Define 14 discrete directions as unit vectors in 3D space
        # Each direction is represented as a (x, y, z) vector
        direction_vectors = [
            np.array([1, 0, 0]),   # Forward
            np.array([-1, 0, 0]),  # Backward
            np.array([0, 1, 0]),   # Left
            np.array([0, -1, 0]),  # Right
            np.array([0, 0, 1]),   # Up
            np.array([0, 0, -1]),  # Down
            np.array([1, 1, 0]) / np.sqrt(2),  # Forward-Left
            np.array([-1, 1, 0]) / np.sqrt(2), # Backward-Left
            np.array([1, -1, 0]) / np.sqrt(2), # Forward-Right
            np.array([-1, -1, 0]) / np.sqrt(2),# Backward-Right
            np.array([1, 0, 1]) / np.sqrt(2),  # Forward-Up
            np.array([-1, 0, 1]) / np.sqrt(2), # Backward-Up
            np.array([1, 0, -1]) / np.sqrt(2), # Forward-Down
            np.array([-1, 0, -1]) / np.sqrt(2) # Backward-Down
        ]

        # Iterate through each vector to determine the closest direction
        for t, vector in enumerate(vec):
            # Normalize the input vector to a unit vector
            norm_vector = vector / np.linalg.norm(vector)

            # Calculate the distance between the input vector and each direction vector
            differences = [np.linalg.norm(norm_vector - direction) for direction in direction_vectors]
            
            # Find the index of the closest direction vector
            discrete_angle = np.argmin(differences)

            # Append the discrete angle and the corresponding direction
            discrete_angles.append(discrete_angle)
            directions.append(direction_vectors[discrete_angle])

        # Handle low acceleration threshold case
        for t in range(len(vec)):
            if np.linalg.norm(vec[t]) < args.acc_threshold:
                discrete_angles[t] = 14  
    else:
        print('not defined other than no_angles == 12 or 14')
        import pdb; pdb.set_trace()
    return np.array(discrete_angles),np.array(directions)

def compute_ud(vels,vels_next,accs,args):
    if len(vels[0].shape) == 1:
        vels_,vels_next_,accs_ = np.concatenate(vels)[:,None],np.concatenate(vels_next)[:,None],np.concatenate(accs)[:,None]
    else:
        vels_,vels_next_,accs_ = np.vstack(vels),np.vstack(vels_next),np.vstack(accs)

    vels_zero,vels_zero_next,vels,vmax,us,ds = [],[],[],[],[],[]
    if args.dim == 2:
        for n in range(args.n_all_agents):
            vels.append(vels_)
            vels_zero.append((vels_[:,n])[vels_[:,n]<args.acc_threshold*args.Fs])
            cond_ = (vels_[:,n] < args.acc_threshold * args.Fs) & (vels_next_[:,n] > args.acc_threshold * args.Fs)
            vels_zero_next.append(vels_next_[cond_, n])
            # vels_zero_next.append((vels_next_[:,n])[vels_[:,n]<args.acc_threshold*args.Fs])
            
            # if vels_zero_next[n].shape[0] > 0:
            # compute u = v' /dt
            us.append(np.median(vels_zero_next[n]) /args.Fs)
            if np.isnan(us[n]):
                import pdb; pdb.set_trace()
            # compute d = udt/v_max
            vmax.append((vels_next_[:,n])[vels_next_[:,n]>np.percentile(vels_next_[:,n], 99)])
            ds.append(us[n]*args.Fs/np.median(vmax[n]))
        # viscosity = 1 - (vels_next-accs*args.Fs)/vels # from definition of viscosity
    elif args.dim == 3 and args.env == "dragonfly":

        # Predefined drag coefficients for 3D case
        drag_coefficients = np.linspace(0.00001, 0.0003, 10)  # d = 0.1, 0.15, ..., 0.5

        # For each drag coefficient, compute the corresponding thrust
        for d in drag_coefficients:
            # Calculate the thrust u corresponding to each drag coefficient
            # Use the formula: u = d * v_max^2
            import pdb; pdb.set_trace()
            v_max_mean = np.max(vels_[0])
            u = d * (v_max_mean ** 2)
            
            # Store the computed thrust and drag coefficient
            us.append(u)
            ds.append(d)
 
    us,ds = np.array(us),np.array(ds)
    print("u: "+ str(us))
    print("d: "+ str(ds))
    return us, ds

def append_data_newt(poss, vels, accs, vel_abss, acc_abss, vels_prev, vels_next, lengths, conditions, rewards, 
                pos, vel, acc, vel_abs, acc_abs, reward, subsample):
    poss.append(pos[::subsample])
    vels.append(vel[::subsample])
    accs.append(acc[::subsample])
    vel_abss.append(vel_abs[::subsample])
    acc_abss.append(acc_abs[::subsample])
    vels_prev.append(vel_abs[:-1])
    vels_next.append(vel_abs[1:])
    lengths.append(vel[::subsample].shape[0])
    conditions.append(0)
    reward[2] = pos[::subsample].shape[0]/10
    rewards.append(reward)

def split_data_newt(pos, vel, acc, vel_abs, acc_abs, reward, max_len, subsample, num_segments = 1):
    poss_ = []
    vels_ = []
    accs_ = []
    vel_abss_ = []
    acc_abss_ = []
    rewards_ = []
    max_len_ = max_len * subsample

    if num_segments == 1:
        poss_.append(pos[-max_len_:])
        vels_.append(vel[-max_len_:])
        accs_.append(acc[-max_len_:])
        vel_abss_.append(vel_abs[-max_len_:])
        acc_abss_.append(acc_abs[-max_len_:])
        rewards_.append(reward)
    else:
        num_segments = np.min([(pos.shape[0] + max_len_ - 1) // max_len_ , num_segments])

        for i in range(num_segments):
            start_idx = max(0, pos.shape[0] - (i + 1) * max_len_)
            end_idx = pos.shape[0] - i * max_len_

            poss_.append(pos[start_idx:end_idx])
            vels_.append(vel[start_idx:end_idx])
            accs_.append(acc[start_idx:end_idx])
            vel_abss_.append(vel_abs[start_idx:end_idx])
            acc_abss_.append(acc_abs[start_idx:end_idx])
            rewards.append(reward)

    return poss_, vels_, accs_, vel_abss_, acc_abss_, rewards_

def update_animation_agent(ii,state, n_mate, n_opponent):
    ax.clear()
    top_bottom = [-1, 1]
    left_right = [-1, 1]
    ax.plot([-1, -1], top_bottom, color="black")
    ax.plot([1, 1], top_bottom, color="black")
    ax.plot(left_right, [-1, -1], color="black")
    ax.plot(left_right, [1, 1], color="black")
    artists = [] 

    for n in range(n_mate):
        point, = ax.plot(state[ii][n][0], state[ii][n][1], 'o', markersize=8, color="gray")
        text = ax.text(state[ii][n][0], state[ii][n][1], str(n))
        artists.extend([point, text]) 

    for n in range(n_opponent):
        idx = n_mate + n
        point, = ax.plot(state[ii][idx][0], state[ii][idx][1], 'o', markersize=8, color="black")
        artists.append(point)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    return artists
    
if __name__ == "__main__":
    # constants
    # args.Fs = 0.1
    if args.env == "silkmoth":
        args.n_agents = 1
        args.n_opponents = 0
        args.reward = "reach"
        args.acc_threshold = 0.01
        args.dim = 2
    elif "dragonfly" in args.env:
        args.n_agents = 1
        args.n_opponents = 1
        args.reward = 'touch'
        args.acc_threshold = 0.1
        args.dim = 3
    elif "fly" in args.env:
        args.n_agents = 2
        args.n_opponents = 1
        args.reward = 'touch'
        args.acc_threshold = 0.1
        args.dim = 2
    elif "newt" in args.env:
        args.n_agents = 2
        args.n_opponents = 1
        args.reward = 'touch'
        args.acc_threshold = 0.01
        args.dim = 2
    elif args.env == "bat":
        args.n_agents = 1
        args.n_opponents = 0
        args.reward = "turn_rate"
        episode_sec = args.episode_sec
        args.acc_threshold = None
        args.dim = 2
    elif "agent" in args.env:
        args.n_agents = 2
        args.n_opponents = 1
        args.reward = "touch"
        args.acc_threshold = 0.3 # if args.option=="CF" else 0.01 # 01
        args.dim = 2

    n_agents = args.n_agents
    seed = 0
    CUDA = True
    args.behavior = True
    # parameters

    val_devide = args.val_devide

    if args.env == "silkmoth":
        args.state_shape = 14
    elif args.env == "dragonfly":
        args.state_shape = 12
    elif args.env == "fly":
        args.state_shape = 12
    elif args.env == "newt":
        args.state_shape = 12
    elif args.env == "bat":
        args.state_shape = 210
    elif args.env == "agent":
        args.state_shape = 12

    args.n_all_agents = args.n_agents + args.n_opponents

    # random seed
    random.seed(seed)
    np.random.seed(seed)

    # if args.env != "silkmoth":
    #    args.env += '_' + str(n_agents) + 'vs' + str(args.n_opponents)
    os.makedirs(args.result_path, exist_ok=True)

    def compute_velocity_acceleration(args,pos=None,vel=None):
        if vel is None:
            vel = np.diff(pos,axis=0)/args.Fs
        vel_abs = (np.sum(vel**2, axis=2) + 1e-16) ** (1 / 2)
        acc = np.diff(vel,axis=0)/args.Fs
        acc_abs = (np.sum(acc.reshape((-1, args.n_all_agents,args.dim))**2, axis=2) + 1e-16) ** (1 / 2)

        return vel,vel_abs,acc,acc_abs
    
    def compute_action_reconstruction(pos,vel,args,ds,us):
    
        if len(pos) == 3:
            pos_p1,pos_p2,pos_e = pos
            vel_abs = (np.sum(vel**2, axis=2) + 1e-16) ** (1 / 2)
            ds_reshaped = ds.reshape(1, args.n_all_agents, 1)
            acc = (vel[1:] - vel[:-1]*(1-ds_reshaped))*args.Fs

            dist1 = get_dist(pos_p1, pos_e)
            dist2 = get_dist(pos_p2, pos_e)
            dist = np.concatenate([dist1[:,np.newaxis], dist2[:,np.newaxis]],1)
            adv_index = np.argmin(dist,axis=1)[0]
            abs_pos_adv = np.concatenate([pos_p1[:,:,np.newaxis], pos_p2[:,:,np.newaxis]],2)[:,:,adv_index]

            # compute actions (directions)
            actions_p1, direction_p1 = discrete_direction(pos_p1[:-1],pos_e[:-1],acc[:,0],args)
            actions_p2, direction_p2 = discrete_direction(pos_p2[:-1],pos_e[:-1],acc[:,1],args)
            actions_e, direction_e = discrete_direction(pos_e[:-1],abs_pos_adv[:-1],acc[:,2],args)

            action = np.array([actions_p1,actions_p2,actions_e]).transpose()
            action = np.concatenate([action,action[-1:,:]],0) 
            
            # reconstruction
            us_reshaped = us.reshape(1, args.n_all_agents, 1)
            directions = np.array([direction_p1,direction_p2,direction_e]).transpose((1,0,2))
            vel_next_rec = vel[:-1]*(1-ds_reshaped)+us_reshaped*directions*args.Fs
            rmse_vel = np.sqrt(np.mean(np.sum((vel_next_rec[:-1] - vel[1:-1])**2+1e-6,2),0))
            #if np.isnan(rmse_vel).any():
            #    import pdb; pdb.set_trace()

        elif args.env == "dragonfly" and len(pos) == 2:
            pos_p1,pos_p2 = pos
            vel_abs = (np.sum(vel**2, axis=1) + 1e-16) ** (1 / 2)
            ds_reshaped = ds.reshape(1, 1, 1)
            acc = (vel[1:] - vel[:-1]*(1-ds_reshaped))*args.Fs

            # compute actions (directions)
            actions_p1, direction_p1 = discrete_direction(pos_p1[:-1],pos_p2[:-1],acc[:,0],args, no_angles=14)
            actions_p2, direction_p2 = discrete_direction(pos_p2[:-1],pos_p1[:-1],acc[:,1],args, no_angles=14)

            action = np.array([actions_p1,actions_p2]).transpose()
            action = np.concatenate([action,action[-1:,:]],0) 
            
            # reconstruction
            us_reshaped = us.reshape(1, 1, 1)
            directions = np.array([direction_p1,direction_p2]).transpose((1,0,2))
            vel_next_rec = vel[:-1]*(1-ds_reshaped)+us_reshaped*directions*args.Fs
            rmse_vel = np.sqrt(np.mean(np.sum((vel_next_rec[:-1] - vel[1:-1])**2+1e-6,2),0))

        return action,vel_abs,rmse_vel

    def low_pass_filter(data, cutoff_frequency, sample_rate):
        nyquist_rate = sample_rate / 2.0
        normal_cutoff = cutoff_frequency / nyquist_rate
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        if data.shape[0] > 3 * max(len(b), len(a)):
            filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=0, arr=data)
        else:
            filtered_data = data
        #    import pdb; pdb.set_trace()
        #    raise ValueError("The length of the input vector x must be greater than 3 * max(len(b), len(a)).")
    
        return filtered_data
        
    if "agent" in args.env:
        input_data_path = (
            args.data_path + "/agent/"
        )  # '/home/fujii/workspace3/work/tag/analysis/cae/npy/'
        n_files = 8 # if args.option == "CF" else 5  # npz
        n_files_in = 100  # sample
        args.max_len = 42  # 302
        args.Fs = 1 / 10
        all_indices = np.arange(n_files * 100)
        ds_val_size = 50
        ds_train_end = 400

        # first, compute velocity and acceleration
        accs,vels,vels_next = [],[],[]
        for j in range(n_files):
            if True: #args.option == "CF":
                if j < 4:
                    rep = np.load(
                        input_data_path + "pos_val_rep_2on1_indiv_equal_K_" + str(j) + ".npz",
                        allow_pickle=True,
                        )
                else:
                    rep = np.load(
                        input_data_path + "pos_val_rep_2on1_share_equal_K_" + str(j-4) + ".npz",
                        allow_pickle=True,
                    )
                pos_list = np.array(rep["pos"]).squeeze()
                
            else:   
                rep = np.load(
                    input_data_path + "pos_val_rep_2on1_indiv_K_" + str(j) + ".npz",
                    allow_pickle=True,
                    )
                state_list = np.array(rep["states"]).squeeze()

            for i in range(n_files_in):  # analysis for acceleration 
                flag_ = False
                if True: #args.option == "CF":
                    poss = []
                    if False: # j < 4: # p1,p2,pe
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,0]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,1]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,2]])[None,:])
                    else: # pe,p1,p2,
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,1]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,2]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,0]])[None,:])
                    #for jj in range(3):
                    #    poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,jj]])[None,:])
                    pos = np.concatenate(poss).transpose((1,0,2)) # [frames, agents, pos]
                    if pos.shape[0] > 2:
                        vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos)
                    else:
                        flag_ = True

                    if i == 0 and (j==0 or j==4):
                        # Create a video for the last sample
                        state = np.array(poss[-3:])[:,0,:,:].transpose((1,0,2))  
                        n_mate = args.n_agents
                        n_opponent = args.n_opponents
                        win = 10  # Number of frames to show trajectory
                        fps = 30  # Frames per second

                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.clear()
                        ax.set_xlim(-1.1, 1.1)
                        ax.set_ylim(-1.1, 1.1)
                        ax.set_aspect('equal')
                        ani = animation.FuncAnimation(fig, update_animation_agent, fargs=(state,n_mate,n_opponent), frames=len(state), blit=True, repeat=False)
                        ani.save("./figures/last_sample_video_"+str(j)+".mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
                        plt.close()


                else:
                    state = np.array(state_list[i])  # [pos, vel] x 3
                    if state.shape[0] > 1:
                        vel = np.concatenate([state[:, None, 2:4],state[:, None, 6:8],state[:, None, 10:12]],1,)
                        vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos=None,vel=vel)
                    else:
                        flag_ = True

                if not flag_:          
                    vels.append(vel_abs[:-1])
                    vels_next.append(vel_abs[1:])
                    accs.append(acc_abs)

        if args.trainSize == -1:
            us, ds = compute_ud(vels,vels_next,accs,args)
        else:
            if True: # args.option == "CF":
                all_indices = np.arange(688)
                train_ids = np.concatenate([all_indices[:200],all_indices[379:579]])
                # val_ids = np.concatenate([all_indices[200:225],all_indices[579:604]])
                test_ids = np.concatenate([all_indices[286:311],all_indices[663:688]])
                vels_prev_ = [vels[i] for i in train_ids]
                vels_next_ = [vels_next[i] for i in train_ids]
                accs_ = [accs[i] for i in train_ids]
                us, ds = compute_ud(vels_prev_,vels_next_,accs_,args)

            else:
                if args.trainSize < ds_train_end:
                    indices = np.linspace(0, ds_train_end - 1, args.trainSize, dtype=int)
                    vels_prev_ = [vels[i] for i in indices]
                    vels_next_ = [vels_next[i] for i in indices]
                    accs_ = [accs[i] for i in indices]
                    us, ds = compute_ud(vels_prev_,vels_next_,accs_,args)
                else:
                    us, ds = compute_ud(vels[:args.trainSize],vels_next[:args.trainSize],accs[:args.trainSize],args)

        if args.check_hist:  # show velocity (save png)
            plot_histogram(np.concatenate(accs), args, title='acc_abs', bins=100)
            # str_eliminate = '_all'
            # plot_histogram(vels, args, title = "velocity"+str_eliminate)
            # plot_histogram(viscosity, args, title = "viscosity"+str_eliminate)
            # plot_velocity(accs, args, title = "acceleration")
            print('please check the saved histogram and run without --check_hist')
            sys.exit()

        # compute actions (directions)
        states, actions, rewards, lengths, conditions, vel_abss, rmse_vels, dists = [], [], [], [], [], [], [], []
        ii = 0
        for j in range(n_files):
            if True: # args.option == "CF":
                if j < 4:
                    rep = np.load(
                        input_data_path + "pos_val_rep_2on1_indiv_equal_K_" + str(j) + ".npz",
                        allow_pickle=True,
                        )
                    cond = 0
                else:
                    rep = np.load(
                        input_data_path + "pos_val_rep_2on1_share_equal_K_" + str(j-4) + ".npz",
                        allow_pickle=True,
                    )
                    cond = 1
                pos_list = np.array(rep["pos"]).squeeze()
                
            else:   
                rep = np.load(
                    input_data_path + "pos_val_rep_2on1_indiv_K_" + str(j) + ".npz",
                    allow_pickle=True,
                )
                state_list = np.array(rep["states"]).squeeze()
                action_list = np.array(rep["actions"]).squeeze()
                reward_list = np.array(rep["rewards"]).squeeze()
                cond = 0
                

            for i in range(n_files_in):  #
                flag_ = False
                if True: # args.option == "CF":
                    poss = []
                    if False: # j < 4: # p1,p2,pe
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,0]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,1]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,2]])[None,:])
                    else: # pe,p1,p2,
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,1]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,2]])[None,:])
                        poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,0]])[None,:])
                    #for jj in range(3):
                    #    poss.append(np.vstack([np.array(sublist) for sublist in pos_list[i,jj]])[None,:])
                    pos = np.concatenate(poss).transpose((1,0,2)) # [frames, agents, pos]
                    if pos.shape[0] > 2:
                        vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos)
                        pos_p1,pos_p2,pos_e = pos[:,0],pos[:,1],pos[:,2]
                    else:
                        flag_ = True
                else:
                    state = np.array(state_list[i])  # [pos, vel] x 3
                    if state.shape[0] > 2:
                        vel = np.concatenate([state[:, None, 2:4],state[:, None, 6:8],state[:, None, 10:12]],1,)
                        vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos=None,vel=vel)
                        pos_p1,pos_p2,pos_e = state[:,0:2],state[:,4:6],state[:,8:10]
                        pos = np.concatenate([pos_p1[:,None],pos_p2[:,None],pos_e[:,None]],1)
                    else:
                        flag_ = True

                if not flag_:                   
                    action,vel_abs,rmse_vel = compute_action_reconstruction([pos_p1,pos_p2,pos_e],vel,args,ds,us)
                    dists.append(np.sum(np.sum(vel_abs,0)[:2])*0.1)

                    if not np.isnan(np.sum(rmse_vel)):
                        if len(dists) in test_ids:
                            # if j == 3 or j == 7:
                            # if args.trainSize == -1 or (ii >= ds_train_end and ii < ds_train_end+ds_val_size):
                            rmse_vels.append(rmse_vel)

                    # state,action,reward,length
                    actions.append(action)  # frames x action_dim, next velocity
                    cond_ = np.zeros((vel.shape[0],1)) + cond
                    state = np.concatenate([pos_p1[:-1],vel[:,0],pos_p2[:-1],vel[:,1],pos_e[:-1],vel[:,2],cond_],1)
                    if True: # args.option == "CF":
                        states.append(state)  # [:-1] frames x state_dim, usually position and velocity
                        distance_p1_e = np.linalg.norm(pos_p1[-1] - pos_e[-1])
                        distance_p2_e = np.linalg.norm(pos_p2[-1] - pos_e[-1])
                        if cond == 0:
                            if distance_p1_e <= 0.1:
                                reward_ = np.array([1, 0, 0])
                            elif distance_p2_e <= 0.1:
                                reward_ = np.array([0, 1, 0])
                            else:
                                reward_ = np.array([0, 0, 0])
                        elif cond == 1:
                            if distance_p1_e <= 0.1 or distance_p2_e <= 0.1:
                                reward_ = np.array([1, 1, 0])
                            else:
                                reward_ = np.array([0, 0, 0])
                    else:
                        states.append(state)  # [:-1] frames x state_dim, usually position and velocity
                        reward_ = np.array(reward_list[i][-1]).astype(float)
                        # cond = j
                    reward_[2] = vel.shape[0]/10
                    rewards.append(reward_)
                    lengths.append(vel.shape[0])
                    conditions.append(j)  # dummy in chase-escape
                    vel_abss.append(vel_abs)
                    # if state.shape[0] ==9: # and np.array(reward_list[i]).shape[0]==10: # ii==16: # 
                    #    print(np.array(reward_list[i]).shape[0])
                    ii += 1
        dists = np.array(dists)
                    
    elif args.env == "dragonfly":
        input_data_path = args.data_path
        n_files = 1 # npz
        n_files_in = 93 # sample
        fs = 100
        subsample = 5
        args.Fs = 1 / fs * subsample


        all_indices = n_files_in
    
        rep = np.load(input_data_path + 'dragonfly/data.npz', allow_pickle=True)
        state_list = np.array(rep['states']).squeeze()
        action_list = np.array(rep['actions']).squeeze()
        reward_list = np.array(rep['rewards']).squeeze()

        agent_1 = 0 # dragonfly
        agent_2 = 1 # stone
        time = 1
        vel_own = 1     

        # compute us and ds
        accs,vels,vels_next,length,nans = [],[],[],[],[]
        for i in range(n_files_in):
            epi_len = len(state_list[agent_1][0])
            pos_1 = np.array(state_list[agent_1][i])[::subsample,0,:args.dim].reshape((-1,1,args.dim)) 
            pos_2 = np.array(state_list[agent_2][i])[::subsample,0,:args.dim].reshape((-1,1,args.dim))
            pos = np.concatenate([pos_1,pos_2],1)
            nan_ = np.isnan(pos).any()
            vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos)
            if False:  # show velocity (save png)
                plot_velocity(vel, args)
                plot_velocity(acc_abs, args, title='acc_abs')

            vels.append(vel_abs[:-1])
            vels_next.append(vel_abs[1:])
            accs.append(acc_abs)
            length.append(epi_len)
            nans.append(nan_)
        us, ds = compute_ud(vels,vels_next,accs,args)

        args.max_len = np.max(np.array(length))
        args.min_len = np.min(np.array(length))
        nan_seqs = np.sum(np.array(nans))
        print("max_len: "+str(args.max_len)+", min_len: "+str(args.max_len)+", nan_seqs: "+str(nan_seqs))

        if args.check_hist:  # show velocity (save png)
            # str_eliminate = '_all'
            # vels = np.concatenate(vels)
            # plot_histogram(vels, args, title = "velocity"+str_eliminate)
            plot_histogram(np.concatenate(accs), args, title='acc_abs', bins=200)
            print('please check the saved histogram and run without --check_hist')
            sys.exit()

        # compute actions (directions)
        states, actions, rewards, lengths, conditions, vel_abss, rmse_vels, dists = [], [], [], [], [], [], [], []
        ii = 0
        for i in range(n_files_in):  #
            
            vel = []
            pos_p1 = np.array(state_list[agent_1][i])[::subsample,0,:args.dim].astype(np.float32)
            pos_p2 = np.array(state_list[agent_2][i])[::subsample,0,:args.dim].astype(np.float32)
            pos = np.concatenate([pos_p1[:,None],pos_p2[:,None]],1)
            vel = np.diff(pos,axis=0)/args.Fs

            if vel.shape[0] > 1:
                # only dragonfly (1: stone)
                action,vel_abs,rmse_vel = compute_action_reconstruction([pos_p1,pos_p2],vel,args,ds[0],us[0])

                if not np.isnan(np.sum(rmse_vel)):
                        rmse_vels.append(rmse_vel)

                # state,action,reward,length
                state = np.concatenate([pos[:-1], vel], 1).reshape((-1, args.n_all_agents * args.dim * 2))
                reward_p1 = np.array(reward_list[agent_1][i])[-1]
                reward = np.array([reward_p1])

                states.append(state)  # [:-1] frames x state_dim, usually position and velocity
                actions.append(action)  # frames x action_dim, next velocity
                rewards.append(reward)
                lengths.append(vel.shape[0])
                conditions.append(0)  # dummy
                vel_abss.append(vel_abs)
                # if state.shape[0] ==9: # and np.array(reward_list[i]).shape[0]==10: # ii==16: # 
                #    print(np.array(reward_list[i]).shape[0])
                ii += 1

    elif "fly" == args.env:
        input_data_path = args.data_path
        n_files = 1 # npz
        n_files_in = 114 # sample
        fs = 30
        subsample = 3
        args.Fs = 1 / fs * subsample
        args.max_len = int(1438 / subsample) - 1
        print("max_len: "+str(args.max_len))
        ds_val_size = 20
        ds_train_end = 94

        all_indices = n_files_in
    
        rep = np.load(input_data_path + 'fly/data.npz', allow_pickle=True)
        state_list = np.array(rep['states']).squeeze()
        action_list = np.array(rep['actions']).squeeze()
        reward_list = np.array(rep['rewards']).squeeze()

        agent_m1= 0
        agent_m2 = 1
        agent_f = 2
        time = 1
        vel_own = 1     
        cutoff_frequency = 2

        lengths = []
        vel_abss = []

        # compute us and ds
        accs,vels,vels_next,max_abs_pos = [],[],[],[]
        Filter = True
        for i in range(n_files_in):
            pos_m1 = np.array(state_list[agent_m1][i])[:,0,:2].reshape((-1,1,2)) 
            pos_m2 = np.array(state_list[agent_m2][i])[:,0,:2].reshape((-1,1,2))
            pos_f = np.array(state_list[agent_f][i])[:,0,:2].reshape((-1,1,2))
            pos = np.concatenate([pos_m1,pos_m2,pos_f],1)
            filtered_data = []
            if Filter:
                for p in range(3):                    
                    filtered_ = low_pass_filter(pos[:,p], cutoff_frequency, fs)
                    filtered_data.append(filtered_)
                pos = np.array(filtered_data).transpose((1,0,2))

            pos = pos[::subsample]
            try:  
                norms = np.sqrt(np.sum(pos.astype(np.float32)**2, axis=2))
                max_abs_pos.append(np.max(norms))
            except: import pdb; pdb.set_trace()

            vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos)
            # plot_trajectory(pos,i,0,i)

            if False:  # show velocity (save png)
                plot_velocity(vel, args)
                plot_velocity(acc_abs, args, title='acc_abs')

            vels.append(vel_abs[:-1])
            vels_next.append(vel_abs[1:])
            accs.append(acc_abs)

        max_abs_pos = np.max(np.array(max_abs_pos))
        print("max_abs_pos: "+str(max_abs_pos))
        if args.trainSize == -1:
            us, ds = compute_ud(vels,vels_next,accs,args)
        else:
            if args.trainSize < ds_train_end:
                indices = np.linspace(0, ds_train_end - 1, args.trainSize, dtype=int)
                vels_prev_ = [vels[i] for i in indices]
                vels_next_ = [vels_next[i] for i in indices]
                accs_ = [accs[i] for i in indices]
                us, ds = compute_ud(vels_prev_,vels_next_,accs_,args)
            else:
                us, ds = compute_ud(vels[:args.trainSize],vels_next[:args.trainSize],accs[:args.trainSize],args)

        if args.check_hist:  # show velocity (save png)
            # str_eliminate = '_all'
            # vels = np.concatenate(vels)
            # plot_histogram(vels, args, title = "velocity"+str_eliminate)
            plot_histogram(np.concatenate(accs), args, title='acc_abs', bins=200)
            print('please check the saved histogram and run without --check_hist')
            sys.exit()

        # compute actions (directions)
        states, actions, rewards, lengths, conditions, vel_abss, rmse_vels, term_dist, term_dist2, term_dist3 = [], [], [], [], [], [], [], [], [], []
        ii = 0
        for i in range(n_files_in):  #
            epi_len = len(state_list[agent_m1][0])
            # p1: m1, p2: m2, e: f
            pos_p1 = np.array(state_list[agent_m1][i])[:,0,:2].astype(np.float32)
            pos_p2 = np.array(state_list[agent_m2][i])[:,0,:2].astype(np.float32)
            pos_e = np.array(state_list[agent_f][i])[:,0,:2].astype(np.float32)
            pos = np.concatenate([pos_p1[:,None],pos_p2[:,None],pos_e[:,None]],1)
            filtered_data = []
            if Filter:
                for p in range(3):                    
                    filtered_ = low_pass_filter(pos[:,p], cutoff_frequency, fs)
                    filtered_data.append(filtered_)
                pos = np.array(filtered_data).transpose((1,0,2))
            pos = pos[::subsample]
            vel = np.diff(pos,axis=0)/args.Fs
            pos_p1 = pos[:,0,:]; pos_p2 = pos[:,1,:]; pos_e = pos[:,2,:]

            if vel.shape[0] > 3: # 1
                action,vel_abs,rmse_vel = compute_action_reconstruction([pos_p1,pos_p2,pos_e],vel,args,ds,us)

                reward_p1 = np.array(reward_list[agent_m1][i])[-1]
                reward_p2 = np.array(reward_list[agent_m2][i])[-1]
                reward_e = np.array(reward_list[agent_f][i])[-1]
                reward = np.array([reward_p1,reward_p2,reward_e])

                dist_0 = np.linalg.norm(pos[-1, 0, :] - pos[-1, 2, :])
                dist_1 = np.linalg.norm(pos[-1, 1, :] - pos[-1, 2, :])
                if reward[0] > 0 or reward[1] > 0:
                    if dist_0 < dist_1:
                        # reward[0] = 1
                        # reward[1] = 0
                        term_dist_ = dist_0
                        term_dist.append(dist_0)
                        term_dist2.append(dist_1)
                        term_dist3.append(dist_0)
                    else:
                        # reward[0] = 0
                        # reward[1] = 1
                        term_dist_ = dist_1
                        term_dist.append(dist_1)
                        term_dist2.append(dist_0)
                        term_dist3.append(dist_1)
                else:
                    term_dist_ = np.nan
                    term_dist3.append(np.min([dist_0,dist_1]))


                if not np.isnan(np.sum(rmse_vel)):
                    if args.trainSize == -1 or (i >= ds_train_end and i < ds_train_end+ds_val_size):
                        rmse_vels.append(rmse_vel)
                # else:
                #    import pdb; pdb.set_trace()

                # state,action,reward,length
                state = np.concatenate([pos[:-1], vel], 2).reshape((-1, args.n_all_agents * 4))
                # state = np.concatenate([pos[:-1], vel], 1).reshape((-1, args.n_all_agents * 4))

                states.append(state)  # [:-1] frames x state_dim, usually position and velocity
                actions.append(action)  # frames x action_dim, next velocity
                reward[2] = state.shape[0]/10
                rewards.append(reward)
                lengths.append(vel.shape[0])
                conditions.append(0)  # dummy
                vel_abss.append(vel_abs)
                # if state.shape[0] ==9: # and np.array(reward_list[i]).shape[0]==10: # ii==16: # 
                #    print(np.array(reward_list[i]).shape[0])
                ii += 1
        print('term_dist mean: '+str(np.mean(np.array(term_dist)))+'term_dist max: '+str(np.max(np.array(term_dist))))


    elif args.env == "newt": 
        input_data_path = args.data_path
        n_files = 33 # npz
        n_files_in = 4 # action,condition,reward,state
        fs = 30
        subsample = 3 # 3
        args.Fs = 1 / fs * subsample
        agent_m1= 0
        agent_m2 = 1
        agent_f = 2
        cutoff_frequency = 2
        threshold = 3
        max_len = 500
        min_len = 45
        ds_val_size = 40
        ds_train_end = 240

        indivs = ['m1']#,'m2','f']
        accs,vels,lengths,poss,acc_abss,vel_abss,vels_prev,vels_next,rewards,conditions,term_dist = [],[],[],[],[],[],[],[],[],[],[]
        for j in range(n_files):
            # for indiv in indivs:
            # state_list = np.load(input_data_path + 'newt/state_'+indiv+'_data'+str(j)+'.npz', allow_pickle=True)
            state_list = np.load(input_data_path + 'newt/state_m1_data'+str(j)+'.npz', allow_pickle=True)
            reward_list_m1 = np.load(input_data_path + 'newt/reward_m1_data'+str(j)+'.npz', allow_pickle=True)
            reward_list_m2 = np.load(input_data_path + 'newt/reward_m2_data'+str(j)+'.npz', allow_pickle=True)
            reward_list_f = np.load(input_data_path + 'newt/reward_f_data'+str(j)+'.npz', allow_pickle=True)
            
            seqs = list(state_list.keys())
            no_seq = 0
            for seq in seqs:
                state = state_list[seq]
                reward_m1 = reward_list_m1[seq]
                reward_m2 = reward_list_m2[seq]
                reward_f = reward_list_f[seq]
                reward = np.array([reward_m1[-1], reward_m2[-1], reward_f[-1]])
                epi_len = state.shape[0]

                filtered_data = []
                if True:
                    for p in range(3):
                        filtered_ = low_pass_filter(state[:,2*p], cutoff_frequency, fs) # 1/args.Fs
                        filtered_data.append(filtered_)
                    pos = np.array(filtered_data).transpose((1,0,2))
                else:
                    pos = state.reshape((-1,3,2))
                
                vel,vel_abs,acc,acc_abs = compute_velocity_acceleration(args,pos)


                # Plot the trajectory of pos
                # plot_trajectory(pos,j,indiv,no_seq)
                # create_trajectory_video(pos,vel,j,indiv,no_seq,threshold=1)
                dist_0 = np.linalg.norm(pos[-1, 0, :] - pos[-1, 2, :])
                dist_1 = np.linalg.norm(pos[-1, 1, :] - pos[-1, 2, :])
                if reward[0] > 0 or reward[1] > 0:
                    if dist_0 < dist_1:
                        reward[0] = 1
                        reward[1] = 0
                        term_dist_ = dist_0
                        term_dist.append(dist_0)
                    else:
                        reward[0] = 0
                        reward[1] = 1
                        term_dist_ = dist_1
                        term_dist.append(dist_1)
                else:
                    term_dist_ = np.nan

                if np.max(np.concatenate(vel_abs)) > threshold: 
                    max_index = np.argmax(np.concatenate(vel_abs))
                    rows, cols = vel_abs.shape
                    max_index_2d = (max_index // cols, max_index % cols)  
                    # print('file == ' + str(j) + ' and epi == ' + str(no_seq)+', '+ ' max_index (time, newt): ' + str(max_index_2d), ', max_vel: ' + "{:.2f}".format(np.max(np.concatenate(vel_abs))), ', length: ' + str(rows))
                    # create_trajectory_video(pos,vel,j,indiv,no_seq,threshold=threshold)
                    if state.shape[0] >500 and max_index_2d[0] < 100:
                        pos = pos[max_index_2d[0]:]    
                        vel = vel[max_index_2d[0]:]
                        vel_abs = vel_abs[max_index_2d[0]:]
                        acc = acc[max_index_2d[0]:]
                        acc_abs = acc_abs[max_index_2d[0]:]
                        if np.max(np.concatenate(vel_abs)) <= threshold: 
                            if pos[::subsample].shape[0] > max_len:
                                poss_, vels_, accs_, vel_abss_, acc_abss_, rewards_ = split_data_newt(pos, vel, acc, vel_abs, acc_abs, reward, max_len, subsample)
                                for i in range(len(poss_)):
                                    append_data_newt(poss, vels, accs, vel_abss, acc_abss, vels_prev, vels_next, lengths, conditions, rewards, poss_[i], vels_[i], accs_[i], vel_abss_[i], acc_abss_[i], rewards_[i], subsample)
                                print('(cut: '+ str(len(poss_)) + ') file == ' + str(j) + ' and epi == ' + str(no_seq)+', '+ ' reward: ' + str(reward), ', max_vel: ' + "{:.2f}".format(np.max(np.concatenate(vel_abs))), ', length: ' + str(pos[::subsample].shape[0])+', term_dist: ' + "{:.2f}".format(term_dist_))
                            else:
                                append_data_newt(poss, vels, accs, vel_abss, acc_abss, vels_prev, vels_next, lengths, conditions, rewards, pos, vel, acc, vel_abs, acc_abs, reward, subsample) 
                            print('(cut) file == ' + str(j) + ' and epi == ' + str(no_seq)+', '+ ' reward: ' + str(reward), ', max_vel: ' + "{:.2f}".format(np.max(np.concatenate(vel_abs))), ', length: ' + str(pos[::subsample].shape[0])+', term_dist: ' + "{:.2f}".format(term_dist_))
                elif state.shape[0] < min_len: 
                    dummy = 1
                    # print('file == ' + str(j) + ' and epi == ' + str(no_seq)+', length: ' + str(state.shape[0]))
                else:
                    if pos[::subsample].shape[0] > max_len:
                        poss_, vels_, accs_, vel_abss_, acc_abss_, rewards_ = split_data_newt(pos, vel, acc, vel_abs, acc_abs, reward, max_len, subsample)
                        for i in range(len(poss_)):
                            append_data_newt(poss, vels, accs, vel_abss, acc_abss, vels_prev, vels_next, lengths, conditions, rewards, poss_[i], vels_[i], accs_[i], vel_abss_[i], acc_abss_[i], rewards_[i], subsample)
                        print('(split: '+ str(len(poss_)) + ') file == ' + str(j) + ' and epi == ' + str(no_seq)+', '+ ' reward: ' + str(reward), ', max_vel: ' + "{:.2f}".format(np.max(np.concatenate(vel_abs))), ', length: ' + str(pos[::subsample].shape[0])+', term_dist: ' + "{:.2f}".format(term_dist_))
                    else:
                        append_data_newt(poss, vels, accs, vel_abss, acc_abss, vels_prev, vels_next, lengths, conditions, rewards, pos, vel, acc, vel_abs, acc_abs, reward, subsample)            
                        print('file == ' + str(j) + ' and epi == ' + str(no_seq)+', '+ ' reward: ' + str(reward), ', max_vel: ' + "{:.2f}".format(np.max(np.concatenate(vel_abs))), ', length: ' + str(pos[::subsample].shape[0])+', term_dist: ' + "{:.2f}".format(term_dist_))


                no_seq += 1
        print('term_dist: '+str(np.mean(np.array(term_dist))))

        if args.check_hist: 
            plot_histogram(np.concatenate(vel_abss), args, title='vel_abs', bins=100)
            plot_histogram(np.concatenate(acc_abss), args, title='acc_abs', bins=100)
            print('please check the saved histogram and run without --check_hist')
            sys.exit()

        if args.trainSize == -1:
            us, ds = compute_ud(vels_prev,vels_next,accs,args)
        else:
            if args.trainSize < ds_train_end:
                indices = np.linspace(0, ds_train_end - 1, args.trainSize, dtype=int)
                vels_prev_ = [vels_prev[i] for i in indices]
                vels_next_ = [vels_next[i] for i in indices]
                accs_ = [accs[i] for i in indices]
                us, ds = compute_ud(vels_prev_,vels_next_,accs_,args)
            else:
                us, ds = compute_ud(vels_prev[:args.trainSize],vels_next[:args.trainSize],accs[:args.trainSize],args)

        # compute actions (directions)
        states, actions, rmse_vels = [], [], []
        
        for j in range(len(poss)):
            pos = poss[j]
            vel = vels[j]

            action,vel_abs,rmse_vel = compute_action_reconstruction([pos[:,0,:],pos[:,1,:],pos[:,2,:]],vel,args,ds,us)

            if not np.isnan(np.sum(rmse_vel)):
                if args.trainSize == -1 or (j >= ds_train_end and j < ds_train_end+ds_val_size):
                    rmse_vels.append(rmse_vel)
            else:
                import pdb; pdb.set_trace()

            # state,action
            if pos.shape[0] == vel.shape[0]:
                state = np.concatenate([pos, vel], 2).reshape((-1, args.n_all_agents * 4))
            elif pos.shape[0] == vel.shape[0]+1:
                state = np.concatenate([pos[:-1], vel], 2).reshape((-1, args.n_all_agents * 4))
            else: import pdb; pdb.set_trace()
            states.append(state)  # [:-1] frames x state_dim, usually position and velocity
            actions.append(action)  # frames x action_dim, next velocity

    elif args.env == "silkmoth":
        input_data_path = args.data_path + "silkmoth/Convertdata/Output2" # VRmovie/"
        input_video_path = args.data_path + "silkmoth/Convertdata" 
        foldernames = ["AllStim","Odoronly"] # ["data_cont","data_odor",]
        videonames = [
            "Movie1",
            "Movie2",
            "Movie3",
            "Movie4",
            "Movie5",
        ]
        videonames_raw = [
            "1Hz-1_res",
            "1Hz-2_res",
            "1Hz-3_res",
            "1Hz-4_res",
            "1Hz-5_res",
        ]
        n_folders = len(foldernames)
        n_videos = len(videonames)
        n_files = 6
        fs = 30
        max_len_ = 300 * fs - 1  # np.inf

        subsample = 15
        args.Fs = 1 / fs * subsample
        args.max_len = 300 * int(1 / args.Fs) - 1
        max_len = args.max_len

        indices = np.arange(n_files)
        args.n_files = n_files
        args.n_folders = n_folders


        states, actions, rewards, lengths, conditions, vel_abss, rewards_cond, images = [], [], [], [], [], [], [], []

        # outputfilename_video = args.result_path + args.env + "_video.npz" # "+str(1/args.Fs)+"_Hz

        if False: # video to npy
            gray_frames = []
            for j in range(n_videos):
                filename = (
                    input_video_path
                    + os.sep
                    + "movie"
                    + os.sep
                    + videonames_raw[j]
                    + ".mp4"
                )
                cap = cv2.VideoCapture(filename)

                # initial frame
                ret, frame = cap.read()
                width = frame.shape[1] # 512
                height = frame.shape[0] # 640

                # initial array
                # gray_frames = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), height, width), dtype=np.uint8)
                print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = [image[120:120+401,:501]]#  [cv2.resize(image, (501,627))[113:113+401]] # 401,501
                i = 0
                while True:
                    if i > 0: # and i % subsample == 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret:
                            print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                            break
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        resized_image = image[120:120+401,:501] # cv2.resize(image, (501, 627))[113:113+401]
                        gray_frame.append(resized_image) 
                        print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    i += 1

                cap.release()
                try: gray_frames.append(np.stack(gray_frame))
                except: import pdb; pdb.set_trace()
            # np.savez(outputfilename_video, gray_frames, allow_pickle=True)
        
            # if False: # seperate video to npz
            # rep = np.load(args.result_path + args.env + "_video.npz", allow_pickle=True)
            # gray_frames = rep["arr_0"]

            for j in range(n_videos):
                # video input
                gray_frame_ = gray_frames[j] # .reshape((gray_frames[j].shape[0],-1))
                gray_frame_ = np.concatenate([gray_frame_,gray_frame_],0) # for long trials
                # image = gray_frame_[::subsample][:int((max_len_+1)/subsample)] # [:pos.shape[0]-1]
                for t in range(int((max_len_+1)/subsample)+5):
                    if t == 0:
                        image = [gray_frame_[0:1].repeat(subsample,0)]
                    else:
                        image.append(gray_frame_[t*subsample+1:(t+1)*subsample+1])
                image = np.stack(image)
                outputfilename_video = args.result_path + args.env + "_video_"+str(j)+"_"+str(1/args.Fs)+"_Hz.npz" # 
                np.savez(outputfilename_video, image, allow_pickle=True)
        if True: # load
            j = 0
            outputfilename_video = args.result_path + args.env + "_video_"+str(j)+"_"+str(1/args.Fs)+"_Hz.npz" # 
            images = np.load(outputfilename_video, allow_pickle=True)["arr_0"]

        # compute us and ds
        vels, vels_next, accs = [], [], []
        for j in range(n_videos):
            for i in range(n_folders):
                for k in range(n_files):
                    filename = (
                        input_data_path
                        + os.sep
                        + foldernames[i]
                        + os.sep
                        + videonames[j]
                        + os.sep
                        + "trial"
                        + str(k + 1)
                        + "_mod.csv"
                    )
                    df = pd.read_csv(filename, header=None, names=['time', 'x', 'y', 'angle', 'left_odor', 'right_odor', 'odor', 'wind'])

                    T = len(df)


                    if False:  # filter (not used; unnecessary)
                        pos = df.values[:, 1:3] / 1000
                        theta = df.values[:,3].astype(float)
                        from scipy.signal import butter, lfilter

                        cutoff = 5  # Hz
                        order = 2
                        b, a = butter(order, cutoff, fs=fs, btype="low")
                        pos_filt = lfilter(b, a, pos, axis=0)
                        theta_filt = lfilter(b, a, theta)

                        if False:  # show filtering results (save png)
                            show_filter(pos, pos_filt, theta, theta_filt, args)

                    pos = df.values[::subsample, 1:3] / 1000
                    vel = np.diff(pos, axis=0) / args.Fs
                    vel_abs = (np.sum(vel**2, axis=1) + 1e-16) ** (1 / 2)

                    acc = np.diff(vel,axis=0)/args.Fs
                    acc_abs = (np.sum(acc.reshape((-1, args.n_all_agents,2))**2, axis=2) + 1e-6) ** (1 / 2)
                    # vel_eliminated = vel_abs[:-1]>1e-2
                    # vels_next.append(vel_eliminated)

                    if False:  # show velocity (save png)
                        plot_velocity(vel, args)
                        plot_velocity(acc_abs, args, title='acc_abs')

                    vels.append(vel_abs[:-1])
                    vels_next.append(vel_abs[1:])
                    accs.append(acc_abs)
        if False:
            plot_histogram(np.concatenate(accs), args, title='acc_abs', bins=50)
        
        if args.trainSize == -1:
            us,ds = compute_ud(vels,vels_next,accs,args)
        else:
            us, ds = compute_ud(vels[:args.trainSize],vels_next[:args.trainSize],accs[:args.trainSize],args)


        rmse_vels = []
        # compute other variables
        for i in range(n_folders):
            reward_cond = []
            for j in range(n_videos):
                for k in range(n_files):
                    try: foldernames[i]
                    except: import pdb; pdb.set_trace()
                    filename = (
                        input_data_path
                        + os.sep
                        + foldernames[i]
                        + os.sep
                        + videonames[j]
                        + os.sep
                        + "trial"
                        + str(k + 1)
                        + "_mod.csv"
                    )
                    df = pd.read_csv(filename)

                    T = len(df)

                    pos = df.values[::subsample, 1:3].astype(float) / 1000
                    vel = np.diff(pos, axis=0) / args.Fs
                    vel_abs = (np.sum(vel**2, axis=1) + 1e-16) ** (1 / 2)
                    theta = df.values[::subsample,3].astype(float)

                    reward = np.array([1]) if T < max_len_ else np.array([0])

                    vel = np.concatenate([vel, vel[-1:]], 0)
                    acc = np.diff(vel, axis=0) / args.Fs

                    # average using 3 frame data ([0,1], [14,15,16], ...) (but not used)
                    if False: 
                        window,theta__ = [],[]
                        for l in range(len(theta)):
                            window.append(theta[l])
                            if l == 1 or len(window) == subsample:
                                theta__.append(np.mean(window[-3:]))
                                window = []  # pop first element
                        theta__ = np.array(theta__)
                    # compute angle between velocity and position (Fig 1 in eLife paper) from -180 deg to 180 deg
                    theta_ = angle_vel_pos_silkmoth(pos, vel)
                    '''dot_product = np.sum(-pos*vel, axis=1) 
                    norm_product = np.linalg.norm(vel,axis=1) * np.linalg.norm(pos,axis=1) +1e-6
                    angle_rad = np.arccos(dot_product / norm_product)
                    theta_ = np.rad2deg(angle_rad)
                    sign_theta = [np.cross(vel, -pos, axisa=1,axisb=1) < 0]
                    theta_[tuple(sign_theta)] = -theta_[tuple(sign_theta)]'''
                    # Due to the difference of theta_ (simulatable) and theta__ (GT), we use theta_ for the following analysis

                    # compute actions (directions)
                    
                    static_input = False
                    if static_input:  # static input
                        wind = np.ones((pos.shape[0], 1))
                        vision = np.ones((pos.shape[0], 1))
                        if foldernames[i] == "data_odor": # Cond1: odor only
                            wind[:] = np.array([0.0]).astype("float32")
                            vision[:] = np.array([0.0]).astype("float32")
                        elif foldernames[i] == "data_cont": # Condi: cont (all modalities)
                            wind[:] = np.array([1.0]).astype("float32")
                            vision[:] = np.array([1.0]).astype("float32")
                        state = np.concatenate(
                            [pos[:-1], vel[:-1], wind[:-1], vision[:-1]], 1
                        )  # ,sin[:-1],cos[:-1],
                    else:
                        # video input
                        '''gray_frame_ = gray_frames[j] # .reshape((gray_frames[j].shape[0],-1))
                        gray_frame_ = np.concatenate([gray_frame_,gray_frame_],0) # for long trials
                        # image = gray_frame_[::subsample][:pos.shape[0]-1]
                        for t in range(pos.shape[0]-1):
                            if t == 0:
                                image = [gray_frame_[0:1].repeat(subsample,0)]
                            else:
                                image.append(gray_frame_[t*subsample:(t+1)*subsample])
                        image = np.stack(image)'''
                        dynamic_discrete_input = False  # it is difficult to use without a simulator
                        if dynamic_discrete_input:
                            odor_ = df.values[:, 4]
                            odor = np.zeros((T, 2))  # N:[0 0]，R:[1 0]，L:[0 1]，B:[1 1]
                            odor[np.where(odor_ == "R")[0]] = np.array([1, 0])
                            odor[np.where(odor_ == "L")[0]] = np.array([0, 1])
                            odor[np.where(odor_ == "B")[0]] = np.array([1, 1])

                            wind_ = df.values[:, 5]
                            wind = np.zeros((T, 4))  # F:[1 0 0 0]，R:[0 1 0 0]，L:[0 0 1 0]，B:[0 0 0 1]
                            wind[np.where(wind_ == "F")[0]] = np.array([1, 0, 0, 0])
                            wind[np.where(wind_ == "R")[0]] = np.array([0, 1, 0, 0])
                            wind[np.where(wind_ == "L")[0]] = np.array([0, 0, 1, 0])
                            wind[np.where(wind_ == "B")[0]] = np.array([0, 0, 0, 1])

                            vision_ = df.values[:, 6]
                            vision = np.zeros((T, 2))  # N:[0 0]，R:[1 0]，L:[0 1]
                            vision[np.where(vision_ == "R")[0]] = np.array([1, 0])
                            vision[np.where(vision_ == "L")[0]] = np.array([0, 1])

                            # subsample
                            odor__ = odor.copy()
                            odor = odor__[::subsample]
                            odor_diff = np.diff(odor__, axis=0).sum(axis=1)
                            odor_diff = np.where(np.abs(odor_diff) > 0)[0]
                            for n in odor_diff:
                                if np.mod(n + 1, subsample) > 0:
                                    odor[int(np.floor((n + 1) / subsample))] = odor__[n + 1]
                            wind__ = wind.copy()
                            wind = wind__[::subsample]
                            wind_diff = np.diff(wind__, axis=0).sum(axis=1)
                            wind_diff = np.where(np.abs(wind_diff) > 0)[0]
                            for n in wind_diff:
                                if np.mod(n + 1, subsample) > 0:
                                    wind[int(np.floor((n + 1) / subsample))] = wind__[n + 1]
                            vision__ = vision.copy()
                            vision = vision__[::subsample]
                            vision_diff = np.diff(vision__, axis=0).sum(axis=1)
                            vision_diff = np.where(np.abs(vision_diff) > 0)[0]
                            for n in vision_diff:
                                if np.mod(n + 1, subsample) > 0:
                                    vision[int(np.floor((n + 1) / subsample))] = vision__[n + 1]

                            state = np.concatenate(
                                [pos[:-1], vel[:-1], odor[:-1], wind[:-1], vision[:-1]], 1
                            )  # ,sin[:-1],cos[:-1],
                        else:
                            odor = df.values[::subsample, 4:6].astype(float)
                            if i == 0:
                                vis = df.values[::subsample, 6:7].astype(float)
                                wind_ = df.values[::subsample, 7].astype(int)
                                wind = np.identity(4)[wind_] # F:[1 0 0 0]，R:[0 1 0 0]，L:[0 0 1 0]，B:[0 0 0 1]
                            else:
                                vis = np.zeros((pos.shape[0], 1))
                                wind = np.zeros((pos.shape[0], 4))
                            odor_vis = np.concatenate([odor, vis], 1)
                            cond_ = np.repeat(np.array([i]),pos.shape[0]-1,axis=0)[:,None]
                            state = np.concatenate([pos[:-1], vel[:-1], theta_[:-1,None], odor_vis[:-1], wind[:-1],cond_], 1)

                    # check odor, vision, and wind using theta (GT angle)
                    if True:
                        odor_vis_winds = []
                        tdlr_rstimLR = [[0, 0, np.zeros((3,)), np.zeros((3,)), 0, 0]]
                        for t in range(state.shape[0]-1):
                            odor_vis_wind,tdlr_rstimLR_ = compute_odor_vis_wind_silkmoth(theta[t],theta[t+1],
                                state[t+1], odor_vis[t,:2], images[t], tdlr_rstimLR[t], t, args.Fs, cond=0) # , angle=True)
                            tdlr_rstimLR.append(tdlr_rstimLR_)
                            odor_vis_winds.append(odor_vis_wind)
                        
                        odor_vis_winds = np.array(odor_vis_winds)
                        # compare odor_vis_winds and odor_vis, but perfect correspondence is not expected in principle
                        
                    # compute actions (directions)        
                    length = state.shape[0]

                    acc = (vel[1:] - vel[:-1]*(1-ds))*args.Fs

                    # if args.option == "left":
                    origin = np.array([-10000,0])
                    #else:
                    #    origin = np.array([0,0])
                    action, direction = discrete_direction(pos[:-1],origin.astype(float),acc,args)

                    # not perfect: due to viscosity 
                    action = np.concatenate([action[:,None],action[-1:,None]],0) 

                    # reconstruction
                    vel_next_rec = vel[:-1]*(1-ds)+us*direction*args.Fs
                    rmse_vel = np.sqrt(np.mean(np.sum((vel_next_rec[:-1] - vel[1:-1])**2,1),0))
                    if not np.isnan(np.sum(rmse_vel)):
                        rmse_vels.append(rmse_vel)

                    end_dist = np.sqrt(np.sum(pos[-3] ** 2))

                    print(
                        "folder: "
                        + str(i)
                        + ", file: "
                        + str(j + 1)
                        + ", length: "
                        + str(len(df))
                        + ", end_dist: "
                        + str(end_dist)
                    )

                    # state,action,reward,length
                    states.append(state)  # frames x state_dim
                    actions.append(action)  # frames x action_dim
                    rewards.append(reward)
                    lengths.append(length)
                    conditions.append(filename)
                    vel_abss.append(vel_abs)
                    # images.append(image)

                    reward_cond.append(reward)
            rewards_cond.append(np.array(reward_cond))

    elif args.env == "bat":
        import preprocess_bat.main as pre_bat

        input_data_path = args.data_path  # "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
        (
            states,
            actions,
            rewards,
            lengths,
            conditions,
            vel_abss,
        ) = pre_bat.preprocess_bat(input_data_path, episode_sec)

    # summarize
    rewards = np.array(rewards)  # frames x 1
    lengths = np.array(lengths)  # frames
    vel_abss = np.concatenate(vel_abss, axis=0)  # all frames
    rmse_vels = np.array(rmse_vels) 
    conditions = np.array(conditions)  

    if args.env == "silkmoth":
        print(
            "mean reward: " + str(np.mean(np.array(rewards_cond), axis=1).squeeze())
        )  # [0.66666667,0.8,0.73333333,0.63333333,0.73333333,0.9,0.4,0.83333333]
    elif "agent" in args.env: #  and args.option == "CF":
        print("mean rewards (indiv): " + str(np.mean(rewards[conditions<4], 0)))
        print("mean rewards (share): " + str(np.mean(rewards[conditions>=4], 0)))
        print("mean dist (indiv): " + str(np.mean(dists[conditions<4], 0)))
        print("mean dist (share): " + str(np.mean(dists[conditions>=4], 0)))
    else:
        print("mean rewards: " + str(np.mean(rewards, 0)))

    # histogram
    plot_histogram(vel_abss, args)

    # statistics
    print("max length: " + str(np.max(lengths)))
    print("min length: " + str(np.min(lengths)))
    print("sum of length: " + str(np.sum(lengths)))
    print("max vel: " + str(vel_abss.max(0)) + ", min vel: " + str(vel_abss.min(0)))
    print("rmse vel: " + str(np.mean(rmse_vels,0)) )

    # save
    # preprocessed = [states, actions, rewards, lengths, conditions, us, ds, rmse_vels]
    if args.option is not None:
        outputfilename = args.result_path + args.env + "_"+ str(args.trainSize) + "_"+args.option+".npz"
    else:
        outputfilename = args.result_path + args.env + "_"+ str(args.trainSize) + ".npz"
    '''if args.env == "silkmoth":
        np.savez(
            outputfilename, states, images, actions, rewards, lengths, conditions, us, ds, rmse_vels, allow_pickle=True
        )
    else:'''
    np.savez(
        outputfilename, states, actions, rewards, lengths, conditions, us, ds, rmse_vels, allow_pickle=True
    )
    print(
        str(rewards.shape[0])
        + " sequences and "
        + str(lengths.sum())
        + " frames in total at "
        + outputfilename
    )

    print("preprocess is finished and a debugger is activated")
    import pdb; pdb.set_trace()
