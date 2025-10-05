import numpy as np
from numpy.linalg import norm
import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None
    
class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def calc_distance_and_neighbor_point(a, b, p):
    # line segement: ab
    # point: p
    ap = p - a
    ab = b - a
    ba = a - b
    bp = p - b
    if np.dot(ap, ab) < 0:
        distance = norm(ap)
        neighbor_point = a
    elif np.dot(bp, ba) < 0:
        distance = norm(p - b)
        neighbor_point = b
    else:
        ai_norm = np.dot(ap, ab)/norm(ab+1e-6)
        neighbor_point = a + (ab)/norm(ab+1e-6)*ai_norm
        distance = norm(p - neighbor_point)
    return (neighbor_point, distance)

def discrete_direction(angle,no_angles=8):
    if no_angles == 8:
        if np.abs(angle) >= 7/np.pi: # left
            discrete_angle = 1
        elif angle >= 5/np.pi and angle < 7/np.pi: # left top
            discrete_angle = 2
        elif angle >= 3/np.pi and angle < 5/np.pi: # top
            discrete_angle = 3
        elif angle >= 1/np.pi and angle < 3/np.pi: # right top
            discrete_angle = 4
        elif np.abs(angle) < 1/np.pi: # right
            discrete_angle = 5
        elif angle >= -3/np.pi and angle < -1/np.pi: # right bottom
            discrete_angle = 6
        elif angle >= -5/np.pi and angle < -3/np.pi: # bottom
            discrete_angle = 7
        elif angle >= -7/np.pi and angle < -5/np.pi: # left bottom
            discrete_angle = 8
    else:
        error('not defined other than no_angles == 8')
    return discrete_angle 


def get_observation_from_state(state,n_agents,n_mate,n_adv,selfpos=True,selfdist=False):
    # n_mate = n_all_agents - n_adv
    obss = []
    T = state.shape[0]
    if n_mate == 2 and n_adv == 1:
        pos_p1 = state[:,:2]
        vel_p1 = state[:,2:4]
        pos_p2 = state[:,4:6]
        vel_p2 = state[:,6:8]
        pos_e = state[:,8:10]
        vel_e = state[:,10:12]

        for t in range(T):
            state_p1 = [pos_p1[t], vel_p1[t], pos_p2[t], vel_p2[t], pos_e[t], vel_e[t]]
            state_p2 = [pos_p2[t], vel_p2[t], pos_p1[t], vel_p1[t], pos_e[t], vel_e[t]]
            state_e = [pos_e[t], vel_e[t], pos_p1[t], vel_p1[t], pos_p2[t], vel_p2[t]]
            if selfdist:
                dist_org_p1 = np.array([np.sqrt(np.sum(pos_p1[t]**2))])[None,:]
                dist_org_p2 = np.array([np.sqrt(np.sum(pos_p2[t]**2))])[None,:]
                dist_org_e = np.array([np.sqrt(np.sum(pos_e[t]**2))])[None,:]

            obs = get_obs_p(state_p1,n_mate,n_adv) # obs_p1
            if selfdist:
                obs = np.concatenate([obs,dist_org_p1],1)

            if selfdist:
                obs_ = np.concatenate([get_obs_p(state_p2,n_mate,n_adv),dist_org_p2],1)
                obs = np.concatenate([obs,obs_],0) # obs_p2
            else:
                obs = np.concatenate([obs,get_obs_p(state_p2,n_mate,n_adv)],0) # obs_p2

            if n_agents == 3:
                if selfdist:
                    obs_ = np.concatenate([get_obs_e(state_e,n_mate,n_adv),dist_org_e],1)
                    obs = np.concatenate([obs,obs_],0) # obs_e
                else:
                    obs = np.concatenate([obs,get_obs_e(state_e,n_mate,n_adv)],0) # obs_e

            obss.append(obs[:,np.newaxis])
    elif n_mate == 1 and n_adv == 0:
        if selfpos:
            obss = [state[None,:].astype('float64')]
        else:
            obss = [state[None,:,2:].astype('float64')] # .transpose((0,2,1))

    else:
        print('not n_mate == 2 and n_adv == 1')
        import pdb; pdb.set_trace()

    obss = np.concatenate(obss,1)
    return obss


def get_obs_p(state,n_mate,n_adv):
    if n_mate == 2 and n_adv == 1:
        pos_p1, vel_p1, pos_p2, vel_p2, pos_e, vel_e = state
        sub_pos_mate = get_sub_pos(pos_p1, pos_p2)
        sub_pos_adv = get_sub_pos(pos_p1, pos_e)
        
        sub_vel_own_mate = get_sub_vel(pos_p1, pos_p2, vel_p1)
        sub_vel_own_adv = get_sub_vel(pos_p1, pos_e, vel_p1)
        
        sub_vel_mate = get_sub_vel(pos_p1, pos_p2, vel_p2)
        sub_vel_adv = get_sub_vel(pos_p1, pos_e, vel_e)

        '''obs_p = np.concatenate([pos_p1] + [vel_p1] + \
                            [sub_pos_mate] + [vel_p2] + \
                            [sub_pos_adv] + [vel_e]).reshape(1,12)

        obs_p = np.concatenate([pos_p1] + [sub_vel_own_mate] + [sub_vel_own_adv] + \
                            [pos_p2] + [sub_pos_mate] + [sub_vel_mate] + \
                            [pos_e] + [sub_pos_adv] + [sub_vel_adv]).reshape(1,18)''' 

        obs_p = np.concatenate([pos_p1] + [sub_vel_own_mate] + [sub_vel_own_adv] + \
                            [pos_p2] + [sub_pos_mate] + [sub_vel_mate] + \
                            [pos_e] + [sub_pos_adv] + [sub_vel_adv]).reshape(1,18) 

    elif n_mate == 1 and n_adv == 0:
        import pdb; pdb.set_trace()

    else:
        print('not a defined case')
        import pdb; pdb.set_trace()

    return obs_p

def get_obs_e(state,n_mate,n_adv):
    if n_mate == 2 and n_adv == 1:
        pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp, pos_e, vel_e = state
        state_e = [pos_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp]
    
        pos_p1, vel_p1, pos_p2, vel_p2 = get_order_adv(state_e,n_mate,n_adv)
        
        sub_pos_adv1 = get_sub_pos(pos_e, pos_p1)
        sub_pos_adv2 = get_sub_pos(pos_e, pos_p2)

        '''obs_e = np.concatenate([pos_e] + [vel_e] + \
                            [sub_pos_adv1] + [vel_p1] + \
                            [sub_pos_adv2] + [vel_p2]).reshape(1,12)'''
                            
        sub_vel_own_adv1 = get_sub_vel(pos_e, pos_p1, vel_e)
        sub_vel_own_adv2 = get_sub_vel(pos_e, pos_p2, vel_e)
            
        sub_vel_adv1 = get_sub_vel(pos_e, pos_p1, vel_p1)
        sub_vel_adv2 = get_sub_vel(pos_e, pos_p2, vel_p2)
                
        obs_e = np.concatenate([pos_e] + [sub_vel_own_adv1] + [sub_vel_own_adv2] + \
                            [pos_p1] + [sub_pos_adv1] + [sub_vel_adv1] + \
                            [pos_p2] + [sub_pos_adv2] + [sub_vel_adv2]).reshape(1,18)
    else:
        print('not n_mate == 2 and n_adv == 1')
        import pdb; pdb.set_trace()

    return obs_e   

def get_order_adv(state,n_mate,n_adv):
    if n_mate == 2 and n_adv == 1:
        pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp = state
        dist1 = get_dist(pos_own, pos_adv1_tmp)
        dist2 = get_dist(pos_own, pos_adv2_tmp)
        
        d = [dist1, dist2]
        p = [pos_adv1_tmp, pos_adv2_tmp]
        v = [vel_adv1_tmp, vel_adv2_tmp]
        list_old = list(zip(d, p, v))

        if isinstance(dist1, np.ndarray) or isinstance(dist1, float) or dist1.dtype == np.float32:
            if np.abs(dist1-dist2)<1e-10:
                list_new = list_old
            else:
                list_new = sorted(list_old) # l.sort()
        else:
            if torch.abs(dist1-dist2)<1e-10:
                list_new = list_old
            else:
                list_new = sorted(list_old)

        d, p, v = zip(*list_new)
        
        pos_adv1, vel_adv1 = p[0], v[0] 
        pos_adv2, vel_adv2 = p[1], v[1]

        state_ = [pos_adv1, vel_adv1, pos_adv2, vel_adv2]
    else:
        print('not n_mate == 2 and n_adv == 1')
        import pdb; pdb.set_trace()
        
    return state_

def get_sub_pos(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_pos = np.dot(rot, pos_rel)

    return sub_pos


def get_sub_vel(abs_pos_own, abs_pos_adv, abs_vel):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_vel = np.dot(rot, abs_vel)

    return sub_vel


def get_abs_u(action, abs_own_pos, abs_adv_pos):
    if action <= 11:
        ang = action * -np.pi / 6
        if isinstance(ang, np.ndarray) or isinstance(ang, float) or ang.dtype == np.float32:
            sub_u = [np.cos(ang), np.sin(ang)]  
        else:
            sub_u = [torch.cos(ang), torch.sin(ang)]      
        abs_u = rotate_u(sub_u, abs_own_pos, abs_adv_pos)
    elif action == 12:
        if isinstance(abs_own_pos, np.ndarray) or isinstance(abs_own_pos, float) or abs_own_pos.dtype == np.float32:
            abs_u = np.array([0, 0])
        else:
            abs_u = torch.tensor([0, 0])
    
    return abs_u


def rotate_u(sub_u, abs_pos_own, abs_pos_adv):
    if isinstance(sub_u, np.ndarray):
        sub_u = np.array(sub_u)
    else:
        sub_u = torch.tensor(sub_u)
    pos_rel = abs_pos_adv - abs_pos_own
    if isinstance(abs_pos_own, np.ndarray):
        theta = np.arctan2(pos_rel[1], pos_rel[0])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        abs_u = np.dot(rot, sub_u) 
    else:
        theta = torch.atan2(pos_rel[1], pos_rel[0])
        rot = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
        abs_u = torch.matmul(rot, sub_u) 
    return abs_u


def get_next_own_state(abs_pos_own, abs_vel_own, abs_u, mass, speed, damping, dt):
    
    if isinstance(abs_u, np.ndarray):
        abs_acc_own = np.array(abs_u) / mass
        next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * speed * dt
    else:
        if torch.cuda.is_available():
            abs_u = abs_u.cuda()
        abs_acc_own = abs_u / mass
        next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * speed * dt
       
    next_abs_pos_own = abs_pos_own + next_abs_vel_own * dt
    return next_abs_pos_own, next_abs_vel_own


def get_dist(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    if isinstance(pos_rel, np.ndarray):
        dist = np.sqrt(np.sum(np.square(pos_rel)))
    else:
        dist = torch.sqrt(torch.sum(torch.square(pos_rel)))

    return dist

def transition_agent(state, action_p1, action_p2, action_e, params, boundary=None):
    """
    Calculate the next state given the current state and actions for each predator and the prey.
    
    Parameters:
    - state: Current state of the game, including positions and velocities of predators and the prey.
    - action_p1: Action taken by predator 1.
    - action_p2: Action taken by predator 2.
    - action_e: Action taken by the prey.
    - params: A dictionary containing parameters such as mass, speed, damping, and dt for predators and the prey.
    
    Returns:
    - A tuple of next states for predator 1, predator 2, and the prey.
    """
    
    # Extract positions and velocities from the current state
    pos_p1, vel_p1 = state[:2], state[2:4]
    pos_p2, vel_p2 = state[4:6], state[6:8]
    pos_e, vel_e = state[8:10], state[10:12]
    
    # Calculate absolute control input for predator 1 and compute the next state
    abs_u_p1 = get_abs_u(action_p1, pos_p1, pos_e).T
    next_pos_p1, next_vel_p1 = get_next_own_state(pos_p1, vel_p1, abs_u_p1, 
                                                  params['mass_p1'], params['speed_p1'], params['damping_p1'], params['dt'])
    
    # Calculate absolute control input for predator 2 and compute the next state
    abs_u_p2 = get_abs_u(action_p2, pos_p2, pos_e).T
    next_pos_p2, next_vel_p2 = get_next_own_state(pos_p2, vel_p2, abs_u_p2, 
                                                  params['mass_p2'], params['speed_p2'], params['damping_p2'], params['dt'])
    
    # Determine the adversary's target position and calculate the next state for the prey
    state_adv = [pos_e, pos_p1, vel_p1, pos_p2, vel_p2]
    pos_adv1, _, _, _ = get_order_adv(state_adv, params['n_mate'], params['n_adv'])
    abs_u_e = get_abs_u(action_e, pos_e, pos_adv1).T
    next_pos_e, next_vel_e = get_next_own_state(pos_e, vel_e, abs_u_e, 
                                                params['mass_e'], params['speed_e'], params['damping_e'], params['dt'])
    
    if boundary is not None:
        next_pos_p1, next_vel_p1 = boundary_condition(next_pos_p1, next_vel_p1, boundary)
        next_pos_p2, next_vel_p2 = boundary_condition(next_pos_p2, next_vel_p2, boundary)
        next_pos_e, next_vel_e = boundary_condition(next_pos_e, next_vel_e, boundary)

    # Concat the next states for each entity
    state_p1 = [next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_e, next_vel_e]
    state_p2 = [next_pos_p2, next_vel_p2, next_pos_p1, next_vel_p1, next_pos_e, next_vel_e]
    state_e = [next_pos_e, next_vel_e, next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2]

    if isinstance(next_pos_p1, np.ndarray):
        return np.array(state_p1), np.array(state_p2), np.array(state_e), next_pos_p1, next_pos_p2, next_pos_e
    else:
        return torch.stack(state_p1), torch.stack(state_p2), torch.stack(state_e), next_pos_p1, next_pos_p2, next_pos_e

def boundary_condition(next_pos, next_vel, boundary_type='square', rebound_coeff=0.5):
    """
    Returns the corrected next_pos and next_vel when next_pos reaches the boundary
    of a square with side length 2 or a circle with radius 1, bouncing back with a given rebound coefficient.

    Parameters:
    - next_pos: Next position (numpy array)
    - next_vel: Next velocity (numpy array)
    - boundary_type: 'square' or 'circle' (str)
    - rebound_coeff: Rebound coefficient (float)

    Returns:
    - Corrected next_pos and next_vel (tuple of numpy arrays)
    """
    if boundary_type == 'square':
        # Square boundary condition
        for i in range(len(next_pos)):
            if next_pos[i] < -1:
                next_pos[i] = -1
                next_vel[i] = -next_vel[i] * rebound_coeff
            elif next_pos[i] > 1:
                next_pos[i] = 1
                next_vel[i] = -next_vel[i] * rebound_coeff
    elif boundary_type == 'circle':
        # Circular boundary condition
        if np.linalg.norm(next_pos) > 1:
            norm_pos = next_pos / np.linalg.norm(next_pos)
            next_pos = norm_pos
            # Update next_vel to reflect the bounce back effect based on the angle of incidence and reflection, considering the rebound coefficient.
            next_vel = next_vel - 2 * np.dot(next_vel, norm_pos) * norm_pos * rebound_coeff
    else:
        raise ValueError("boundary_type must be 'square' or 'circle'")

    return next_pos, next_vel

def transition_single(state, action, params):
    """
    Calculate the next state given the current state and actions for a single agent.
    
    Parameters:
    - state: Current state of the game, including positions and velocities of agent.
    - action: Action taken by agent.
    - params: A dictionary containing parameters such as mass, speed, damping, and dt.
    
    Returns:
    - A tuple of next states for a agent.
    """
    
    # Extract positions and velocities from the current state
    if len(state.shape) == 1:
        pos, vel = state[:2], state[2:4]
    else:
        pos, vel = state[:2,0], state[2:4,0]
    
    # Calculate absolute control input and compute the next state
    try: abs_u = get_abs_u(action, pos, params["origin"]).T
    except: import pdb; pdb.set_trace()
    
    next_pos, next_vel = get_next_own_state(pos, vel, abs_u, 
                                                params['mass'], params['speed'], params['damping'], params['dt'])
    
    # Concat the next states 
    state = [next_pos, next_vel]
    if len(np.array(state).shape) == 3:
        state = [next_pos[0], next_vel[0]]
    if isinstance(next_pos, np.ndarray):
        return np.array(state), next_pos
    else:
        return torch.stack(state), next_pos
    
def angle_vel_pos_silkmoth(pos, vel):
    # compute angle between velocity and position (Fig 1 in eLife paper) from -180 deg to 180 deg
    if len(pos.shape) == 1:
        pos = pos[None,:]
        vel = vel[None,:]
    elif pos.shape[1] == 1:
        pos = pos.T
        vel = vel.T

    dot_product = np.sum(-pos*vel, axis=1) 
    norm_product = np.linalg.norm(vel,axis=1) * np.linalg.norm(pos,axis=1) +1e-6
    angle_rad = np.arccos(dot_product / norm_product)
    theta_ = np.rad2deg(angle_rad)
    sign_theta = [np.cross(vel, -pos, axisa=1,axisb=1) < 0]
    theta_[tuple(sign_theta)] = -theta_[tuple(sign_theta)]
    return theta_

def calculate_sensor_positions(state, sensor_distance, sensor_angle_offset, angle=None):
    """
    Calculate the positions of left and right sensors based on the moth's position,
    body orientation, sensor distance from the body center, and the angle offset
    of each sensor from the body's heading angle.

    Parameters:
    x (float): X coordinate of the moth's center.
    y (float): Y coordinate of the moth's center.
    body_angle (float): Body's heading angle in radians.
    sensor_distance (float): Distance of each sensor from the center of the body.
    sensor_angle_offset (float): Angular offset of each sensor from the body's heading, in radians.

    Returns:
    tuple: A tuple containing the positions (x, y) of the left and right sensors.
    """
    # if angle is not None:
    #
    if len(state.shape) == 1:
        x, y = state[0], state[1]
    elif len(state.shape) == 2:
        x, y = state[0,0], state[0,1]
    body_angle = angle/180*np.pi
    '''else:
        x, y = state[0,0], state[0,1] # state[0], state[1]
        pos = state[0] 
        vel = state[1] # state[2:4]
        body_angle = angle_vel_pos_silkmoth(pos, vel)'''

    # Calculate the angle for the left and right sensors
    left_sensor_angle = body_angle + sensor_angle_offset
    right_sensor_angle = body_angle - sensor_angle_offset

    # Calculate the position of the left sensor
    left_sensor_x = x + sensor_distance * np.cos(left_sensor_angle)
    left_sensor_y = y + sensor_distance * np.sin(left_sensor_angle)

    # Calculate the position of the right sensor
    right_sensor_x = x + sensor_distance * np.cos(right_sensor_angle)
    right_sensor_y = y + sensor_distance * np.sin(right_sensor_angle)

    return np.array([[left_sensor_x, left_sensor_y], [right_sensor_x, right_sensor_y]])

def get_max_odor_value(sensor_pos, radius, odor_map, t):
    """ Fetch the maximum odor value within a sector defined by radius and angular range. """
    max_value = 0  # Initialize max value to be returned
    x_center, y_center = sensor_pos  # Unpack sensor position
    y_center = -y_center*1000
    x_center = x_center*1000

    # Compute bounds for the search area within the map
    x_min = max(0, x_center - radius)
    x_max = min(odor_map.shape[2], x_center + radius + 1)
    y_min = max(-200, y_center - radius) 
    y_max = min(odor_map.shape[1]-200, y_center + radius + 1)

    # Iterate over the defined box and apply the radius and angle conditions
    for x in range(int(x_min), int(x_max)):
        for y in range(int(y_min), int(y_max)):
            # Check if the point is within the radius
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                # Check if the point is within the angle range
                # angle = np.arctan2(y - y_center, x - x_center)
                #if angle_range[0] <= angle <= angle_range[1]:
                #    # Update the maximum value found
                # np.savetxt('data.csv', odor_map[-1], delimiter=',')
                max_value = max(max_value, np.max(odor_map[:,y+200, x]))
    # if t == 135:
    #   import pdb; pdb.set_trace()
    #    # y: 118-218 x: 248-315
    return max_value

def compute_odor_vis_wind_silkmoth(src_angle, next_src_angle, next_state, mothsens_prev, odor_map, tdlr_rstimLR_prev, time_step, dt, cond):

    tdl, tdr, rstimL_prev, rstimR_prev, alphal, alphar = tdlr_rstimLR_prev 
    # Remove the posture angle restriction
    offset = 0
    if (next_src_angle - src_angle) / dt >= 500: # e.g., 179 -> -179
        offset -= 2 * 180
    elif (next_src_angle - src_angle) / dt <= -500: # e.g., -179 -> 179
        offset += 2 * 180
    # src_angle = next_src_angle + offset
    
    # Calculate the angular velocity
    angV = (next_src_angle - src_angle + offset) / dt
    
    # Calculate the visual stimulus
    if cond == 0:
        if angV >= 0.1 * 180 / np.pi:
            visstim = -angV
        elif angV <= -0.1 * 180 / np.pi:
            visstim = -angV
        else:
            visstim = 0
        if visstim >= 1.2 * 180 / np.pi:
            visstim = 1.2 * 180 / np.pi
        elif visstim <= -1.2 * 180 / np.pi:
            visstim = -1.2 * 180 / np.pi
    else:
        visstim = 0

    # Calculate the wind stimulus 
    # F:[1 0 0 0]，R:[0 1 0 0]，L:[0 0 1 0]，B:[0 0 0 1]
    windstim = np.zeros(4)
    if cond == 0:
        if next_src_angle <= 45 or next_src_angle <= -135:
            windstim[0] = 1 # forward
        elif -45 < next_src_angle <= -135:
            windstim[1] = 1 # right
        elif 45 <= next_src_angle <= 135:
            windstim[2] = 1 # left
        else:
            windstim[3] = 1 # backward
    
    # Calculate the odor stimulus
    # discretize sensor value (2: both, 0: Left, 1: Right, 3: None)
    
    sensor_distance = 13/1000  # Distance from the center to each sensor [mm]
    sensor_angle_offset = 19/180*np.pi  # Angle offset for sensors from the heading angle (19 degrees)

    sensor_pos = calculate_sensor_positions(next_state, sensor_distance, sensor_angle_offset, angle=next_src_angle)

    radius = 3 # Radius in mm 
    pxRate = 1.19 # Pixel rate (mm/px)
    
    sensor_values = []
    for i, sensor_pos_ in enumerate(sensor_pos):
        # pos = convert_coordinate(sensor_pos_)
        # angle_range = (angles[i][0] + np.pi, angles[i][1] + np.pi)
        max_value = get_max_odor_value(sensor_pos_*pxRate, radius, odor_map, time_step)
        sensor_values.append(max_value+0.017) # 93: max odor value, 3.328: max sensor value, 0.017: min sensor value

    left = sensor_values[0] > 1 # 変える必要あり
    right = sensor_values[1] > 1 # 変える必要あり

    # Determine the sensor reaction based on the values of left and right
    # 2: both, 0: Left, 1: Right, 3: None
    if left and right:
        reacted_sensor = 2
    elif left:
        reacted_sensor = 0
    elif right:
        reacted_sensor = 1
    else:
        reacted_sensor = 3

    # make the odor map grid world
    bex = np.round(next_state[0]*1000).astype(int)
    bey = np.round(next_state[1]*1000 + 200).astype(int)
    
    # detection of odor decrease
    rstimL = np.zeros((3,))
    rstimR = np.zeros((3,))

    rstimL[0] = 3 - reacted_sensor # 1: both, 3: Left, 2: Right, 0: None 
    rstimR[0] = 3 - reacted_sensor
    if rstimL[0] == 3 or rstimL[0] == 1:
        rstimL[1] = 1
    if rstimR[0] == 2 or rstimR[0] == 1:
        rstimR[1] = 1
    rstimL[2] = rstimL[1] - rstimL_prev[1]
    rstimR[2] = rstimR[1] - rstimR_prev[1]
    
    mothsens = np.zeros((2,))
    # mothsens_prev = state[5:6]
    # tdl = 0
    # tdr = 0
    # alpha = 0
    beta = 0.07
    # print(rstimL)

    if rstimL[1] == 1: # for the left sensor
        if bey < 395 and bex < 495 and bey >= 0 and bex >= 0:
            mothsens[0] = sensor_values[0] # odor_map[-1,bey, bex]
            alphal = sensor_values[0]
        else:
            mothsens[0] = mothsens_prev[0]
            alphal = mothsens_prev[0]
    else:
        if rstimL[2] == -1: # Fall for the left sensor
            tdl = (time_step-1) * dt
            if bey < 395 and bex < 495 and bey >= 0 and bex >= 0:
                alphal = sensor_values[0] # odor_map[-1,bey, bex]
                mothsens[0] = sensor_values[0] # odor_map[-1,bey, bex]
            else:
                mothsens[0] = mothsens_prev[0]
    mothsens[0] = alphal * np.exp(-(time_step*dt- tdl) / 1.96) + beta 
    
    if rstimR[1] == 1: # for the right sensor
        if bey < 395 and bex < 495 and bey >= 0 and bex >= 0:
            mothsens[1] = sensor_values[1] # odor_map[-1,bey, bex]
            alphar = sensor_values[1]
        else:
            mothsens[1] = mothsens_prev[1] 
            alphar = mothsens_prev[1] 
    else:
        if rstimR[2] == -1: # Fall for the right sensor
            tdr = (time_step-1) * dt
            if bey < 395 and bex < 495 and bey >= 0 and bex >= 0:
                alphar = sensor_values[1] # odor_map[-1,bey, bex]
                mothsens[1] = sensor_values[1] # odor_map[-1,bey, bex]
            else:
                mothsens[1] = mothsens_prev[1]
    mothsens[1] = alphar * np.exp(-(time_step*dt - tdr) / 1.96) + beta

    tdlr_rstimLR = [tdl, tdr, rstimL, rstimR, alphal, alphar]

    #if time_step == 135:
    #    import pdb; pdb.set_trace()
    #    [ 29.88497,  39.88032,   0.     ], # 15
    #    [ 60.99348,  45.69715,   0.     ], # 16
    #    [  4.98568,   4.98568, -53.63422], # 17
    return np.concatenate([mothsens, np.array([visstim]), np.array(windstim)]), tdlr_rstimLR