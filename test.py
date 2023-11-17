import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile

from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config
from controllers.open_loop_controller import OpenLoopController
from controllers.mppi_controller import MppiController
from controllers.pid_controller import PidController
from controllers.optimal_controller import OptimalController

def sim_step(t, Ts, quad, ctrl, wind, traj=None):
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    sDes = traj#traj.desiredState(t, Ts, quad)        

    # Set control command for next timestep
    ctrl.control(quad, sDes, Ts) # traj
    return t

def main():

    traj = np.array([[  -1,           0.,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -2,           1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -2,           1.,          -4.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  0,           0.,          -4.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  0,           0.,          -3.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  0,           0.,          -5.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  0,           0.,          -4.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  1,           -2,          -4.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.]], dtype=np.float32)
    
    traj = np.array([[  -5,           -5.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  5,           -5.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  5,           5.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -5,           5.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -5,           -5.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -5,           -5.,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  5,           -5.,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  5,           5,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -5,           5.,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -5,           -5.,          -8.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.]], dtype=np.float32)
    
    traj = np.array([[  -1,           -1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  1,           -1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  1,           1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -1,           1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -1,           -1.,          0.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -1,           -1.,          -1.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  1,           -1.,          -1.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  1,           1,          -1.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -1,           1.,          -1.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.],
                       [  -1,           -1.,          -1.,          1.,           0.,
                       0.,           0.,           0.,           0.,           0.,
                       0.,           0.,           0.,         522.98471407,   0.,
                       522.98471407,   0.,         522.98471407,   0.,         522.98471407, 0.]], dtype=np.float32)


    # ACTIONS: 0: front left, 1: front right, 2: back right, 3: back left
    # Set simulation variables
    Ti = 0    # Time initial
    Ts = 0.005 # Time steps
    Tf = 25#50   # Time final
    
    quad = Quadcopter(Ti)
    #quad.get_linearized()
    #return
    #quad.get_linearized()
    ctrl = OptimalController(quad)
    #print("A: ", other.A)
    #print("B: ", other.B)
    #ctrl = PidController(quad)#MppiController(quad)#OpenLoopController(quad)#
    #ctrl = OpenLoopController(quad)
    wind = Wind('NONE', 2.0, 90, -15) #'NONE' 'FIXED'

    # Initialize values to track for visualization
    numTimeStep = int(Tf/Ts+1)

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    #sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all        = np.zeros([numTimeStep, len(quad.thr)])
    tor_all        = np.zeros([numTimeStep, len(quad.tor)])

    t_all[0]            = Ti
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    #sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:]     = quad.wMotor
    thr_all[0,:]        = quad.thr
    tor_all[0,:]        = quad.tor

    t = Ti
    i = 1
    cnt = 0
    while round(t, 3) < Tf:
        #quad.update(t, Ts, cmd, wind)
        t = sim_step(t, Ts, quad, ctrl, wind, traj[cnt])
        if np.linalg.norm(traj[cnt][:3] - quad.state[:3] ) < 0.1 and cnt != len(traj)-1:
            cnt += 1

        t_all[i]             = t
        s_all[i,:]           = quad.state
        pos_all[i,:]         = quad.pos
        vel_all[i,:]         = quad.vel
        quat_all[i,:]        = quad.quat
        omega_all[i,:]       = quad.omega
        euler_all[i,:]       = quad.euler
        #sDes_traj_all[i,:]   = traj.sDes
        sDes_calc_all[i,:]   = ctrl.sDesCalc
        w_cmd_all[i,:]       = ctrl.w_cmd
        wMotor_all[i,:]      = quad.wMotor
        thr_all[i,:]         = quad.thr
        tor_all[i,:]         = quad.tor

        #print("QUAD POSE: ", quad.pos)
        print("QUAD STATE: ", quad.state)

        i += 1

    #utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
    #ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    #plt.show()

    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_calc_all)
    ani = utils.sameAxisAnimation(t_all, pos_all, quat_all, Ts, quad.params)
    plt.show()


if __name__ == '__main__':
    main()
