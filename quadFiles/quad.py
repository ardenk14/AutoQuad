# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode
from scipy import signal

from quadFiles.initQuad import sys_params, init_cmd, init_state
import utils
import config
from utils.windModel import Wind
import random
import torch

deg2rad = pi/180.0

class Quadcopter:

    def __init__(self, Ti):
        
        # Quad Params
        # ---------------------------
        self.params = sys_params()
        
        # Command for initial stable hover
        # ---------------------------
        ini_hover = init_cmd(self.params)
        self.params["FF"] = ini_hover[0]       # Feed-Forward Command for Hover
        self.params["w_hover"] = ini_hover[1]    # Motor Speed for Hover
        self.params["thr_hover"] = ini_hover[2]  # Motor Thrust for Hover  
        self.thr = np.ones(4)*ini_hover[2]
        self.tor = np.ones(4)*ini_hover[3]

        # Initial State
        # ---------------------------
        self.state = init_state(self.params)
        print("INITIAL STATE: ", self.state)
        print("-------------------------------------------------")

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)

        with open('Linearization.npy', 'rb') as f:
            self.A = np.load(f)
            self.B = np.load(f)
            #self.A, self.B, C, D, dt = signal.cont2discrete((self.A, self.B, 0, 0), 0.005, method='zoh')
            #print("A: ", self.A)
            #print("B: ", self.B)


    def extended_state(self):

        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.dcm = utils.quat2Dcm(self.quat)

        # Euler angles of current state
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    
    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def forward_model(self, state, cmd):
        #ts = 0.0005
        ts = 0.005
        t = 0
        wind = Wind('NONE', 2.0, 90, -15)
        k1 = self.state_dot_batch(0, state, cmd, wind)
        k2_state = state + k1*ts/2
        k2 = self.state_dot_batch(0, k2_state, cmd, wind)
        k3_state = state + k2*ts/2
        k3 = self.state_dot_batch(0, k3_state, cmd, wind)
        k4_state = state + k3*ts
        k4 = self.state_dot_batch(0, k4_state, cmd, wind)
        state = state + ts/6 * (k1 + 2*k2 + 2*k3 + k4)

        altitude = state[:, 2]
        """for i in altitude:
            if i < 0:
                print('state: ', i)"""
        #print("STATE: ", where)
        #while t <= 0.005:
        #    state += self.state_dot_batch(0, state, cmd, wind)
        #    t += ts
        return state #state + (self.state_dot_batch(0, state, cmd, Wind('NONE', 2.0, 90, -15)) * 0.005)

    # TODO: Make a function for this that can run in batch
    def state_dot_batch(self, t, state, cmd, wind):
        state = torch.from_numpy(state)
        cmd = torch.from_numpy(cmd)

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[:, 0]#.reshape(state.shape[0], 1)
        y      = state[:, 1]#.reshape(state.shape[0], 1)
        z      = state[:, 2]#.reshape(state.shape[0], 1)
        q0     = state[:, 3]#.reshape(state.shape[0], 1)
        q1     = state[:, 4]#.reshape(state.shape[0], 1)
        q2     = state[:, 5]#.reshape(state.shape[0], 1)
        q3     = state[:, 6]#.reshape(state.shape[0], 1)
        xdot   = state[:, 7]#.reshape(state.shape[0], 1)
        ydot   = state[:, 8]#.reshape(state.shape[0], 1)
        zdot   = state[:, 9]#.reshape(state.shape[0], 1)
        p      = state[:, 10]#.reshape(state.shape[0], 1)
        q      = state[:, 11]#.reshape(state.shape[0], 1)
        r      = state[:, 12]#.reshape(state.shape[0], 1)
        wM1    = state[:, 13]#.reshape(state.shape[0], 1)
        wdotM1 = state[:, 14]#.reshape(state.shape[0], 1)
        wM2    = state[:, 15]#.reshape(state.shape[0], 1)
        wdotM2 = state[:, 16]#.reshape(state.shape[0], 1)
        wM3    = state[:, 17]#.reshape(state.shape[0], 1)
        wdotM3 = state[:, 18]#.reshape(state.shape[0], 1)
        wM4    = state[:, 19]#.reshape(state.shape[0], 1)
        wdotM4 = state[:, 20]#.reshape(state.shape[0], 1)

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        
        # cmd is (B, cmd_size)
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[:, 0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[:, 1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[:, 2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[:, 3])/(tau**2)
    
        wMotor = torch.vstack([wM1, wM2, wM3, wM4]).t()#np.array([wM1, wM2, wM3, wM4]) (B, 4)
        wMotor = torch.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[:, 0]#.reshape(wMotor.shape[0], 1)
        ThrM2 = thrust[:, 1]#.reshape(wMotor.shape[0], 1)
        ThrM3 = thrust[:, 2]#.reshape(wMotor.shape[0], 1)
        ThrM4 = thrust[:, 3]#.reshape(wMotor.shape[0], 1)
        TorM1 = torque[:, 0]#.reshape(wMotor.shape[0], 1)
        TorM2 = torque[:, 1]#.reshape(wMotor.shape[0], 1)
        TorM3 = torque[:, 2]#.reshape(wMotor.shape[0], 1)
        TorM4 = torque[:, 3]#.reshape(wMotor.shape[0], 1)

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)

        xdot = xdot.reshape(state.shape[0])
        ydot = ydot.reshape(state.shape[0])
        zdot = zdot.reshape(state.shape[0])
        wM1    = state[:, 13].reshape(state.shape[0])
        wdotM1 = state[:, 14].reshape(state.shape[0])
        wM2    = state[:, 15].reshape(state.shape[0])
        wdotM2 = state[:, 16].reshape(state.shape[0])
        wM3    = state[:, 17].reshape(state.shape[0])
        wdotM3 = state[:, 18].reshape(state.shape[0])
        wM4    = state[:, 19].reshape(state.shape[0])
        wdotM4 = state[:, 20].reshape(state.shape[0])

        sdot     = torch.zeros((state.shape[0], 21))
        sdot[:, 0]  = xdot
        sdot[:, 1]  = ydot
        sdot[:, 2]  = zdot
        sdot[:, 3]  = -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r
        sdot[:, 4]  = 0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r
        sdot[:, 5]  = 0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r
        sdot[:, 6]  = -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r
        sdot[:, 7]  = (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB
        sdot[:, 8]  = (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB
        sdot[:, 9]  = (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB
        sdot[:, 10] = ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx
        sdot[:, 11] = ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy
        sdot[:, 12] = ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz
        sdot[:, 13] = wdotM1
        sdot[:, 14] = wddotM1
        sdot[:, 15] = wdotM2
        sdot[:, 16] = wddotM2
        sdot[:, 17] = wdotM3
        sdot[:, 18] = wddotM3
        sdot[:, 19] = wdotM4
        sdot[:, 20] = wddotM4

        #self.acc = sdot[7:10]
    
        """# State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (config.orient == "NED"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (config.orient == "ENU"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([21])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]"""
        sdot = sdot.numpy()

        return sdot

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[0]
        y      = state[1]
        z      = state[2]
        q0     = state[3]
        q1     = state[4]
        q2     = state[5]
        q3     = state[6]
        xdot   = state[7]
        ydot   = state[8]
        zdot   = state[9]
        p      = state[10]
        q      = state[11]
        r      = state[12]
        wM1    = state[13]
        wdotM1 = state[14]
        wM2    = state[15]
        wdotM2 = state[16]
        wM3    = state[17]
        wdotM3 = state[18]
        wM4    = state[19]
        wdotM4 = state[20]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        
        uMotor = cmd
        #print("UMOTOR: ", uMotor)
        #print("WDOTM1: ", wdotM1)
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)
    
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = [0, 0, 0]#wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (config.orient == "NED"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (config.orient == "ENU"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([21])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4
        #print("WDDOTM1: ", sdot[14])
        #print("wM1", sdot[13])

        self.acc = sdot[7:10]

        return sdot

    def update(self, t, Ts, cmd, wind):

        prev_vel   = self.vel
        prev_omega = self.omega

        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)
        #self.state = self.A @ self.state + self.B @ cmd
        #self.state = self.state + self.state_dot_batch(t, np.array([self.state]), np.array([cmd]), wind)[0] * Ts
        #self.state = self.forward_model(np.array([self.state]), np.array([cmd]))[0]

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()

    def get_linearized(self):
        NUM_TRIALS = 1000
        NUM_STATES = 20
        NUM_ACTIONS = 1000
        start = np.zeros((NUM_TRIALS*NUM_STATES*NUM_ACTIONS, 25))
        end = np.zeros((NUM_TRIALS*NUM_STATES*NUM_ACTIONS, 25))
        for b in range(NUM_TRIALS):
            print("TRIAL: ", b)
            start_state = np.array([  0.,           0.,          0.,          1.,           0.,
                                            0.,           0.,           0.,           0.,           0.,
                                            0.,           0.,           0.,         522.98471407,   0.,
                                            522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                                            0.        ], dtype=np.float32)
            
            starting_states = np.zeros((NUM_STATES*NUM_ACTIONS, 25))
            next_states = np.zeros((NUM_STATES*NUM_ACTIONS, 25))
            for i in range(NUM_STATES):
                #for j in range(NUM_ACTIONS):
                #print("START STATE: ", start_state.shape)
                #states[NUM_ACTIONS*i:NUM_ACTIONS*(i+1), :21] = start_state # Set start state as state for each
                states = np.array([[start_state] for k in range(NUM_ACTIONS)]).reshape((NUM_ACTIONS, -1))
                #print("STATES: ", states.shape)
                cmds = np.array([[(random.random()-0.5)*0.5 + self.params["w_hover"] for w in range(4)] for u in range(NUM_ACTIONS)])
                #print("CMDS: ", cmds.shape)
                
                new_states = self.forward_model(states, cmds)
                #print("NEW STATES: ", new_states.shape)

                starting_states[NUM_ACTIONS*i:NUM_ACTIONS*(i+1), :21] = states
                starting_states[NUM_ACTIONS*i:NUM_ACTIONS*(i+1), 21:] = cmds
                next_states[NUM_ACTIONS*i:NUM_ACTIONS*(i+1), :21] = new_states
                #print("RANDOM: ", random.randint(NUM_ACTIONS*i, NUM_ACTIONS*(i+1)))
                start_state = next_states[random.randint(NUM_ACTIONS*i, NUM_ACTIONS*(i+1)-1), :21]
            start[NUM_STATES*NUM_ACTIONS*b:NUM_STATES*NUM_ACTIONS*(b+1)] = starting_states
            end[NUM_STATES*NUM_ACTIONS*b:NUM_STATES*NUM_ACTIONS*(b+1)] = next_states

        A, r, rank, s = np.linalg.lstsq(start, end)#(starting_states, next_states)
        #A = np.linalg.inv(starting_states.T @ starting_states) @ starting_states.T @ next_states
        print("A: ", A)
        print("A: ", A.shape)
        final_A = A.T[:21, :21]
        final_B = A.T[:21, 21:]
        print("FINAL B: ", final_B)
        print("SUM OF SQUARED RESIDUALS: ", r)

        with open('Linearization.npy', 'wb') as f:
            np.save(f, final_A)
            np.save(f, final_B)

        """import sympy as sp

        # Define symbolic variables for the state and control inputs
        x = sp.symbols('x:21')  # Represents the state vector
        u = sp.symbols('u:4')   # Represents the control input

        wind = Wind('NONE', 2.0, 90, -15)

        # Compute the time derivative of each element of the state vector x
        # Replace the symbolic control inputs with specific numeric values for linearization
        u_hover = [522.98471407, 522.98471407, 522.98471407, 522.98471407]  # Hover command
        xdot_numeric = [self.state_dot(0, x, u_hover, 0)[i].subs(list(zip(u, u_hover))) for i in range(21)]

        # Extract the elements of xdot as separate equations
        dxdt = [sp.Eq(x[i], xdot_numeric[i]) for i in range(21)]

        # Calculate the Jacobian matrices A and B
        A = sp.Matrix([[sp.diff(eq.lhs, xj).subs(list(zip(u, u_hover))) for xj in x] for eq in dxdt])
        B = sp.Matrix([[sp.diff(eq.lhs, ui).subs(list(zip(u, u_hover))) for ui in u] for eq in dxdt])

        # Convert the symbolic matrices A and B to NumPy arrays for further processing
        A_numeric = sp.lambdify((x,), A, modules='numpy')(u_hover)
        B_numeric = sp.lambdify((x,), B, modules='numpy')(u_hover)

        # Display the resulting A and B matrices
        print("A matrix:")
        print(A_numeric)
        print("B matrix:")
        print(B_numeric)"""

