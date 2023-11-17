import numpy as np
from numpy.linalg import norm
from controllers.controller import Controller
import utils
import config
from controllers.pid import pid_controller

rad2deg = 180.0/np.pi
deg2rad = np.pi/180.0

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
Py    = 1.0
Px    = Py
Pz    = 1.0

pos_P_gain = np.array([Px, Py, Pz])

# Velocity P-D gains
Pxdot = 5.0
Dxdot = 0.5
Ixdot = 5.0

Pydot = Pxdot
Dydot = Dxdot
Iydot = Ixdot

Pzdot = 4.0
Dzdot = 0.5
Izdot = 5.0

vel_P_gain = np.array([Pxdot, Pydot, Pzdot])
vel_D_gain = np.array([Dxdot, Dydot, Dzdot])
vel_I_gain = np.array([Ixdot, Iydot, Izdot])

# Attitude P gains
Pphi = 8.0
Ptheta = Pphi
Ppsi = 1.5
PpsiStrong = 8

att_P_gain = np.array([Pphi, Ptheta, Ppsi])

# Rate P-D gains
Pp = 1.5
Dp = 0.04

Pq = Pp
Dq = Dp 

Pr = 1.0
Dr = 0.1

rate_P_gain = np.array([Pp, Pq, Pr])
rate_D_gain = np.array([Dp, Dq, Dr])

saturateVel_separetely = False

uMax = 5.0
vMax = 5.0
wMax = 5.0

velMax = np.array([uMax, vMax, wMax])
velMaxAll = 5.0

saturateVel_separetely = False

# Max tilt
tiltMax = 50.0*deg2rad

# Max Rate
pMax = 200.0*deg2rad
qMax = 200.0*deg2rad
rMax = 150.0*deg2rad

rateMax = np.array([pMax, qMax, rMax])

class PidController(Controller):

    def __init__(self, quad):#, yawType):
        super().__init__(quad)#, yawType)
        self.quad = quad
        self.sDesCalc = np.zeros(16)
        self.w_cmd = np.ones(4)*quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        #if (yawType == 0):
        #    att_P_gain[2] = 0
        self.setYawWeight()
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)

        self.z_pos_controller = pid_controller(pos_P_gain[2], 0, 0) # 1, 0, 0
        self.xy_pos_controller = pid_controller(pos_P_gain[0:2], 0, 0)
        self.z_vel_controller = pid_controller(vel_P_gain[2], 0, vel_D_gain[2])
        self.xy_vel_controller = pid_controller(vel_P_gain[0:2], 0, vel_D_gain[0:2])

    def control(self, quad, sDes, Ts):
        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.target_state = sDes #np.array([  3,           1.,          -3.,          1.,           0.,
                                 #       0.,           0.,           0.,           0.,           0.,
                                 #       0.,           0.,           0.,         522.98471407,   0.,
                                 #       522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                                 #       0.        ], dtype=np.float32)
        self.pos_sp[:]    = self.target_state[0:3]#traj.sDes[0:3]
        self.vel_sp[:]    = self.target_state[3:6]#traj.sDes[3:6]
        self.acc_sp[:]    = self.target_state[6:9]#traj.sDes[6:9]
        self.thrust_sp[:] = self.target_state[9:12]#traj.sDes[9:12]
        self.eul_sp[:]    = self.target_state[12:15]#traj.sDes[12:15]
        self.pqr_sp[:]    = self.target_state[15:18]#traj.sDes[15:18]
        self.yawFF[:]     = self.target_state[18:]#traj.sDes[18]

        self.z_pos_control(quad, Ts)
        self.xy_pos_control(quad, Ts)
        self.saturateVel()
        self.z_vel_control(quad, Ts)
        self.xy_vel_control(quad, Ts)
        self.thrustToAttitude(quad, Ts)
        self.attitude_control(quad, Ts)
        self.rate_control(quad, Ts)

        # Mixer
        # --------------------------- 
        self.w_cmd = utils.mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp


    def z_pos_control(self, quad, Ts):
        self.vel_sp[2] = self.z_pos_controller.update_control(self.pos_sp[2], quad.pos[2])
        
    
    def xy_pos_control(self, quad, Ts):
        self.vel_sp[0:2] = self.xy_pos_controller.update_control(self.pos_sp[0:2], quad.pos[0:2]) 
        
        
    def saturateVel(self):
        if (saturateVel_separetely):
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > velMaxAll):
                self.vel_sp = self.vel_sp/totalVel_sp*velMaxAll


    def z_vel_control(self, quad, Ts):
        thrust_z_sp = self.z_pos_controller.update_control(self.vel_sp[2], quad.vel[2]) + quad.params["mB"]*(self.acc_sp[2] - quad.params["g"]) + self.thr_int[2]

        vel_z_error = self.vel_sp[2] - quad.vel[2]
        
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -quad.params["minThr"]
        uMin = -quad.params["maxThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += vel_I_gain[2]*vel_z_error*Ts * quad.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    
    def xy_vel_control(self, quad, Ts):
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp = self.xy_vel_controller.update_control(self.vel_sp[0:2], quad.vel[0:2])

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(tiltMax)
        thrust_max_xy = np.sqrt(quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2]*vel_err_lim*Ts * quad.params["useIntergral"]
    
    def thrustToAttitude(self, quad, Ts):
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thrust_sp)
        if (config.orient == "ENU"):
            body_z = -body_z
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = utils.RotToQuat(R_sp)
        
        
    def attitude_control(self, quad, Ts):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = quad.dcm[:,2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)
        if (config.orient == "ENU"):
            e_z_d = -e_z_d

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + np.sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = utils.quatMultiply(qe_red, quad.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, np.array([np.cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, np.sin(self.yaw_w*np.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(quad.quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0*np.sign(self.qe[0])*self.qe[1:4])*att_P_gain
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -rateMax[2], rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_sp += utils.quat2Dcm(utils.inverse(quad.quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(self.rate_sp, -rateMax, rateMax)


    def rate_control(self, quad, Ts):
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = rate_P_gain*rate_error - rate_D_gain*quad.omega_dot     # Be sure it is right sign for the D part
        

    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain

    