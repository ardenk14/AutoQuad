import numpy as np

class Controller:
    
    def __init__(self, quad):#, yawType):
        self.rad2deg = 180.0/np.pi
        self.deg2rad = np.pi/180.0

        # Max Velocities
        self.uMax = 5.0
        self.vMax = 5.0
        self.wMax = 5.0

        self.velMax = np.array([self.uMax, self.vMax, self.wMax])
        self.velMaxAll = 5.0

        self.saturateVel_separetely = False

        # Max tilt
        self.tiltMax = 50.0*self.deg2rad

        # Max Rate
        self.pMax = 200.0*self.deg2rad
        self.qMax = 200.0*self.deg2rad
        self.rMax = 150.0*self.deg2rad

        self.rateMax = np.array([self.pMax, self.qMax, self.rMax])

        self.sDesCalc = np.zeros(16)
        self.w_cmd = np.ones(4)*quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        #if (yawType == 0):
        #    att_P_gain[2] = 0
        #self.setYawWeight()
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)

    def control(self, quad, sDes, Ts):
        pass