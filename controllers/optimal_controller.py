import numpy as np
from numpy.linalg import norm, cond, matrix_power
import cvxpy as cp
from controllers.controller import Controller
import utils
import config
from scipy import signal

class OptimalController(Controller):

    def __init__(self, quad):#, yawType):
        super().__init__(quad)

        self.w_cmd = np.ones(4)*quad.params["w_hover"]

        mB   = quad.params["mB"]
        g    = quad.params["g"]
        dxm  = quad.params["dxm"]
        dym  = quad.params["dym"]
        IB   = quad.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = quad.params["Cd"]
        
        kTh  = quad.params["kTh"]
        kTo  = quad.params["kTo"]
        tau  = quad.params["tau"]
        kp   = quad.params["kp"]
        damp = quad.params["damp"]
        self.minWmotor = quad.params["minWmotor"]
        self.maxWmotor = quad.params["maxWmotor"]

        #IRzz = self.params["IRzz"]
        self.cmds = None
        self.cnt = 0


        # WORKS!
        A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, 0, 0, 0, -1/2, -1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, -1/2, -1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, 0, 0, 1/2, 0, 1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, -1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, 0, 0, 1/2, -1/2, 0, 1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, 0, 0, 1/2, 1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, 0, -1/2, 1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, -2, -2, -2, -2, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2, 2, -2, 2, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 2*g, -2*g, -2*g, 2*g, kTh/mB, kTh/mB, kTh/mB, kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, (2*dym*kTh)/IBxx, 0, 0, 0, 0, (IByy-IBzz)/IBxx, (IByy-IBzz)/IBxx, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dxm*kTh)/IByy, (2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, 0, 0, 0, (IBzz - IBxx)/IByy, 0, (IBzz - IBxx)/IByy, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -2*kTo/IBzz, 2*kTo/IBzz, -2*kTo/IBzz, 2*kTo/IBzz, 0, 0, 0, (IBxx-IByy)/IBzz, (IBxx-IByy)/IBzz, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1]])
        
        A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1/2, -1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, -1/2, -1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, 1, 1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, -1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, -1/2, 1, 1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, 1/2, -1/2, 1, 0, 0, 0, 0, 0, 0, 0, -1/2, 1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, -2*g/mB, -2*g/mB, -2*g/mB, -2*g/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2*g/mB, 2*g/mB, -2*g/mB, 2*g/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 2*g/mB, -2*g/mB, -2*g/mB, 2*g/mB, kTh/mB, kTh/mB, kTh/mB, kTh/mB, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, (2*dym*kTh)/IBxx, 0, 0, 0, 1, (IByy-IBzz)/IBxx, (IByy-IBzz)/IBxx, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dxm*kTh)/IByy, (2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, 0, 0, 0, (IBzz - IBxx)/IByy, 1, (IBzz - IBxx)/IByy, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -2*kTo/IBzz, 2*kTo/IBzz, -2*kTo/IBzz, 2*kTo/IBzz, 0, 0, 0, (IBxx-IByy)/IBzz, (IBxx-IByy)/IBzz, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1]])

        A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 2*g/mB, 2*g/mB, 2*g/mB, 2*g/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2*g/mB, -2*g/mB, 2*g/mB, 2*g/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, (2*g+1)/mB, (-2*g+1)/mB, (-2*g+1)/mB, (2*g+1)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2)]]) * 0.005

        A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 2*g/mB, 2*g/mB, 2*g/mB, 2*g/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2*g/mB, -2*g/mB, 2*g/mB, 2*g/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, (2*g+1)/mB, (-2*g)/mB, (-2*g)/mB, (2*g)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, (2*kTh+1)/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2), 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2*damp*tau/tau**2)]]) * 0.005

        """A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, -1/2, -1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, -1/2, -1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, 1, 1/2, -1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, -1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, -1/2, 1, 1/2, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, -1/2, 0, 0, 0, 0],
                          [0, 0, 0, 1/2, 1/2, -1/2, 1, 0, 0, 0, 0, 0, 0, 0, -1/2, 1/2, 1/2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 2*g/mB, 2*g/mB, 2*g/mB, 2*g/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 4*kTh/mB, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 2*g/mB, -2*g/mB, 2*g/mB, -2*g/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, -4*kTh/mB, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2*g/mB, 2*g/mB, 2*g/mB, -2*g/mB, -2*kTh/mB, -2*kTh/mB, -2*kTh/mB, -2*kTh/mB, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, -(2*dym*kTh)/IBxx, (2*dym*kTh)/IBxx, 0, 0, 0, 1, (IByy-IBzz)/IBxx, (IByy-IBzz)/IBxx, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (2*dxm*kTh)/IByy, (2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, -(2*dxm*kTh)/IByy, 0, 0, 0, (IBzz - IBxx)/IByy, 1, (IBzz - IBxx)/IByy, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -2*kTo/IBzz, 2*kTo/IBzz, -2*kTo/IBzz, 2*kTo/IBzz, 0, 0, 0, (IBxx-IByy)/IBzz, (IBxx-IByy)/IBzz, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/tau**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*damp/tau+1]])"""
        
        """self.A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 2*g, -2*g, -2*g, 2*g, kTh/mB, kTh/mB, kTh/mB, kTh/mB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])"""
        
        B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [kp/tau**2, 0, 0, 0],
                           [0, kp/tau**2, 0, 0],
                           [0, 0, kp/tau**2, 0],
                           [0, 0, 0, kp/tau**2]]) * 0.005
        
        #self.A, self.B, C, D, dt = signal.cont2discrete((A, B, 0, 0), 0.005, method='zoh')
        self.A = A + np.eye(21)
        self.B = B
        with open('Linearization.npy', 'rb') as f:
            self.A = np.load(f)
            self.B = np.load(f)
            print("A: ", self.A)
            print("B: ", self.B)

    def control(self, quad, sDes, Ts, video_target=False):
        number_steps = 40
        #if self.cnt == 0:
        x0 = quad.state.reshape((21,1))#quad.state[np.array([0, 1, 2, 3, 4, 5, 6, 13, 15, 17, 19, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20])].reshape((21, 1))
        
        L, O, Q, r = self.get_lifted_system(number_steps, x0)
        #print("Q: ", Q)
        #print("r: ", r)

        # Construct the objective.
        target = np.zeros(21*number_steps)
        target_state = sDes #np.array([  1.,           1.,          0.,          1.,           0.,
                            #            0.,           0.,           0.,           0.,           0.,
                            #            0.,           0.,           0.,         522.98471407,   0.,
                            #            522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                            #            0.        ], dtype=np.float32)
        im = np.zeros(21*number_steps)
        importance = np.array([  4., 4., 3., 0.6, 1.5, 1.5, 0.6, 0.2, 0.2, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
        #target_state = np.array([  0.,           0.,          0.,          1.,           0.,
        #                                0.,           0.,           0.,           0.,           0.,
        #                                0.,           0.,           0.,         522.98471407,   0.,
        #                                522.98471407,   0.,         522.98471407,   0.,         522.98471407,
        #                                0.        ], dtype=np.float32)
        for i in range(number_steps):
            target[21*i:21*(i+1)] = target_state
            im[21*i:21*(i+1)] = importance * np.exp(0.1*i)
        target = target.reshape((-1, 1))
        im = im.reshape((-1, 1))
        #im[-16, 0] = 100
        #im[-17, 0] = 100
        tu = np.ones((4*number_steps, 1)) * quad.params["w_hover"]
        #print("TU: ", tu.T @ Q @ tu + r @ tu)

        u = cp.Variable((4*number_steps, 1))
        #print("Target: ", target)
        # TODO: Fix objective because some parts shouldn't be forced to zero
        #objective = cp.Minimize(cp.norm(cp.multiply(im,((L @ u + O) - target))))#cp.sum_squares(L@u+O))#L @ u + O)) # TODO: Fix objective function
        #objective = cp.Minimize(cp.quad_form(u, Q) + r @ u)

        Target_Q = np.eye(21*number_steps) * im
        objective = cp.Minimize(cp.quad_form((L @ u + O) - target, Target_Q))

        # Define the constraints
        #angle = np.array([0, 0, 0, 0, -1/4, -1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #angle2 = np.array([0, 0, 0, 0, 1/4, -1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #angle3 = np.array([0, 0, 0, 0, -1/4, 1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #angle4 = np.array([0, 0, 0, 0, 1/4, 1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        constraints = [u >= quad.params["w_hover"]-40, u <= quad.params["w_hover"]+40]
        #constraints.extend([(L@u + O)[4::21] <= 0.25])
        #constraints.extend([(L@u + O)[4::21] >= -0.25])
        #constraints.extend([(L@u + O)[5::21] <= 0.25])
        #constraints.extend([(L@u + O)[5::21] >= -0.25])
        #for i in range(number_steps):
            #constraints.extend([1 - 2*(cp.power((L@u+O)[21*i:21*(i+1)][4], 2) + cp.power((L@u+O)[21*i:21*(i+1)][4], 2)) >= 0.8])
            #constraints.extend([angle@(L@u+O)[21*i:21*(i+1)]+1 >=  0.8])
            #constraints.extend([angle2@(L@u+O)[21*i:21*(i+1)]+1 >=  0.8])
            #constraints.extend([angle3@(L@u+O)[21*i:21*(i+1)]+1 >=  0.8])
            #constraints.extend([angle4@(L@u+O)[21*i:21*(i+1)]+1 >=  0.8])
        #print("CONSTRAINTS: ", constraints)
        if video_target:
            frame_const = self.get_target_frame_constraints()
            constraints.extend(frame_const)
        #dynamics_const = self.get_dynamics_constraints()
        #obstacle_const = self.get_obstacle_constraints()
        #motion_const = self.get_motion_constraints()
        #constraints.extend([dynamics_const])#, obstacle_const, motion_const])

        
        #constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver=cp.OSQP)#verbose=True)
        if self.cnt == 0:
            self.cmds = u
        #print("RESULT: ", (L@u.value+O).T @ (L@u.value+O) < (L@tu+O).T @ (L@tu+O))
        #print("Manual: ", (L@tu+O))
        print("PREDICTED: ", (L@u.value+O)[-21:])
        # The optimal value for x is stored in `x.value`.
        #print("U VAL: ", u.value)
        #print("TU: ", tu)
        #print("U: ", u.value)
        self.w_cmd = np.array([u.value[0, 0], u.value[1, 0], u.value[2, 0], u.value[3, 0]])
        #print("MCD: ", self.cmds.value)
        #if self.cnt == 400:
        #    self.w_cmd = np.ones(4) * quad.params["w_hover"]
        #    print("END_____________________________________________________________________________________________")
        #else:
        #    self.w_cmd = np.array([self.cmds.value[4*self.cnt + 0, 0], self.cmds.value[4*self.cnt + 1, 0], self.cmds.value[4*self.cnt + 2, 0], self.cmds.value[4*self.cnt + 3, 0]])
        #    self.cnt += 1
        #print("DFGHJ: ", u.value.T@Q@u.value + r@u.value)
        #print("TU VALS: ", tu[:4])
        #print("OPT VALS: ", u.value[:4])
        #print("TU state: ", self.A @ x0 + self.B @ tu[:4])
        #print("OPT state: ", self.A @ x0 + self.B @ u.value[:4])
        #print("TU state: ", (L@tu + O)[np.array([2, 23, 44, 65, 86, 107])])
        #print("OPT state: ", (L@u.value + O)[-21:-18])
        # The optimal Lagrange multiplier for a constraint is stored in
        # `constraint.dual_value`.
        #print(constraints[0].dual_value)

    def get_lifted_system(self, num_steps, x0):
        target_state = np.array([  3,           1.,          -3.,          1.,           0.,
                                        0.,           0.,           0.,           0.,           0.,
                                        0.,           0.,           0.,         522.98471407,   0.,
                                        522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                                        0.        ], dtype=np.float32)
        L = np.zeros((21*num_steps, 4*num_steps))
        O = np.zeros((21*num_steps, 1))
        for i in range(num_steps):
            for j in range(i+1):
                L[21*i:21*(i+1), 4*j:4*(j+1)] = matrix_power(self.A, (i-j)) @ self.B
            O[21*i:21*(i+1), :] = matrix_power(self.A, (i+1)) @ x0
        R = 0
        Q = np.eye(21)
        Q[3:, 3:] *= 0.0
        Q_bar = R*np.eye(4*num_steps)
        r = np.zeros((1,4*num_steps))
        l = L[21*(i-1):21*i,:] #- target_state
        for i in range(1, num_steps+1):
            Q_bar = Q_bar + l.T@Q@l
            #print("FGHJ: ", (2*x0.T@(matrix_power(self.A, i)).T@Q@L[21*(i-1):21*i,:]).shape)
            r = r + 2*x0.T@(matrix_power(self.A, i)).T@Q@l
        return L, O, Q_bar, r

    def get_target_frame_constraints(self):
        # Ensure the entire target lays within the camera's view
        # Compute the four planes creating the camera frame "cone"
        # Get relative position of target with respect to the camera
        # Compute offsets s for each corner of the box that contains the target
        # Compute the dot products and set the inequality as a constraint
        pass

    def get_dynamics_constraints(self):
        # Use the lifted system to ensure the solution follows the linear dynamics of the drone
        pass

    def get_obstacle_constraints(self):
        # Ensure the drone does not run into any obstacles
        # Get the depth image from the drone
        # Optional for compute: Subsample the image by taking the nearest location in each box
        # Define the distance you must keep away from each point
        # Compute L1 distance for inequality constraint
        pass

    def get_motion_constraints(self):
        # Ensure drone only moves in directions that it can see (within the camera frame)
        # Compute the four planes creating the camera frame "cone"
        # Use the lifted system dynamics to get the position based on the decision variable
        # Dot product each potential point with each boudning plane to get the inequality constraints
        pass