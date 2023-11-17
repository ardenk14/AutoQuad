import numpy as np
from controllers.controller import Controller

class OpenLoopController(Controller):

    def __init__(self, quad):#, yawType):
        super().__init__(quad)#, yawType)
        self.t = 0

    def control(self, quad, sDes, Ts):
        self.w_cmd = np.ones(4)*self.quad.params["w_hover"] #np.ones(4) * np.random.rand(4) * 50
        self.w_cmd[-1] += 1
        """if 100 < self.t < 200 or 500 < self.t < 600:
            self.w_cmd[-1] += 100
            self.w_cmd[-3] += 100
            self.w_cmd[-2] += 100
            self.w_cmd[0] += 100
        if 400 > self.t > 300:
            self.w_cmd[-1] -= 100
            self.w_cmd[-3] -= 100
            self.w_cmd[-2] -= 100
            self.w_cmd[0] -= 100"""
        print(self.w_cmd)
        self.t+= 1