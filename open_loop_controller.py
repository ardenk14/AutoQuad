import numpy as np
from controller import Controller

class OpenLoopController(Controller):

    def __init__(self, quad):#, yawType):
        super().__init__(quad)#, yawType)

    def control(self, quad, sDes, Ts):
        self.w_cmd = np.ones(4)*quad.params["w_hover"] #np.ones(4) * np.random.rand(4) * 50
        self.w_cmd[-1] += 100
        self.w_cmd[-3] += 100
        self.w_cmd[-2] -= 100
        self.w_cmd[0] -= 100
        print(self.w_cmd)