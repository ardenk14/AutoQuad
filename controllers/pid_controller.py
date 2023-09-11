import numpy as np
from controllers.controller import Controller

class PidController(Controller):

    def __init__(self, quad):#, yawType):
        super().__init__(quad)#, yawType)

    def control(self, quad, sDes, Ts):
        pass