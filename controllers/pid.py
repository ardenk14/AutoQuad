'''
    Very simple PID controller
    Has option to input state-rate rather than calculating this numerically
'''
#import rospy
import time

class pid_controller():
    def __init__(self, kp, ki, kd):
        ''' Can optionally require a state rate input
            This avoids latencies from numberical calculation the derivative
        '''
        self.kp = kp
        self.ki = ki
        self.kd = kd       
        self.previous_time = None
        self.previous_error = None
        self.previous_target = None
        self.I_error = 0

    def update_control(self, target, state):
        ''' Will calculate derivative numerically '''

        current_error = target - state
        current_time = time.time()#rospy.Time.now()

        #if self.previous_time:
        #    dt = ( current_time - self.previous_time ).to_sec()
        #else:
        #    dt = 0
        dt = 0.005
       
        self.I_error +=  current_error * dt

        if self.previous_error is not None and dt > 0:        
            D_error = ( current_error - self.previous_error ) / dt
        else:
            D_error = 0
        
        self.previous_time = current_time
        self.previous_error = current_error
        self.previous_target = target

        return  self.kp * current_error + self.ki * self.I_error + self.kd * D_error

    """def update_control_with_rate(self, target, state, state_rate):
        ''' Uses state rate as part of the derivative '''

        current_error = target - state
        current_time = rospy.Time.now()

        if self.previous_time:
            dt = ( current_time - self.previous_time ).to_sec()
        else:
            dt = 0
       
        self.I_error +=  current_error * dt

        # Use state_rate instead of differencing state -- can be more stable
        if self.previous_target and dt > 0:
            D_error = (target - self.previous_target) / dt - state_rate
        else:
            D_error = -state_rate
        
        self.previous_time = current_time
        self.previous_error = current_error
        self.previous_target = target

        return  self.kp * current_error + self.ki * self.I_error + self.kd * D_error"""


