import time

class time_elapsed(object):
    """
    Debug tool that returns time elapsed since last call
    (or since init at first call).
    """
    
    def __init__(self):
        self.initial_time = time.time()
        self.time = self.initial_time
        
    def __call__(self):
        this_time = time.time()
        elapsed = this_time-self.time
        self.time = this_time
        return elapsed 

    def total_elapsed(self):
        return time.time()-self.initial_time
