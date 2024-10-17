class LinearScheduler:
    def __init__(self, beginning_t, beginning_value, end_value,ending_t = None, decay_steps=None):
        if (ending_t is None) == (decay_steps is None):
            raise ValueError("either of ending_t or decay_steps must be specified")
        else:
            self.decay_steps = decay_steps
            self.beginning_t = beginning_t
            self._beginning_value = beginning_value
            self._end_value = end_value
            if decay_steps is None and ending_t is not None:
                self.decay_steps = ending_t - beginning_t
    
    def __call__(self,time_step):
        fraction = min(max(time_step - self.beginning_t, 0),self.decay_steps)/self.decay_steps
        return (1-fraction) *self._beginning_value + fraction * self._end_value