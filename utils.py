import torch
import contextlib
import warp as wp
class TorchProfilingManager:
    def __init__(self, profile_on=False):
        self.profile_on = profile_on
        self.prof = None

    @contextlib.contextmanager
    def torch_profile_function(self, name):
        if self.profile_on:
            with torch.profiler.record_function(name):
                yield
        else:
            yield
        
    @contextlib.contextmanager
    def torch_profile_section(self): 
        if self.profile_on:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                self.prof = prof
                yield
        else:
            yield

    def get_profiler(self):
        return self.prof
    
    
class WarpProfilingManager:
    def __init__(self, profile_on=False):
        self.profile_on = profile_on

    @contextlib.contextmanager
    def wp_profile_function(self, name):
        if self.profile_on:
            with wp.ScopedTimer(name, synchronize=True, cuda_filter=wp.TIMING_ALL):
                yield
        else:
            yield