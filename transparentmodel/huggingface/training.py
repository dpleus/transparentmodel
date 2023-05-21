from transformers import TrainerCallback
from transparentmodel.analysis import available_ram, model_size, optimizer_state_size, gradients_size, activations_size
from transparentmodel.pytorch_hooks import backward_hook_gradients, forward_hook_activations
import torch
from transparentmodel.analysis import capture_memory_usage, memory_in_use
import threading
import json


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Model size in MB {model_size(kwargs['model'])}")
        print(available_ram())

        kwargs["model"].generation_config.use_cache = False
        kwargs["model"].register_forward_hook(forward_hook_activations)
        kwargs["model"].register_backward_hook(backward_hook_gradients)

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
        print(available_ram())
        # print(model_size(kwargs["model"]))
        # print(gradients_size(kwargs["model"]))
        print(f"Optimizer Size in MB {optimizer_state_size(kwargs['optimizer'])}")


def apply_torchprofiler_and_callback(func):
    def wrapper(trainer, *args, **kwargs):
        model = trainer.model

        print("---------- System memory ----------")
        print(f"Memory available: {json.dumps(available_ram(), indent=4)}")
        print(f"Memory used by process: {memory_in_use()} ")

        print("")
        print("---------- Model ----------")
        print(f"Model Size: {model_size(model)} GB")
        print(f"Model dtype: {model.dtype}")

        realtime = False
        if realtime:
            stop_event = threading.Event()

            memory_thread = threading.Thread(target=capture_memory_usage, args=(stop_event,))
            memory_thread.start()

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2,
                                                                     repeat=2),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
                                    profile_memory=True,
                                    with_stack=True,
                                    record_shapes=True) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            func(trainer, *args, **kwargs)
            if realtime:
                stop_event.set()

        if realtime:
            memory_thread.join()

    return wrapper


@apply_torchprofiler_and_callback
def train_with_memory_tracking(trainer, *args, **kwargs):
    trainer.train()
    return trainer
