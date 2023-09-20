from transparentmodel.pytorch_hooks import (
    backward_hook_gradients_general,
    forward_hook_activations_torch,
)
from transparentmodel.analysis import available_ram, model_size
from transparentmodel.analysis import capture_memory_usage, memory_in_use
import threading
import json


def _register_hooks(model):
    model.register_forward_hook(forward_hook_activations_torch)
    model.register_backward_hook(backward_hook_gradients_general)
    return model


def apply_tracking(model, realtime=True):
    model = _register_hooks(model)

    print("---------- System memory ----------")
    print(f"Memory available: {json.dumps(available_ram(), indent=4)}")
    print(f"Memory used by process: {memory_in_use()} ")

    print("")
    print("---------- Model ----------")
    print(f"Model Size: {model_size(model)} MB")

    if realtime:
        stop_event = threading.Event()

        memory_thread = threading.Thread(
            target=capture_memory_usage, args=(stop_event,)
        )
        memory_thread.start()

    return model
