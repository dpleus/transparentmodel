import time
from transparentmodel.analysis import model_size, memory_in_use, available_ram
from torch.profiler import profile, record_function, ProfilerActivity
import threading
import json

from transparentmodel.utils import capture_memory_usage


def track_memory_usage(func):
    def wrapper(model, input_token, *args, **kwargs):
        print("---------- System memory ----------")
        print(f"Memory available: {json.dumps(available_ram(), indent=4)}")
        print(f"Memory used by process: {memory_in_use()} ")

        print("")
        print("---------- Model ----------")
        print(f"Model Size: {model_size(model)} GB")
        print(f"Model dtype: {model.dtype}")

        print("")
        print("---------- Memory usage ----------")
        stop_event = threading.Event()

        memory_thread = threading.Thread(target=capture_memory_usage, args=(stop_event,))
        memory_thread.start()

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
            with record_function("func"):
                model_output = func(model, input_token, *args, **kwargs)
                stop_event.set()

        memory_thread.join()

        print("")
        print("---------- Detailed Information ----------")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        return model_output

    return wrapper


@track_memory_usage
def generate_with_memory_tracking(model, input_token, *args, **kwargs):
    output = model.generate(input_token, **kwargs)
    return output
