import time
from transparentmodel.analysis import model_size, memory_in_use, available_ram
from torch.profiler import profile, record_function, ProfilerActivity
import threading
import json


def capture_memory_usage(stop_event):
    # Create an empty list to store memory usage values
    memory_usage = []

    # Start time
    start_time = time.time()

    while not stop_event.is_set():
        # Get current memory usage
        current_memory = available_ram()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Append memory usage and elapsed time to the list
        memory_usage.append((elapsed_time, current_memory))

        print(f"Elapsed time: {elapsed_time:.2f}s | Memory usage: {json.dumps(available_ram(), indent=4)}")

        # Sleep for one second
        time.sleep(1)

    cpu_ram = [mem["system_ram"]["available"] for time, mem in memory_usage]
    # ToDo: GPU to be implemented

    # Print min cpu and gpu ram
    print(f"Min CPU RAM Available: {min(cpu_ram)} GB")
    print(f"Peak CPU RAM used: {cpu_ram[0]-min(cpu_ram)} GB")

    return memory_usage


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
