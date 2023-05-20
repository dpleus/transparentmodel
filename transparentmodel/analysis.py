from transparentmodel.utils import bytes_to_gb, bytes_to_mb
import psutil
import torch

import time
import json

def memory_in_use():
    process = psutil.Process()
    memory_used_bytes = process.memory_info().rss
    memory_used_gb = bytes_to_gb(memory_used_bytes)
    return memory_used_gb


def model_size(model):
    model_size = {}
    for param in model.parameters():
        device = param.device.type
        if device not in model_size:
            model_size[device] = param.nelement() * param.element_size()
        else:
            model_size[device] += param.nelement() * param.element_size()

    for buffer in model.buffers():
        device = buffer.device.type

        if device not in model_size:
            model_size[device] = buffer.nelement() * buffer.element_size()
        else:
            model_size[device] += buffer.nelement() * buffer.element_size()
        model_size[device] += buffer.nelement() * buffer.element_size()

    model_size_gb = {k: bytes_to_mb(v) for k, v in model_size.items()}

    return model_size_gb


def gradients_size(grad_input):
    gradient_size = {}

    for grads in grad_input:
        device = grads.device.type
        gradients = grads.data
        if gradients is not None:
            if device not in gradient_size:
                gradient_size[device] = gradients.nelement() * gradients.element_size()
            else:
                gradient_size[device] += gradients.nelement() * gradients.element_size()

    gradient_size_gb = {k: bytes_to_mb(v) for k, v in gradient_size.items()}

    return gradient_size_gb


def optimizer_state_size(optimizer):
    optimizer_state_size = {}
    for key, values in optimizer.state.items():
        for sub_key, sub_values in values.items():
            if isinstance(sub_values, torch.Tensor):
                state = sub_values
                device = state.device.type
                if device not in optimizer_state_size:
                    optimizer_state_size[device] = state.nelement() * state.element_size()
                else:
                    optimizer_state_size[device] += state.nelement() * state.element_size()

    optimizer_state_size_gb = {k: bytes_to_mb(v) for k, v in optimizer_state_size.items()}

    return optimizer_state_size_gb


def activations_size(act_input):
    activations_size = {}

    device = act_input.device.type
    activations = act_input

    if activations is not None:
        if device not in activations_size:
            activations_size[device] = activations.nelement() * activations.element_size()
        else:
            activations_size[device] += activations.nelement() * activations.element_size()

    activations_size_mb = {k: bytes_to_mb(v) for k, v in activations_size.items()}

    return activations_size_mb


def available_ram():
    # System RAM

    ram_info = {}

    # System RAM
    mem_info = psutil.virtual_memory()
    total_ram_gb = bytes_to_gb(mem_info.total)
    available_ram_gb = bytes_to_gb(mem_info.available)
    ram_info["system_ram"] = {
        "total": total_ram_gb,
        "available": available_ram_gb
    }

    # GPU RAM
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_ram_info = []
            for i in range(gpu_count):
                gpu_info = torch.cuda.get_device_properties(i)
                total_gpu_ram_gb = bytes_to_gb(gpu_info.total_memory)
                available_gpu_ram_gb = bytes_to_gb(torch.cuda.max_memory_allocated(i))
                gpu_ram_info.append({
                    "gpu_index": i,
                    "total_ram": total_gpu_ram_gb,
                    "available_ram": available_gpu_ram_gb
                })
            ram_info["gpu_ram"] = gpu_ram_info
        else:
            ram_info["gpu_ram"] = "No GPU available."
    except ImportError:
        ram_info["gpu_ram"] = "PyTorch is not installed. Skipping GPU RAM information."

    return ram_info


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
    print(f"Peak CPU RAM used: {cpu_ram[0] - min(cpu_ram)} GB")

    return memory_usage
