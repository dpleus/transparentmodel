from transparentmodel.utils import bytes_to_gb
import psutil

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
            model_size[device] = 0
        else:
            model_size[device] += param.nelement() * param.element_size()

    for buffer in model.buffers():
        device = buffer.device.type

        if device not in model_size:
            model_size[device] = 0
        else:
            model_size[device] += buffer.nelement() * buffer.element_size()
        model_size[device] += buffer.nelement() * buffer.element_size()

    model_size_gb = {k: bytes_to_gb(v) for k, v in model_size.items()}

    return model_size_gb


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
