
import gc
import torch


def cuda_memory_profiler(step_name: str = ""):
    """打印当前 GPU 显存状态"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [显存|{step_name}] "
          f"已分配={allocated:.2f}GB / 已保留={reserved:.2f}GB / 总计={total:.2f}GB")


def clear_gpu_memory(*tensors):
    """删除指定张量并清理显存"""
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()