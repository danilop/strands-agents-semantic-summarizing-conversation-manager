import sys
from typing import Any, Set


def estimate_memory_size(obj: Any, seen: Set[int] = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(
            estimate_memory_size(k, seen) + estimate_memory_size(v, seen)
            for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(estimate_memory_size(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += estimate_memory_size(obj.__dict__, seen)
    elif hasattr(obj, "__slots__"):
        size += sum(
            estimate_memory_size(getattr(obj, slot, None), seen)
            for slot in obj.__slots__
            if hasattr(obj, slot)
        )
    return size


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


