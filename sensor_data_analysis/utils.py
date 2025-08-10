import subprocess
from contextlib import contextmanager
from typing import Any, Iterator
import ray


def get_pip_dependencies() -> list[str]:
    try:
        result = subprocess.run(
            ["pip", "freeze"], stdout=subprocess.PIPE, text=True, check=True
        )
    except FileNotFoundError:
        result = subprocess.run(
            ["uv", "pip", "freeze"], stdout=subprocess.PIPE, text=True, check=True
        )
    return result.stdout.splitlines()


@contextmanager
def init_ray(*args: Any, **kwargs: Any) -> Iterator[None]:
    if "runtime_env" not in kwargs:
        kwargs["runtime_env"] = {"pip": get_pip_dependencies()}

    ray.init(*args, **kwargs)  # type: ignore
    yield
    ray.shutdown()  # type: ignore
