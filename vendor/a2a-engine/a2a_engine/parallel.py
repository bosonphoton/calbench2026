"""Run a list of tasks either sequentially (max_workers=1) or concurrently.

    results, errors = run_with_parallelism(
        fn=run_one,
        items=contexts,
        max_workers=4,
        on_result=lambda ctx, r: log.info("done %s", ctx.id),
        on_error=lambda ctx, e: log.error("failed %s: %s", ctx.id, e),
    )
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_with_parallelism(
    fn: Callable[[T], R],
    items: list[T],
    max_workers: int = 1,
    on_result: Callable[[T, R], None] | None = None,
    on_error: Callable[[T, Exception], None] | None = None,
) -> tuple[list[tuple[T, R]], list[tuple[T, Exception]]]:
    """Run fn(item) for each item. max_workers=1 -> plain loop; >1 -> ThreadPoolExecutor."""
    results: list[tuple[T, R]] = []
    errors: list[tuple[T, Exception]] = []

    if not items:
        return results, errors

    if max_workers <= 1:
        for item in items:
            try:
                value = fn(item)
            except Exception as e:
                errors.append((item, e))
                if on_error:
                    on_error(item, e)
                continue
            results.append((item, value))
            if on_result:
                on_result(item, value)
    else:
        workers = min(max_workers, len(items))
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fn, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    value = future.result()
                except Exception as e:
                    with lock:
                        errors.append((item, e))
                    if on_error:
                        on_error(item, e)
                    continue
                with lock:
                    results.append((item, value))
                if on_result:
                    on_result(item, value)

    return results, errors
