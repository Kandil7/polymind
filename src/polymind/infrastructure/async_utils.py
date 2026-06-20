"""Shared utilities for running async code from sync contexts."""

from __future__ import annotations

import asyncio
import concurrent.futures


def run_async(coro):
    """Run an async coroutine from a sync context safely.

    Handles both cases:
    - No event loop running: creates a new one with asyncio.run()
    - Event loop already running (e.g., inside a thread): uses a thread pool

    Args:
        coro: The async coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    else:
        return asyncio.run(coro)
