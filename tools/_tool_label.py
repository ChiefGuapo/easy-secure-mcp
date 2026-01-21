from functools import wraps
import inspect

def tool_label(func):
    """
    Mark a function as an MCP tool without altering its async/sync nature.
    Ensures async functions are awaited by returning an async wrapper.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        async_wrapper._tool_label = True  # type: ignore[attr-defined]
        async_wrapper.__tool_name__ = func.__name__  # type: ignore[attr-defined]
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        sync_wrapper._tool_label = True  # type: ignore[attr-defined]
        sync_wrapper.__tool_name__ = func.__name__  # type: ignore[attr-defined]
        return sync_wrapper


