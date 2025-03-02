import random
import time
from typing import Callable, Iterable, Optional

from .types import P, T


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts before giving up
        delay: Initial delay between retries in seconds
        backoff: Multiplicative factor to increase delay between retries
        exceptions: Tuple of exception types to catch and retry on

    Returns:
        Decorated function that will retry on specified exceptions
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff

            assert last_exception is not None  # for type checker
            raise last_exception

        return wrapper

    return decorator


def shuffled(iterable: Iterable[T]) -> list[T]:
    """Returns a shuffled list of the input iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items
