"""
Rate-limit handling for Alpaca API calls with exponential backoff.
Prevents failures during peak trading hours when API rate limits are hit.
"""

import time
from functools import wraps
from typing import Callable, Any, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def retry_on_rate_limit(
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    backoff_multiplier: float = 2.0
) -> Callable[[F], F]:
    """Decorator to retry Alpaca API calls on rate limit errors.
    
    Args:
        max_retries: Maximum number of retries (default 5)
        initial_backoff: Starting backoff in seconds (default 1.0)
        max_backoff: Maximum backoff in seconds (default 60.0)
        backoff_multiplier: Multiply backoff by this after each retry (default 2.0, exponential)
    
    Retries on:
    - HTTP 429 (Too Many Requests) from Alpaca
    - ConnectionError or timeout
    - Other transient errors (marked as such by exception message)
    
    Example:
        @retry_on_rate_limit(max_retries=3)
        def get_bars(self, symbol):
            return self.api.get_latest_bar(symbol)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            backoff = initial_backoff
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Check if error is rate-limit or transient
                    is_rate_limit = (
                        "429" in error_str or
                        "rate limit" in error_str or
                        "too many requests" in error_str
                    )
                    is_transient = (
                        "connection" in error_str or
                        "timeout" in error_str or
                        "temporarily unavailable" in error_str or
                        "service unavailable" in error_str
                    )
                    
                    if not (is_rate_limit or is_transient):
                        # Not a transient error, don't retry
                        raise
                    
                    if attempt >= max_retries - 1:
                        # Last retry, give up
                        raise
                    
                    # Sleep before retry
                    print(f"[yellow]Rate limit/transient error[/yellow]: {e}. Retrying in {backoff:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    
                    # Increase backoff exponentially, capped at max_backoff
                    backoff = min(backoff * backoff_multiplier, max_backoff)
            
            # Should not reach here, but raise last error if we do
            raise last_error
        
        return wrapper  # type: ignore
    
    return decorator
