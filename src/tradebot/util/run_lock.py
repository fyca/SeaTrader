"""
File-based run lock to prevent concurrent rebalance and risk-check execution.
Prevents race conditions where both try to modify positions simultaneously.
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class RunLock:
    """Simple file-based lock for rebalance/risk-check coordination."""
    
    def __init__(self, lock_dir: Optional[Path] = None):
        """Initialize lock manager.
        
        Args:
            lock_dir: Directory for lock files. Defaults to data/locks/
        """
        if lock_dir is None:
            lock_dir = Path("data/locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = lock_dir
        self.rebalance_lock = lock_dir / "rebalance.lock"
        self.risk_check_lock = lock_dir / "risk_check.lock"
        self.lock_timeout_seconds = 3600  # 1 hour timeout (rebalance shouldn't take this long)
    
    def is_stale(self, lock_file: Path) -> bool:
        """Check if lock file is stale (older than timeout)."""
        if not lock_file.exists():
            return False
        age = time.time() - lock_file.stat().st_mtime
        return age > self.lock_timeout_seconds
    
    def acquire_rebalance_lock(self) -> bool:
        """Acquire lock for rebalance. Blocks risk-check.
        
        Returns:
            True if lock acquired, False if already held.
        """
        # Check if rebalance is already running
        if self.rebalance_lock.exists():
            if self.is_stale(self.rebalance_lock):
                # Stale lock, remove it
                self.rebalance_lock.unlink()
            else:
                return False
        
        # Write lock file with timestamp
        self.rebalance_lock.write_text(
            f"rebalance_lock\nstart_time: {datetime.utcnow().isoformat()}\npid: {os.getpid()}"
        )
        return True
    
    def release_rebalance_lock(self):
        """Release rebalance lock."""
        if self.rebalance_lock.exists():
            self.rebalance_lock.unlink()
    
    def is_rebalance_running(self) -> bool:
        """Check if rebalance is currently running."""
        if not self.rebalance_lock.exists():
            return False
        if self.is_stale(self.rebalance_lock):
            self.rebalance_lock.unlink()
            return False
        return True
    
    def acquire_risk_check_lock(self) -> bool:
        """Acquire lock for risk-check. Blocked by rebalance.
        
        Returns:
            True if lock acquired (and rebalance not running), False if rebalance is running.
        """
        # If rebalance is running, skip risk-check
        if self.is_rebalance_running():
            return False
        
        # Check if risk-check is already running (shouldn't be, they're fast)
        if self.risk_check_lock.exists():
            if self.is_stale(self.risk_check_lock):
                self.risk_check_lock.unlink()
            else:
                return False
        
        # Write lock file
        self.risk_check_lock.write_text(
            f"risk_check_lock\nstart_time: {datetime.utcnow().isoformat()}\npid: {os.getpid()}"
        )
        return True
    
    def release_risk_check_lock(self):
        """Release risk-check lock."""
        if self.risk_check_lock.exists():
            self.risk_check_lock.unlink()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release locks on context exit."""
        self.release_rebalance_lock()
        self.release_risk_check_lock()
