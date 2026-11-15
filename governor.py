"""
governor.py
-----------
Intelligent Computer Sizing Algorithm (ICSA v4)

Responsibilities:
  - Evaluate active free RAM via psutil
  - Calculate the optimal chunk window (bytes) using the ICSA v4 formula:
        max(65536, int((free_ram * (1 - pressure) * 0.15) / (2.5 * max_connections)))
  - Throttling interceptor: pause async pipe when global RAM use >= 85%
  - Expose chunk-window size to data_engine.py at runtime
  - Helper: evict array variables and force gc.collect()
"""

import gc
import logging
import time
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Safety buffer -- execution is capped at 15% of active free RAM
SAFETY_BUFFER: float = 0.15

#: Concatenation overhead multiplier (xarray.concat spikes up to 2.5x)
CONCAT_OVERHEAD: float = 2.5

#: Hard ceiling: pause streaming if host-wide RAM use reaches this fraction
THROTTLE_CEILING: float = 0.85  # 85 %

#: Absolute minimum chunk size in bytes (one 64 KiB HDF5 page)
MIN_CHUNK_BYTES: int = 65_536  # 64 KiB

#: How long (seconds) to sleep between throttle-check polls
THROTTLE_POLL_INTERVAL: float = 0.5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ICSA v4 -- Intelligent Computer Sizing Algorithm
# ---------------------------------------------------------------------------

class ICSAv4:
    """
    Hardware-aware memory governor for virtual SWOT data streaming.

    Parameters
    ----------
    max_connections : int
        Size of the concurrent async connection pool used by data_engine.py.
        Defaults to 4, matching the PRD semaphore-guarded aiohttp pool.
    """

    def __init__(self, max_connections: int = 4) -> None:
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        self.max_connections: int = max_connections

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def optimal_chunk_bytes(self) -> int:
        """
        Calculate the optimal chunk window in bytes.

        Formula (ICSA v4):
            window = max(65536,
                         int((free_ram * (1 - pressure) * 0.15)
                             / (2.5 * max_connections)))

        Returns
        -------
        int
            Byte count capped to a minimum of 65 536 (one HDF5 page).
        """
        mem = psutil.virtual_memory()
        free_ram: int = mem.available          # bytes actually usable right now
        pressure: float = mem.percent / 100.0  # fraction [0.0, 1.0]

        raw = (free_ram * (1.0 - pressure) * SAFETY_BUFFER) / (
            CONCAT_OVERHEAD * self.max_connections
        )
        window = max(MIN_CHUNK_BYTES, int(raw))

        logger.debug(
            "ICSA v4 | free_ram=%.1f MB  pressure=%.1f%%  "
            "max_conn=%d  chunk_window=%d bytes",
            free_ram / 1024**2,
            pressure * 100,
            self.max_connections,
            window,
        )
        return window

    # ------------------------------------------------------------------
    # Throttling interceptor
    # ------------------------------------------------------------------

    def throttle_if_needed(self, poll_interval: float = THROTTLE_POLL_INTERVAL) -> None:
        """
        Block execution (busy-wait with sleep) while host-wide RAM use >= 85%.

        Call this before initiating each async byte-range fetch batch so the
        pipeline automatically stalls rather than triggering an OOM panic.

        Parameters
        ----------
        poll_interval : float
            Seconds to sleep between successive psutil polls.
        """
        while True:
            usage: float = psutil.virtual_memory().percent / 100.0
            if usage < THROTTLE_CEILING:
                break
            logger.warning(
                "ICSA v4 throttle active -- host RAM at %.1f%% (ceiling %.0f%%). "
                "Pausing %.1fs ...",
                usage * 100,
                THROTTLE_CEILING * 100,
                poll_interval,
            )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Convenience snapshot
    # ------------------------------------------------------------------

    def memory_snapshot(self) -> dict:
        """
        Return a lightweight dict of current memory metrics for the UI gauges.

        Keys
        ----
        total_gb, available_gb, used_pct, chunk_window_mb
        """
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1024**3,
            "available_gb": mem.available / 1024**3,
            "used_pct": mem.percent,
            "chunk_window_mb": self.optimal_chunk_bytes() / 1024**2,
        }


# ---------------------------------------------------------------------------
# Module-level helper: proactive array eviction
# ---------------------------------------------------------------------------

def evict(*arrays: Any) -> None:
    """
    Drop extra references to one or more array objects and force an immediate
    CPython garbage-collection pass to reclaim ghost memory blocks.

    Usage (inside data_engine.py fetch loop)::

        evict(chunk_data, clipped_data)
        del chunk_data, clipped_data   # remove caller-scope bindings too

    Parameters
    ----------
    *arrays : Any
        Any number of objects to be dereferenced.  The caller's local
        binding is NOT automatically deleted; this function drops the
        extra reference held by the *arrays tuple and then calls gc.collect().
        Always pair with an explicit ``del`` in the caller to fully release.
    """
    del arrays
    collected = gc.collect()
    logger.debug("gc.collect() reclaimed %d unreachable objects.", collected)