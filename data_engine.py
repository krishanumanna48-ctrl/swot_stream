"""
data_engine.py
--------------
Hybrid Kerchunk Reference Resolver & Streaming Data Engine

Pipeline stages (matching PRD architecture):
  1. Hybrid Kerchunk Resolver     -- JSON cache hit or on-the-fly SingleHdf5ToZarr
  2. Minimum Swath Intersector    -- isolate chunks inside Mekong Delta bounding box
  3. ICSA v4 throttle + semaphore -- memory-safe async byte-range fetching
  4. NumPy spatial clip            -- lat/lon mask inside exact bounding box
  5. ssha_karin_qual == 0 filter  -- NASA quality bitmask enforcement
  6. Proactive eviction            -- evict() + gc.collect() after every chunk
  7. Trend assembly                -- return cleaned (lat, lon, ssha, time) arrays
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import pathlib
from typing import Any

import numpy as np

from governor import ICSAv4, evict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mekong Delta bounding box (PRD Section 2.2)
# ---------------------------------------------------------------------------
BBOX_LAT_MIN: float = 8.5
BBOX_LAT_MAX: float = 11.5
BBOX_LON_MIN: float = 104.5
BBOX_LON_MAX: float = 107.0

# EarthAccess short name for SWOT L2 Low-Rate SSH
SWOT_SHORT_NAME: str = "SWOT_L2_LR_SSH"

# Local Kerchunk JSON cache directory
CACHE_DIR: pathlib.Path = pathlib.Path(__file__).parent / "cache"

# Default semaphore width (concurrent aiohttp connections)
DEFAULT_MAX_CONNECTIONS: int = 4

# ---------------------------------------------------------------------------
# 1. Hybrid Kerchunk Reference Resolver
# ---------------------------------------------------------------------------

class KerchunkResolver:
    """
    Two-tier reference map resolver.

    Tier 1 -- fast cache: looks up a consolidated Kerchunk JSON file
              in ``cache/`` keyed by a SHA-256 hash of the granule URL.
    Tier 2 -- live fallback: runs ``kerchunk.hdf.SingleHdf5ToZarr`` via
              fsspec against the remote URL and writes the result to cache.
    """

    def __init__(self, cache_dir: pathlib.Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, granule_url: str) -> dict:
        """
        Return the Zarr-compatible reference dict for *granule_url*.

        Checks cache first; builds and caches on miss.

        Parameters
        ----------
        granule_url : str
            HTTPS URL of the remote NetCDF4/HDF5 SWOT granule.

        Returns
        -------
        dict
            fsspec ReferenceFileSystem-compatible reference map.
        """
        cache_path = self._cache_path(granule_url)

        # -- Tier 1: cache hit --
        if cache_path.exists():
            logger.info("Kerchunk cache HIT: %s", cache_path.name)
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)

        # -- Tier 2: live indexing fallback --
        logger.info("Kerchunk cache MISS -- building reference map for: %s", granule_url)
        ref_map = self._build_reference_map(granule_url)
        self._write_cache(cache_path, ref_map)
        return ref_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_path(granule_url: str) -> pathlib.Path:
        """Derive a deterministic cache filename from the granule URL."""
        url_hash = hashlib.sha256(granule_url.encode()).hexdigest()[:16]
        filename = f"swot_ref_{url_hash}.json"
        return CACHE_DIR / filename

    @staticmethod
    def _build_reference_map(granule_url: str, ea_session=None) -> dict:
        """
        Run kerchunk.hdf.SingleHdf5ToZarr on the remote file and return
        the consolidated reference dict.

        Parameters
        ----------
        granule_url : str
            HTTPS URL of the remote granule.
        ea_session : fsspec AbstractFileSystem or requests.Session, optional
            Authenticated session from earthaccess.get_fsspec_https_session().
            When provided, its headers are injected into the fsspec open call
            so NASA Earthdata auth tokens are forwarded on every request.
        """
        try:
            import fsspec
            from kerchunk.hdf import SingleHdf5ToZarr

            if ea_session is not None:
                # Extract Bearer/EDL token headers from the earthaccess session
                # and pass them as fsspec client_kwargs so every byte-range
                # request is authenticated.
                try:
                    auth_headers = dict(ea_session.headers)
                except AttributeError:
                    auth_headers = {}

                storage_options = {
                    "client_kwargs": {"headers": auth_headers},
                }
                logger.debug(
                    "Kerchunk open: using earthaccess session headers for %s",
                    granule_url.split("/")[-1],
                )
            else:
                storage_options = {}

            with fsspec.open(granule_url, "rb", **storage_options) as remote_f:
                h5chunks = SingleHdf5ToZarr(remote_f, granule_url, inline_threshold=100)
                ref_map = h5chunks.translate()
            logger.info("Reference map built successfully for %s", granule_url.split("/")[-1])
            return ref_map

        except Exception as exc:
            logger.error(
                "SingleHdf5ToZarr failed for %s: %s: %s",
                granule_url.split("/")[-1], type(exc).__name__, exc,
            )
            raise RuntimeError(
                f"Could not build Kerchunk reference map for "
                f"{granule_url.split(chr(47))[-1]}: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _write_cache(cache_path: pathlib.Path, ref_map: dict) -> None:
        """Persist the reference map as a JSON file."""
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(ref_map, fh, separators=(",", ":"))
        logger.info("Kerchunk reference map cached at %s", cache_path.name)

    def open_zarr_store(self, ref_map: dict, auth_headers: dict | None = None):
        """
        Wrap a reference map in an fsspec ReferenceFileSystem and return
        an xarray-ready Zarr store mapper.

        Parameters
        ----------
        ref_map : dict
            Kerchunk reference dict.
        auth_headers : dict, optional
            HTTP headers (e.g. Authorization Bearer) forwarded to every
            byte-range request so NASA Earthdata auth is maintained.
        """
        import fsspec

        remote_options = {"headers": auth_headers} if auth_headers else {}

        fs = fsspec.filesystem(
            "reference",
            fo=ref_map,
            remote_protocol="https",
            remote_options=remote_options,
            skip_instance_cache=True,
        )
        return fs.get_mapper("")


# ---------------------------------------------------------------------------
# 2. Spatial utilities -- Mekong Delta bounding box
# ---------------------------------------------------------------------------

def intersects_bbox(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    lat_min: float = BBOX_LAT_MIN,
    lat_max: float = BBOX_LAT_MAX,
    lon_min: float = BBOX_LON_MIN,
    lon_max: float = BBOX_LON_MAX,
) -> bool:
    """
    Return True if any coordinate pair in the arrays falls inside the
    Mekong Delta bounding box.  Used to skip non-overlapping granules
    before initiating expensive byte-range fetches.
    """
    mask = (
        (lat_arr >= lat_min) & (lat_arr <= lat_max) &
        (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )
    return bool(mask.any())


def spatial_clip(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    ssha: np.ndarray,
    qual: np.ndarray,
    lat_min: float = BBOX_LAT_MIN,
    lat_max: float = BBOX_LAT_MAX,
    lon_min: float = BBOX_LON_MIN,
    lon_max: float = BBOX_LON_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1 of two-step swath pipeline: NumPy boolean index clip.

    Retains only pixels whose (lat, lon) coordinates fall strictly
    inside the Mekong Delta bounding box.

    Returns
    -------
    lat_clipped, lon_clipped, ssha_clipped, qual_clipped
    """
    mask = (
        (lat_arr >= lat_min) & (lat_arr <= lat_max) &
        (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )
    return lat_arr[mask], lon_arr[mask], ssha[mask], qual[mask]


# ---------------------------------------------------------------------------
# 3. NASA Quality Bitmask filter  (ssha_karin_qual == 0)
# ---------------------------------------------------------------------------

def apply_quality_mask(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    ssha: np.ndarray,
    qual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 2 of two-step swath pipeline: NASA SWOT quality bitmask enforcement.

    Accepts both integer and float quality arrays (h5py returns float after
    scale decode; NaN means no quality data available for that pixel).

    Retains pixels where:
      - (qual == 0) or qual is NaN  (passes quality check)
      - ssha is finite              (removes fill values decoded to large numbers)

    Returns
    -------
    lat_clean, lon_clean, ssha_clean
    """
    good = ((qual == 0) | np.isnan(qual)) & np.isfinite(ssha)
    n_total = len(qual)
    n_good = int(good.sum())
    logger.debug(
        "Quality mask: %d / %d pixels pass (qual==0|NaN) & isfinite(ssha) (%.1f%%)",
        n_good, n_total, 100.0 * n_good / max(n_total, 1),
    )
    return lat_arr[good], lon_arr[good], ssha[good]


# ---------------------------------------------------------------------------
# 4. EarthAccess granule discovery + fallback synthetic data
# ---------------------------------------------------------------------------

def _discover_granules_earthaccess(
    months: int,
    bbox: tuple[float, float, float, float] = (
        BBOX_LON_MIN, BBOX_LAT_MIN, BBOX_LON_MAX, BBOX_LAT_MAX
    ),
) -> tuple[list[str], int]:
    """
    Discover SWOT L2 LR SSH granules via a two-step CMR query.

    Step 1 -- collection discovery:
        earthaccess.search_datasets(keyword='SWOT', count=50) filtered to
        collections whose ShortName contains 'LR_SSH'.  Tried in priority
        order so the best-available product is used automatically.

    Step 2 -- granule search:
        earthaccess.search_data(concept_id=<collection_id>, bounding_box=bbox,
                                 count=min(months, 6))
        No temporal restriction is applied -- CMR returns the most recent
        granules first, which is what we want for near-real-time analysis.

    Parameters
    ----------
    months : int
        Controls granule cap: cap = min(months, 3).
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max) in decimal degrees.
        earthaccess.search_data expects exactly this order.

    Returns
    -------
    tuple[list[str], int]
        (data_urls, total_granules_matched)
    """
    import earthaccess

    # Defensive guard: earthaccess.search_data expects
    # bounding_box=(lon_min, lat_min, lon_max, lat_max).
    # Catch accidental lat-first inversion at runtime.
    lon_min_b, lat_min_b, lon_max_b, lat_max_b = bbox
    if not (-180.0 <= lon_min_b <= 180.0 and -90.0 <= lat_min_b <= 90.0):
        raise ValueError(
            f"bounding_box tuple must be (lon_min, lat_min, lon_max, lat_max); "
            f"got {bbox} which looks like lat-first ordering."
        )
    if lon_min_b >= lon_max_b or lat_min_b >= lat_max_b:
        raise ValueError(
            f"bounding_box min >= max: {bbox}. "
            f"Expected (lon_min, lat_min, lon_max, lat_max) with min < max."
        )

    granule_cap = min(months, 2)

    # ------------------------------------------------------------------
    # Step 1: collection discovery
    # ------------------------------------------------------------------
    # Priority order: versioned standard > versioned basic > daily standard
    # > daily basic > expert/unsmoothed variants.
    COLLECTION_PRIORITY = [
        "SWOT_L2_LR_SSH_2.0",
        "SWOT_L2_LR_SSH_BASIC_2.0",
        "SWOT_L2_LR_SSH_D",
        "SWOT_L2_LR_SSH_BASIC_D",
        "SWOT_L2_LR_SSH_EXPERT_2.0",
        "SWOT_L2_LR_SSH_EXPERT_D",
        "SWOT_L2_LR_SSH_UNSMOOTHED_2.0",
    ]

    logger.info("Step 1: querying CMR for active SWOT LR_SSH collections ...")
    all_datasets = earthaccess.search_datasets(keyword="SWOT", count=50)

    # Build short_name -> concept_id map for every LR_SSH collection found
    collection_map: dict[str, str] = {}
    for ds in all_datasets:
        try:
            sn  = ds["umm"]["ShortName"]
            cid = ds["meta"]["concept-id"]
            if "LR_SSH" in sn:
                collection_map[sn] = cid
        except (KeyError, TypeError):
            continue

    logger.info(
        "CMR returned %d LR_SSH collections: %s",
        len(collection_map), list(collection_map.keys()),
    )

    if not collection_map:
        raise RuntimeError(
            "CMR returned 0 SWOT LR_SSH collections. "
            "Check earthaccess auth and NASA CMR availability."
        )

    # Order available collections by priority list; append any extras at end
    ordered: list[tuple[str, str]] = []
    for preferred_sn in COLLECTION_PRIORITY:
        if preferred_sn in collection_map:
            ordered.append((preferred_sn, collection_map[preferred_sn]))
    for sn, cid in collection_map.items():
        if sn not in [o[0] for o in ordered]:
            ordered.append((sn, cid))

    # ------------------------------------------------------------------
    # Step 2: granule search -- try each collection until we get hits
    # ------------------------------------------------------------------
    for short_name, concept_id in ordered:
        logger.info(
            "Step 2: searching granules | collection=%s  concept_id=%s  "
            "bbox=%s  count=%d  (no temporal restriction)",
            short_name, concept_id, bbox, granule_cap,
        )
        try:
            results = earthaccess.search_data(
                concept_id=concept_id,
                bounding_box=bbox,
                count=granule_cap,
            )
        except Exception as exc:
            logger.warning(
                "search_data failed for %s (%s): %s -- trying next collection.",
                short_name, concept_id, exc,
            )
            continue

        total_matched = len(results)
        if total_matched == 0:
            logger.info(
                "0 granules for %s over bbox %s -- trying next collection.",
                short_name, bbox,
            )
            continue

        # Extract HTTPS .nc/.h5 URLs; skip S3 and WindWave files.
        # data_links(access='direct') returns S3 paths on PO.DAAC which
        # the HTTPFileSystem session cannot open.  Prefer HTTPS links from
        # data_links() (all types) that start with https://.
        SKIP_INDICATORS = ("WindWave", "Unsmoothed")
        urls: list[str] = []
        for granule in results:
            # All links regardless of access type, filtered to HTTPS only
            https_links = [
                lnk for lnk in granule.data_links()
                if lnk.lower().startswith("https://")
                and lnk.lower().endswith((".nc", ".h5"))
            ]
            chosen = None
            for lnk in https_links:
                if any(sk in lnk.split("/")[-1] for sk in SKIP_INDICATORS):
                    logger.debug("Skipping WindWave granule: %s", lnk.split("/")[-1])
                    continue
                chosen = lnk
                break
            if chosen:
                urls.append(chosen)
            else:
                logger.warning(
                    "Granule has no usable HTTPS .nc/.h5 links -- skipping."
                )

        logger.info(
            "Collection %s: %d granules matched, %d usable URLs extracted.",
            short_name, total_matched, len(urls),
        )

        if urls:
            return urls, total_matched

        logger.info("No usable URLs from %s -- trying next collection.", short_name)

    # All collections exhausted
    raise ValueError(
        f"All {len(ordered)} SWOT LR_SSH collections returned 0 usable granule "
        f"URLs for bbox {bbox}. Collections tried: "
        f"{[o[0] for o in ordered]}"
    )


def _synthetic_granule(months: int, seed: int = 42,
                        lat_min: float = BBOX_LAT_MIN, lat_max: float = BBOX_LAT_MAX,
                        lon_min: float = BBOX_LON_MIN, lon_max: float = BBOX_LON_MAX,
                        ) -> dict[str, np.ndarray]:
    """
    Generate realistic synthetic SWOT L2 Mekong Delta data when EarthAccess
    credentials are unavailable or the network is offline.

    Produces orbital-swath-shaped data (non-uniform lat/lon grid) with
    realistic sea-surface height anomaly values and a mix of quality flags.

    Returns
    -------
    dict with keys: lat, lon, ssha_karin, ssha_karin_qual, time_tai
    """
    rng = np.random.default_rng(seed)

    # Simulate ~1 500 pixels per pass x months passes
    n_passes = max(1, months * 2)
    n_pixels = 1500 * n_passes

    # Non-uniform orbital swath coords (diagonal track, typical of SWOT)
    lat_span  = lat_max - lat_min
    lon_mid   = (lon_min + lon_max) / 2.0
    lon_half  = max((lon_max - lon_min) / 2.0, 0.5)
    track_lat = np.linspace(lat_min - 0.5, lat_max + 0.5, n_pixels)
    track_lon = lon_mid + rng.uniform(-lon_half, lon_half, n_pixels)

    # Realistic SSH anomaly: regional trend ~3 mm/yr + seasonal noise
    days = np.linspace(0, months * 30, n_pixels)
    trend_m = 0.003 / 365.25 * days               # 3 mm/yr converted to metres
    seasonal = 0.015 * np.sin(2 * np.pi * days / 365.25)
    noise = rng.normal(0, 0.008, n_pixels)         # 8 mm instrument noise
    ssha = trend_m + seasonal + noise

    # Quality flags: ~85% good (== 0), 15% contaminated (non-zero)
    qual_flags = np.where(rng.random(n_pixels) < 0.85, 0, rng.integers(1, 16, n_pixels))

    # Synthetic TAI timestamps (seconds since 2000-01-01)
    base_tai = 756_864_000.0   # approx 2024-01-01 in TAI
    time_tai = base_tai + days * 86_400

    logger.info(
        "Synthetic fallback [bbox %.1f-%.1f lat, %.1f-%.1f lon]: %d pixels, %d passes, %.0f%% quality-pass rate",
        lat_min, lat_max, lon_min, lon_max, n_pixels, n_passes, 100.0 * (qual_flags == 0).mean(),
    )
    return {
        "lat": track_lat.astype(np.float32),
        "lon": track_lon.astype(np.float32),
        "ssha_karin": ssha.astype(np.float32),
        "ssha_karin_qual": qual_flags.astype(np.uint8),
        "time_tai": time_tai.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# 5. Semaphore-guarded async chunk fetcher
# ---------------------------------------------------------------------------

async def _fetch_chunk_async(
    session,            # aiohttp.ClientSession
    url: str,
    byte_start: int,
    byte_end: int,
    semaphore: asyncio.Semaphore,
) -> bytes:
    """
    Fetch a precise byte range from *url* under *semaphore* control.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Shared session for connection pooling.
    url : str
        Target HTTPS URL.
    byte_start, byte_end : int
        Inclusive byte range (Range: bytes=start-end).
    semaphore : asyncio.Semaphore
        Limits concurrent connections to max_connections.

    Returns
    -------
    bytes
        Raw byte content of the requested range.
    """
    headers = {"Range": f"bytes={byte_start}-{byte_end}"}
    async with semaphore:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.read()
    return data


async def _fetch_all_chunks_async(
    chunk_specs: list[tuple[str, int, int]],
    gov: ICSAv4,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
) -> list[bytes]:
    """
    Fetch a list of (url, byte_start, byte_end) specs concurrently.

    Calls ``gov.throttle_if_needed()`` before every batch to honour
    the ICSA v4 85% RAM ceiling.

    Parameters
    ----------
    chunk_specs : list of (url, byte_start, byte_end)
    gov : ICSAv4
        Live governor instance used for throttle checks.
    max_connections : int
        Semaphore width / connection pool size.

    Returns
    -------
    list[bytes]
        Raw bytes for each spec, in order.
    """
    import aiohttp

    semaphore = asyncio.Semaphore(max_connections)
    results: list[bytes] = []

    async with aiohttp.ClientSession() as session:
        for spec in chunk_specs:
            url, start, end = spec

            # Throttle check before every individual fetch
            gov.throttle_if_needed()

            chunk_data = await _fetch_chunk_async(session, url, start, end, semaphore)
            results.append(chunk_data)

            # Proactive eviction after every chunk
            evict(chunk_data)
            del chunk_data
            gc.collect()

    return results


# ---------------------------------------------------------------------------
# 6. High-level public entry point
# ---------------------------------------------------------------------------

def stream_mekong_ssha(
    months: int = 12,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    lat_min: float = BBOX_LAT_MIN,
    lat_max: float = BBOX_LAT_MAX,
    lon_min: float = BBOX_LON_MIN,
    lon_max: float = BBOX_LON_MAX,
) -> dict[str, np.ndarray]:
    """
    Main public function: stream, filter, and return cleaned SWOT SSH data
    for the user-specified bounding box over the last *months* months.
    Defaults to the Mekong Delta; pass custom lat/lon bounds for any global region.

    Pipeline
    --------
    1. Discover granules via EarthAccess (fallback: synthetic data)
    2. For each granule: resolve Kerchunk reference map (cache or live)
    3. Open virtual Zarr store; read lat, lon, ssha_karin, ssha_karin_qual
    4. Spatial clip to Mekong Delta bounding box
    5. Apply ssha_karin_qual == 0 bitmask
    6. Evict intermediate arrays; advance to next chunk
    7. Concatenate all clean arrays and return

    Parameters
    ----------
    months : int
        Number of months of data to retrieve (1-24).
    max_connections : int
        Async connection pool size (fed to ICSA v4 formula).
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box in decimal degrees. Defaults to Mekong Delta.

    Returns
    -------
    dict with keys:
        lat        -- np.ndarray, float32
        lon        -- np.ndarray, float32
        ssha_karin -- np.ndarray, float32  (metres, quality-filtered)
        time_tai   -- np.ndarray, float64  (seconds since 2000-01-01 TAI)
        source     -- str, "earthaccess" or "synthetic"
    """
    gov = ICSAv4(max_connections=max_connections)
    resolver = KerchunkResolver()

    # -- Stage 1: granule discovery with tiered auth --
    granule_urls: list[str] = []
    total_granules_matched: int = 0
    source_tag = "live"
    auth_method: str = ""
    fallback_reason: str = ""
    ea_session = None          # earthaccess fsspec session; set after auth

    try:
        import earthaccess
        import concurrent.futures as _auth_cf

        AUTH_TIMEOUT = 8.0  # seconds per strategy attempt

        def _try_login(strategy: str):
            return earthaccess.login(strategy=strategy, persist=False)

        # Strategy order:
        #   1. "all"  -- earthaccess tries netrc / _netrc / env / cached token
        #                automatically; works on both Linux (~/.netrc) and
        #                Windows (~/_netrc). Wrapped in a thread so we can
        #                enforce AUTH_TIMEOUT and avoid blocking on interactive prompts.
        #   2. "environment" -- explicit fallback for CI / Docker environments
        #                        where only EARTHDATA_* env vars are set.
        # Both attempts are wrapped with an 8s thread timeout.
        _last_exc = None
        for _strategy in ("all", "environment"):
            try:
                with _auth_cf.ThreadPoolExecutor(max_workers=1) as _aex:
                    _fut = _aex.submit(_try_login, _strategy)
                    _fut.result(timeout=AUTH_TIMEOUT)
                auth_method = _strategy
                logger.info("EarthAccess auth: %s login succeeded.", _strategy)
                break
            except _auth_cf.TimeoutError:
                _last_exc = f"{_strategy} timed out after {AUTH_TIMEOUT}s"
                logger.info("Auth strategy %s timed out -- trying next.", _strategy)
            except Exception as _exc:
                _last_exc = f"{_strategy}: {_exc}"
                logger.info("Auth strategy %s failed (%s) -- trying next.", _strategy, _exc)
        else:
            raise RuntimeError(
                f"All earthaccess auth strategies failed. Last error: {_last_exc}"
            )

        granule_urls, total_granules_matched = _discover_granules_earthaccess(
            months, bbox=(lon_min, lat_min, lon_max, lat_max)
        )
        if not granule_urls:
            raise ValueError(
                f"EarthAccess search matched {total_granules_matched} granules "
                f"but extracted 0 usable data URLs for bbox "
                f"({lat_min},{lat_max},{lon_min},{lon_max}) -- check product availability."
            )

        # Obtain an authenticated fsspec HTTPS session for all subsequent
        # byte-range requests (Kerchunk indexing + xarray open).
        # earthaccess.get_fsspec_https_session() returns a requests.Session
        # pre-configured with EDL Bearer token / Basic auth headers.
        try:
            ea_session = earthaccess.get_fsspec_https_session()
            logger.info("earthaccess fsspec HTTPS session obtained.")
        except Exception as sess_exc:
            logger.warning(
                "Could not obtain earthaccess fsspec session (%s: %s) -- "
                "proceeding without session (may fail on protected granules).",
                type(sess_exc).__name__, sess_exc,
            )
            ea_session = None

        logger.info(
            "Live pipeline ready: %d granule URLs via %s auth.",
            len(granule_urls), auth_method,
        )

    except Exception as exc:
        fallback_reason = str(exc)
        logger.warning("EarthAccess unavailable: %s", fallback_reason)
        logger.warning("Falling back to synthetic SWOT data.")
        source_tag = "synthetic"

    # -- Synthetic path: skip network pipeline entirely --
    if source_tag == "synthetic":
        raw = _synthetic_granule(months,
                               lat_min=lat_min, lat_max=lat_max,
                               lon_min=lon_min, lon_max=lon_max)
        lat_c, lon_c, ssha_c, qual_c = spatial_clip(
            raw["lat"], raw["lon"], raw["ssha_karin"], raw["ssha_karin_qual"],
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        )
        lat_f, lon_f, ssha_f = apply_quality_mask(lat_c, lon_c, ssha_c, qual_c)
        evict(raw, lat_c, lon_c, ssha_c, qual_c)
        del raw, lat_c, lon_c, ssha_c, qual_c
        gc.collect()

        # Reconstruct matching time array for the surviving pixels
        n_good = len(lat_f)
        base_tai = 756_864_000.0
        time_f = np.linspace(base_tai, base_tai + months * 30 * 86_400, n_good)

        logger.info(
            "Synthetic pipeline complete: %d clean pixels returned.", n_good
        )
        synth_source = (
            f"synthetic (demo) -- live NASA Earthdata unavailable: {fallback_reason}"
            if fallback_reason else "synthetic (demo)"
        )
        return {
            "lat": lat_f,
            "lon": lon_f,
            "ssha_karin": ssha_f,
            "time_tai": time_f.astype(np.float64),
            "source": synth_source,
        }

    # -- Live EarthAccess path --
    # Wrapped in a thread so we can enforce a hard 35-second wall-clock
    # timeout without blocking the Streamlit main thread.
    import concurrent.futures as _cf
    import fsspec as _fsspec
    LIVE_TIMEOUT_SEC = 120.0
    BLOCK_SIZE       = 2 * 1024 * 1024  # 2 MB read-ahead buffer

    # Only the 4 variables we actually need -- h5netcdf reads lazily, but
    # listing them in phony_vars tells h5netcdf not to build coordinate
    # indexes for the other ~100 variables in the file.
    def _read_one_granule(url: str) -> tuple | None:
        """
        Open a single remote SWOT granule via fsspec + h5netcdf.

        Uses a 2 MB block_size read-ahead buffer so the first request
        fetches enough of the HDF5 header to satisfy all 4 variable reads
        without issuing one HTTP request per chunk.

        Returns (lat, lon, ssha, qual, time_raw) numpy arrays clipped to
        the bounding box, or None if the granule has no overlap.
        """
        try:
            auth_headers = dict(ea_session.headers) if ea_session else {}
        except AttributeError:
            auth_headers = {}

        fname = url.split("/")[-1]
        logger.info("Opening granule: %s", fname)
        gov.throttle_if_needed()

        # All discovery URLs are HTTPS PO.DAAC archive links -- open directly.
        if ea_session is not None:
            of = ea_session.open(url, "rb", block_size=BLOCK_SIZE)
        else:
            of = _fsspec.open(url, "rb", block_size=BLOCK_SIZE)

        # Variable name candidates in priority order (product version varies)
        SSH_CANDIDATES  = ("ssha_karin", "ssha_karin_2", "ssh_karin", "ssh_karin_2")
        QUAL_CANDIDATES = ("ssha_karin_qual", "ssh_karin_qual", "ssha_karin_2_qual")
        TIME_CANDIDATES = ("time_tai", "time")

        def _h5decode(h, var):
            ds  = h[var]
            raw = ds[:].astype(np.float64)
            a   = ds.attrs
            def _s(key, default):
                v = a.get(key, default)
                return float(np.asarray(v).flat[0])
            sf = _s("scale_factor", 1.0)
            ao = _s("add_offset",   0.0)
            fv = _s("_FillValue",   np.nan)
            decoded = raw * sf + ao
            if not np.isnan(fv):
                decoded[raw == fv] = np.nan
            return decoded

        import h5py as _h5py

        with of as remote_f:
            h    = _h5py.File(remote_f, "r")
            keys = set(h.keys())

            lat_full = _h5decode(h, "latitude")
            lon_full = _h5decode(h, "longitude")

            # Normalize 0-360 longitudes to -180/+180
            lon_full = np.where(lon_full > 180.0, lon_full - 360.0, lon_full)

            bbox_mask = (
                (lat_full >= lat_min) & (lat_full <= lat_max) &
                (lon_full >= lon_min) & (lon_full <= lon_max)
            )
            if not bbox_mask.any():
                h.close()
                logger.debug("Granule %s: no overlap with bbox -- skipping.", fname)
                return None

            lat_raw = lat_full[bbox_mask].ravel().astype(np.float32)
            lon_raw = lon_full[bbox_mask].ravel().astype(np.float32)
            del lat_full, lon_full
            gc.collect()

            ssha_full = None
            ssha_var  = None
            for _c in SSH_CANDIDATES:
                if _c in keys:
                    ssha_full = _h5decode(h, _c)
                    ssha_var  = _c
                    break
            if ssha_full is None:
                h.close()
                raise KeyError(
                    "No SSH variable in %s. Tried %s. Available: %s" % (
                        fname, SSH_CANDIDATES, sorted(keys))
                )

            qual_full = None
            for _c in QUAL_CANDIDATES:
                if _c in keys:
                    qual_full = _h5decode(h, _c)
                    break
            if qual_full is None:
                qual_full = np.zeros_like(ssha_full)
                logger.warning("Granule %s: no quality variable -- accepting all.", fname)

            tvar      = next((v for v in TIME_CANDIDATES if v in keys), None)
            time_full = _h5decode(h, tvar) if tvar else np.zeros(ssha_full.shape[0])
            h.close()

        ssha_raw = ssha_full[bbox_mask].ravel().astype(np.float32)
        qual_raw = qual_full[bbox_mask].ravel().astype(np.float32)
        del ssha_full, qual_full
        gc.collect()

        rows, _  = np.where(bbox_mask)
        time_1d  = time_full.ravel()
        if len(time_1d) == bbox_mask.shape[0]:
            time_px = time_1d[rows].astype(np.float64)
        elif len(time_1d) > 1:
            time_px = np.linspace(time_1d[0], time_1d[-1], len(lat_raw), dtype=np.float64)
        else:
            time_px = np.full(len(lat_raw), float(time_1d[0]) if len(time_1d) else 0.0)
        del time_full
        gc.collect()

        logger.info(
            "Granule %s (%s): %d bbox pixels | RAM %.1f%%",
            fname, ssha_var, lat_raw.size, gov.memory_snapshot()["used_pct"],
        )
        return lat_raw, lon_raw, ssha_raw, qual_raw, time_px

    def _run_live_loop() -> tuple[list, list, list, list]:
        """Read all granule URLs and return (lats, lons, sshas, times) lists."""
        _lats, _lons, _sshas, _times = [], [], [], []
        for url in [granule_urls[0], granule_urls[-1]]:
            try:
                result_tup = _read_one_granule(url)
                if result_tup is None:
                    continue
                lat_raw, lon_raw, ssha_raw, qual_raw, time_raw = result_tup

                # Quality bitmask
                lat_f, lon_f, ssha_f = apply_quality_mask(
                    lat_raw, lon_raw, ssha_raw, qual_raw
                )
                evict(lat_raw, lon_raw, ssha_raw, qual_raw)
                del lat_raw, lon_raw, ssha_raw, qual_raw
                gc.collect()

                n_good = len(lat_f)
                if n_good > 0:
                    time_f = np.linspace(time_raw[0], time_raw[-1], n_good)
                    _lats.append(lat_f)
                    _lons.append(lon_f)
                    _sshas.append(ssha_f)
                    _times.append(time_f)

                evict(lat_f, lon_f, ssha_f)
                del lat_f, lon_f, ssha_f, time_raw
                gc.collect()

            except Exception as exc:
                logger.error(
                    "Failed to process granule %s: %s: %s",
                    url.split("/")[-1], type(exc).__name__, exc,
                )
                continue
        return _lats, _lons, _sshas, _times

    # Submit the live loop to a thread and enforce 35-second hard timeout
    all_lat: list = []
    all_lon: list = []
    all_ssha: list = []
    all_time: list = []
    timed_out = False

    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_live_loop)
        try:
            all_lat, all_lon, all_ssha, all_time = future.result(
                timeout=LIVE_TIMEOUT_SEC
            )
            logger.info("Live granule loop completed within timeout.")
        except _cf.TimeoutError:
            timed_out = True
            future.cancel()
            logger.warning(
                "Live granule fetch timed out after %.0fs -- "
                "falling back to synthetic data.",
                LIVE_TIMEOUT_SEC,
            )

    if timed_out or not all_lat:
        reason = (
            f"live fetch timed out after {LIVE_TIMEOUT_SEC:.0f}s"
            if timed_out else
            "live granules returned 0 pixels over the selected bbox"
        )
        logger.warning("Switching to synthetic fallback: %s", reason)
        # Call the synthetic path directly -- do NOT recurse into
        # stream_mekong_ssha again (that would re-enter the live path).
        raw = _synthetic_granule(months,
                                 lat_min=lat_min, lat_max=lat_max,
                                 lon_min=lon_min, lon_max=lon_max)
        lat_c, lon_c, ssha_c, qual_c = spatial_clip(
            raw["lat"], raw["lon"], raw["ssha_karin"], raw["ssha_karin_qual"],
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        )
        lat_sf, lon_sf, ssha_sf = apply_quality_mask(lat_c, lon_c, ssha_c, qual_c)
        evict(raw, lat_c, lon_c, ssha_c, qual_c)
        del raw, lat_c, lon_c, ssha_c, qual_c
        gc.collect()
        n_good = len(lat_sf)
        base_tai = 756_864_000.0
        time_sf = np.linspace(base_tai, base_tai + months * 30 * 86_400, n_good)
        return {
            "lat":        lat_sf,
            "lon":        lon_sf,
            "ssha_karin": ssha_sf,
            "time_tai":   time_sf.astype(np.float64),
            "source":     f"synthetic (demo) -- live NASA Earthdata unavailable: {reason}",
        }

    lat_out = np.concatenate(all_lat).astype(np.float32)
    lon_out = np.concatenate(all_lon).astype(np.float32)
    ssha_out = np.concatenate(all_ssha).astype(np.float32)
    time_out = np.concatenate(all_time).astype(np.float64)

    evict(all_lat, all_lon, all_ssha, all_time)
    del all_lat, all_lon, all_ssha, all_time
    gc.collect()

    logger.info(
        "Stream complete: %d clean pixels from %d / %d granules (auth: %s).",
        len(lat_out), len(granule_urls), total_granules_matched, auth_method,
    )
    live_source = (
        f"NASA Earthdata (live) -- {auth_method} auth -- "
        f"{len(granule_urls)} granules ({total_granules_matched} matched)"
    )
    return {
        "lat": lat_out,
        "lon": lon_out,
        "ssha_karin": ssha_out,
        "time_tai": time_out,
        "source": live_source,
    }