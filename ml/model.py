"""
Model registry — load, save, and version ICT signal models.

Model files are stored in models/<version>/bundle.pkl
Each bundle is a dict:
    model        : fitted classifier
    scaler       : fitted StandardScaler
    model_name   : "random_forest" | "xgboost"
    threshold    : float (default 0.60)
    feature_names: list[str]
    metrics      : eval metrics dict
    cv_results   : cross-validation scores
    trained_at   : ISO timestamp
    n_train      : number of training samples
    version      : int
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).resolve().parent.parent / "models"
BUNDLE_NAME  = "bundle.pkl"
METADATA_NAME = "metadata.json"
LATEST_LINK  = MODELS_DIR / "latest"    # file that stores current version int


# ─────────────────────────────────────────────────────────────────────────────
# Versioning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_version() -> int:
    """Return the next version number (highest existing + 1)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return (max(existing) + 1) if existing else 1


def _current_version() -> Optional[int]:
    """Read the 'latest' pointer file; return None if no model exists."""
    if LATEST_LINK.exists():
        try:
            return int(LATEST_LINK.read_text().strip())
        except (ValueError, OSError):
            pass
    # Fall back to highest version directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return max(existing) if existing else None


def _version_dir(version: int) -> Path:
    return MODELS_DIR / str(version)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(bundle: dict, make_latest: bool = True) -> int:
    """
    Save a model bundle to disk with automatic versioning.

    Returns the version number assigned.
    """
    version = _next_version()
    vdir    = _version_dir(version)
    vdir.mkdir(parents=True, exist_ok=True)

    bundle["version"]    = version
    bundle["trained_at"] = datetime.now(timezone.utc).isoformat()

    # Pickle
    with open(vdir / BUNDLE_NAME, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Human-readable metadata (no model object)
    meta = {k: v for k, v in bundle.items()
            if k not in ("model", "scaler", "feature_names")}
    with open(vdir / METADATA_NAME, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if make_latest:
        LATEST_LINK.write_text(str(version))

    logger.info("Model v%d saved to %s", version, vdir)
    return version


def load_model(version: Optional[int] = None) -> Optional[dict]:
    """
    Load a model bundle from disk.

    ``version=None`` loads the current latest version.
    Returns None when no model exists yet.
    """
    if version is None:
        version = _current_version()
    if version is None:
        return None

    bundle_path = _version_dir(version) / BUNDLE_NAME
    if not bundle_path.exists():
        logger.warning("Model v%d not found at %s", version, bundle_path)
        return None

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    logger.debug("Loaded model v%d (%s)", version, bundle.get("model_name"))
    return bundle


def list_versions() -> list[dict]:
    """Return metadata for all saved model versions."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(MODELS_DIR.iterdir()):
        if p.is_dir() and p.name.isdigit():
            meta_path = p / METADATA_NAME
            if meta_path.exists():
                with open(meta_path) as f:
                    result.append(json.load(f))
    return result


def promote_version(version: int) -> None:
    """Set a specific version as the latest (rollback support)."""
    if not (_version_dir(version) / BUNDLE_NAME).exists():
        raise FileNotFoundError(f"Model version {version} does not exist")
    LATEST_LINK.write_text(str(version))
    logger.info("Model latest pointer set to v%d", version)
