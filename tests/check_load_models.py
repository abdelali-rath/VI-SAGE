#!/usr/bin/env python3
"""
Small utility script to sanity‑check that the different model wrappers
can be constructed and their checkpoints loaded without crashing.

Intended for manual execution during development, not as a formal test.
"""

import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path so that `src` imports work when the file
# is executed directly via `python tests/check_load_models.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def try_load(name, ctor, *args, **kwargs):
    """
    Helper to instantiate a model wrapper and report success/failure.

    Args:
        name: Human‑readable name to show in the log output.
        ctor: Constructor or callable used to create the object.
        *args, **kwargs: Arguments forwarded to the constructor.
    """
    print(f"\n--- Testing {name} ---")
    try:
        obj = ctor(*args, **kwargs)
        print(f"OK: {name} loaded: {type(obj)}")
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        traceback.print_exc()


def main():
    """
    Run a set of load attempts for the age / gender / ethnicity wrappers.

    Note: You may need to adjust the import paths if files are moved
    or renamed in the project structure.
    """
    from src.models import age_model as age_mod
    from src.models import gender_model as gender_mod
    from src.models import ethnicity_model as eth_mod

    # checkpoints to try
    try_load(
        "Age (age_model.pt)",
        age_mod.AgeInference,
        "checkpoints/age_model.pt",
        "cpu",
        True,
    )
    try_load(
        "Age (utk_age_model.pt)",
        age_mod.AgeInference,
        "checkpoints/utk_age_model.pt",
        "cpu",
        True,
    )

    try_load(
        "Gender (gender_model.GenderInference)",
        gender_mod.GenderInference,
        "checkpoints/utk_gender_model.pt",
        "cpu",
    )

    try_load(
        "Ethnicity (ethnicity_model)",
        eth_mod.EthnicityInference,
        "checkpoints/utk_ethnicity_model.pt",
        "cpu",
    )


if __name__ == "__main__":
    main()
