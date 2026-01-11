#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def try_load(name, ctor, *args, **kwargs):
    print(f"\n--- Testing {name} ---")
    try:
        obj = ctor(*args, **kwargs)
        print(f"OK: {name} loaded: {type(obj)}")
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        traceback.print_exc()


def main():
    from src import age_model as age_mod
    from src import gender_model as gender_mod
    from src import gender_infer as gender_inf_mod
    from src import ethnicity_model as eth_mod

    # checkpoints to try
    try_load("Age (age_model.pt)", age_mod.AgeInference, "checkpoints/age_model.pt", "cpu", True)
    try_load("Age (utk_age_mobilenet.pt)", age_mod.AgeInference, "checkpoints/utk_age_mobilenet.pt", "cpu", True)

    try_load("Gender (gender_model.GenderInference)", gender_mod.GenderInference, "checkpoints/utk_gender_mobilenet.pt", "cpu")
    try_load("Gender (gender_infer.GenderInference)", gender_inf_mod.GenderInference, "checkpoints/utk_gender_mobilenet.pt")

    try_load("Ethnicity (ethnicity_model)", eth_mod.EthnicityInference, "checkpoints/ethnicity_model.pt", "cpu")


if __name__ == '__main__':
    main()
