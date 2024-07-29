import pandas as pd
from config import (
    PROFILE_COLUMN_NAMES,
    COOLER_CONDITION_MAP,
    VALVE_CONDITION_MAP,
    PUMP_LEAKAGE_MAP,
    ACCUMULATOR_PRESSURE_MAP,
    STABLE_FLAG_MAP,
)


def process_data(profile_path, fs1_path, ps2_path):
    # Load data
    profile = pd.read_csv(profile_path, sep="\t", header=None)
    fs1 = pd.read_csv(fs1_path, sep="\t", header=None)
    ps2 = pd.read_csv(ps2_path, sep="\t", header=None)

    # Add column names to profile data
    profile.columns = PROFILE_COLUMN_NAMES

    # Normalize profile data
    profile["cooler_condition_normalized"] = profile["cooler_condition"].map(
        COOLER_CONDITION_MAP
    )
    profile["valve_condition_normalized"] = profile["valve_condition"].map(
        VALVE_CONDITION_MAP
    )
    profile["pump_leakage_normalized"] = profile["pump_leakage"].map(PUMP_LEAKAGE_MAP)
    profile["accumulator_pressure_normalized"] = profile["accumulator_pressure"].map(
        ACCUMULATOR_PRESSURE_MAP
    )
    profile["stable_flag_normalized"] = profile["stable_flag"].map(STABLE_FLAG_MAP)

    profile_normalized = profile[
        [
            "cooler_condition_normalized",
            "valve_condition_normalized",
            "pump_leakage_normalized",
            "accumulator_pressure_normalized",
            "stable_flag_normalized",
        ]
    ]

    # Normalize fs1 data
    fs1_normalized = (fs1 - fs1.min()) / (fs1.max() - fs1.min())

    # Normalize ps2 data
    ps2_normalized = (ps2 - ps2.min()) / (ps2.max() - ps2.min())

    # Concatenate all data
    data = pd.concat([ps2_normalized, fs1_normalized, profile_normalized], axis=1)

    return data, fs1, ps2
