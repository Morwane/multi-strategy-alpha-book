"""Load the component (sleeve) return streams.

The sleeves are produced by two standalone projects:
  - energy-spreads-statarb   → `energy_statarb` (crack 3:2:1 + Brent-WTI book)
  - vix-vol-carry            → `vix_carry`      (gated VIX vol-carry)

Their daily returns are exported to data/components/components.csv. This project
is the ALLOCATION layer that combines them.
"""
from pathlib import Path
import pandas as pd


def load_components(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "components" / "components.csv",
                     parse_dates=["Date"], index_col="Date")
    return df.dropna()
