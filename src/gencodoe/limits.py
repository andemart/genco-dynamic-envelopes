"""Voltage limits: the single source of truth for every feasibility check.

Limits are defined in volts (customer phase-to-neutral) and converted to per
unit on the model base. The model base is the ENWL1 transformer's no-load
secondary voltage: 416 V line-to-line, i.e. 240.18 V phase-to-neutral; both the
balanced model and the four-wire OpenDSS checker express customer voltage on
this base.

Jurisdictions:
  uk_esqcr      216.2-253.0 V (230 V -6% / +10%), the statutory range ENWL
                applies on its network; the primary criterion of this study
  en50160_band  207.0-253.0 V (230 V +/-10%), the EN 50160 voltage band used as
                a Belgian-practice sensitivity (only the band, not the standard's
                measurement-duration and aggregation rules)

Select with the environment variable GENCODOE_VOLTAGE_LIMITS (default uk_esqcr),
read once at import so that every module uses the same values in one process.

History: until 15 Oct 2026 the study applied 0.94 / 1.10 pu on this base
(225.8 / 264.2 V) instead of converting the absolute limits; see
docs/preregistration_amendment_1.md.
"""

from __future__ import annotations

import math
import os

V_LL_BASE = 416.0
V_BASE = V_LL_BASE / math.sqrt(3)          # 240.18 V phase-to-neutral = 1 pu

JURISDICTIONS = {
    "uk_esqcr": (216.2, 253.0),
    "en50160_band": (207.0, 253.0),
}

JURISDICTION = os.environ.get("GENCODOE_VOLTAGE_LIMITS", "uk_esqcr")
if JURISDICTION not in JURISDICTIONS:
    raise ValueError(f"GENCODOE_VOLTAGE_LIMITS={JURISDICTION!r}; choose from {sorted(JURISDICTIONS)}")

VMIN_V, VMAX_V = JURISDICTIONS[JURISDICTION]
VMIN_PU = VMIN_V / V_BASE                  # 0.9002 (uk_esqcr), 0.8619 (en50160_band)
VMAX_PU = VMAX_V / V_BASE                  # 1.0534


def record() -> dict:
    """Metadata stored in every result file."""
    return {"jurisdiction": JURISDICTION, "vbase_volts": V_BASE, "vmin_volts": VMIN_V, "vmax_volts": VMAX_V,
            "vmin_pu": VMIN_PU, "vmax_pu": VMAX_PU}
