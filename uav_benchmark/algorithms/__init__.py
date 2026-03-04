from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from uav_benchmark.algorithms.momfea import run_momfea, run_momfea2
from uav_benchmark.algorithms.mopso import run_mopso
from uav_benchmark.algorithms.smpso import run_smpso
from uav_benchmark.algorithms.nmopso import run_nmopso
from uav_benchmark.algorithms.nsga2 import run_nsga2
from uav_benchmark.algorithms.nsga3 import run_nsga3
from uav_benchmark.algorithms.moead import run_moead
from uav_benchmark.algorithms.spea2 import run_spea2
from uav_benchmark.algorithms.mfo_spea2 import run_mfo_spea2
from uav_benchmark.algorithms.gcnmoea import run_gcnmoea
from uav_benchmark.algorithms.cmosma import run_cmosma
from uav_benchmark.algorithms.mogwo import (
    run_fleet_mogwo,
    run_fleet_mogwo_no_attention,
    run_fleet_mogwo_standard_gwo,  # kept for backward-compat; not in REGISTRY
)
from uav_benchmark.algorithms.apex_shade import run_fleet_apex_shade
from uav_benchmark.algorithms.tskac_nsga2 import run_tskac_nsga2

if TYPE_CHECKING:
    from uav_benchmark.algorithms.shared.interface import UAVAlgorithm
    from uav_benchmark.config import BenchmarkParams

# Central registry for all benchmark algorithms.
# Each entry must satisfy the UAVAlgorithm protocol.
REGISTRY: dict[str, Callable[[dict[str, Any], BenchmarkParams], np.ndarray]] = {
    "NMOPSO": run_nmopso,
    "MOPSO": run_mopso,
    "SMPSO": run_smpso,
    "NSGA-II": run_nsga2,
    "NSGA-III": run_nsga3,
    "MOEAD": run_moead,
    "SPEA2": run_spea2,
    "MFO-SPEA2": run_mfo_spea2,
    "GCNMOEA": run_gcnmoea,
    "CMOSMA": run_cmosma,
    "MO-MFEA": run_momfea,
    "MO-MFEA-II": run_momfea2,
    "MOGWO": run_fleet_mogwo,
    "MOGWO-NO-ATTENTION": run_fleet_mogwo_no_attention,
    "APEX-SHADE": run_fleet_apex_shade,
    "TSKAC-NSGA-II": run_tskac_nsga2,
}

__all__ = [
    "REGISTRY",
    "run_momfea",
    "run_momfea2",
    "run_mopso",
    "run_smpso",
    "run_nmopso",
    "run_nsga2",
    "run_nsga3",
    "run_moead",
    "run_spea2",
    "run_mfo_spea2",
    "run_gcnmoea",
    "run_cmosma",
    "run_fleet_mogwo",
    "run_fleet_mogwo_no_attention",
    "run_fleet_mogwo_standard_gwo",
    "run_fleet_apex_shade",
    "run_tskac_nsga2",
]
