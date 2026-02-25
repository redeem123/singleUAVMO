from uav_benchmark.algorithms.momfea import run_momfea, run_momfea2
from uav_benchmark.algorithms.mopso import run_mopso
from uav_benchmark.algorithms.smpso import run_smpso
from uav_benchmark.algorithms.nmopso import run_nmopso
from uav_benchmark.algorithms.nsga2 import run_nsga2
from uav_benchmark.algorithms.nsga3 import run_nsga3
from uav_benchmark.algorithms.moead import run_moead
from uav_benchmark.algorithms.spea2 import run_spea2
from uav_benchmark.algorithms.mfo_spea2 import run_mfo_spea2
from uav_benchmark.algorithms.cmosma import run_cmosma
from uav_benchmark.algorithms.moqgwo import (
    run_multi_moqgwo,
    run_multi_moqgwo_no_attention,
    run_multi_moqgwo_no_atlas,
    run_multi_moqgwo_standard_gwo,
)
from uav_benchmark.algorithms.apex_shade import run_multi_apex_shade
from uav_benchmark.algorithms.tskac_nsga2 import run_tskac_nsga2

__all__ = [
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
    "run_cmosma",
    "run_multi_moqgwo",
    "run_multi_moqgwo_no_attention",
    "run_multi_moqgwo_no_atlas",
    "run_multi_moqgwo_standard_gwo",
    "run_multi_apex_shade",
    "run_tskac_nsga2",
]
