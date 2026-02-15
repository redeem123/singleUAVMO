"""Python implementation of the UAV path-planning benchmark."""

from uav_benchmark.bootstrap import bootstrap_homebrew_science_stack

bootstrap_homebrew_science_stack()

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct

__all__ = ["BenchmarkParams", "load_terrain_struct"]
