import numpy as np
import os
import yaml
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.algorithms.mogwo import run_fleet_mogwo

print("Evaluating MOGWO to see why it broke...")
