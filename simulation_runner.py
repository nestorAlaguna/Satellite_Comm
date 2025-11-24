# simulation_runner.py
from mbse_model import Satellite
import numpy as np

print("=== Starting the main simulation from simulation_runner.py ===")

# We are IMPORTING the Satellite class, not running mbse_model.py as the main script.
my_new_satellite = Satellite(
    id="IMPORTED-SAT",
    position=np.array([500000, 600000, 700000.0])
)

print(f"Successfully created {my_new_satellite.id} from another file!")