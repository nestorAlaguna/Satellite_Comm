# simulator.py 
from mbse_model import Satellite, GroundStation, OpticalLink, AtmosphericCondition, Requirement
from skyfield.api import load, wgs84, EarthSatellite
from data_manager import SatelliteDataManager 
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta

class OrbitSimulator:
    def __init__(self):
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.data_manager = SatelliteDataManager()  # Data manager
        
    def load_real_satellites(self) -> List[Satellite]:
        """Load real satellites using our data manager"""
        print("Loading real satellite data...")
        tle_data = self.data_manager.get_tle_data()
        
        real_satellites = []
        for name, line1, line2 in tle_data[:5]:  # Just use first 5 for demo
            # Create Skyfield satellite object
            skyfield_sat = EarthSatellite(line1, line2, name, self.ts)
            
            # Create our MBSE Satellite object with real data
            sat = Satellite(
                id=name,
                position=np.array([0, 0, 0]),  # Will be updated in real-time
                transmit_power_watts=5.0,
                antenna_gain=100.0,
                skyfield_object=skyfield_sat  # Store the real orbital object
            )
            real_satellites.append(sat)
        
        print(f"Loaded {len(real_satellites)} real satellites")
        return real_satellites
    
    def calculate_satellite_visibility(self, ground_station: GroundStation, 
                                    satellite: Satellite, current_time) -> Tuple[float, float]:
        """Calculate REAL elevation and distance using Skyfield"""
        
        if hasattr(satellite, 'skyfield_object'):
            # Convert our ground station location to Skyfield format
            gs_lat, gs_lon, gs_alt = ground_station.location
            ground_station_point = wgs84.latlon(gs_lat, gs_lon, gs_alt)
            
            # Calculate the difference vector
            difference = satellite.skyfield_object - ground_station_point
            topocentric = difference.at(current_time)
            
            # Get elevation and distance
            elevation_deg = topocentric.altaz()[0].degrees
            distance_km = topocentric.distance().km
            distance_m = distance_km * 1000
            
            return elevation_deg, distance_m
        else:
            # Fallback for simulated satellites
            return 25.0, 500000.0



class DigitalTwinEngine:
    """
    The main engine that runs the digital twin simulation.
    This coordinates all the system blocks and runs the simulation over time.
    """
    def __init__(self):
        self.orbit_simulator = OrbitSimulator()
        self.ground_stations: List[GroundStation] = []
        self.satellites: List[Satellite] = []
        self.requirements: List[Requirement] = []
        
    def add_ground_station(self, ground_station: GroundStation):
        """Add a ground station to the simulation."""
        self.ground_stations.append(ground_station)
        
    def add_satellite(self, satellite: Satellite):
        """Add a satellite to the simulation."""
        self.satellites.append(satellite)
        
    def add_requirement(self, requirement: Requirement):
        """Add a system requirement to verify."""
        self.requirements.append(requirement)
        
    def run_time_step(self, time_index: int, current_time):
        """
        Run one time step of the simulation.
        This is to check visibility, establish links, and verify requirements.
        """
        print(f"\n--- Time Step {time_index} at {current_time} ---")
        
        for ground_station in self.ground_stations:
            for satellite in self.satellites:
                # Calculate if the satellite is visible
                elevation, distance = self.orbit_simulator.calculate_satellite_visibility(
                    ground_station, satellite, current_time
                )
                
                # Check if we can track this satellite
                if ground_station.can_track(elevation):
                    print(f"✓ {ground_station.id} can track {satellite.id} (Elevation: {elevation:.1f}°)")
                    
                    # Create and analyze the communication link
                    link = OpticalLink(
                        satellite=satellite,
                        ground_station=ground_station,
                        atmospheric_condition=AtmosphericCondition.CLEAR,
                        distance=distance
                    )
                    
                    # Calculate the link budget
                    received_power, ber = link.calculate_link_budget()
                    
                    # Verify requirements for this link
                    self._verify_link_requirements(link, ber)
                    
                else:
                    print(f"✗ {ground_station.id} cannot track {satellite.id} (Elevation: {elevation:.1f}° < {ground_station.min_elevation}°)")
    
    def _verify_link_requirements(self, link: OpticalLink, ber: float):
        """Verify system requirements for a specific link."""
        # REQ-002: BER requirement
        ber_req = Requirement(
            id="REQ-002", 
            text="The link shall maintain a BER better than 1e-6"
        )
        ber_req.verify(ber < 1e-6)
        
        print(f"  Requirement Verification:")
        print(f"    {ber_req.id}: {'PASS' if ber_req.verified else 'FAIL'} (BER: {ber:.2e})")
        
        if ber_req.verified:
            print(f"     Link {link.satellite.id} -> {link.ground_station.id} is OPERATIONAL")
        else:
            print(f"     Link {link.satellite.id} -> {link.ground_station.id} is DEGRADED")

# Let's test our new simulation engine
if __name__ == "__main__":
    print("=== INITIALIZING SATELLITE COMMUNICATIONS DIGITAL TWIN ===")
    
    # 1. Create the digital twin engine
    digital_twin = DigitalTwinEngine()
    
    # 2. Add our ground station (Eindhoven)
    groundS_gs = GroundStation(
        id="EINDHOVEN",
        location=[51.4416, 5.4697, 0.0],
        aperture_diameter=1.0,
        min_elevation=5.0
    )
    digital_twin.add_ground_station(groundS_gs)
    
    # 3. Create and add a satellite
    satellite_1 = Satellite(
        id="ASTRA-1",
        position=np.array([7000000.0, 0.0, 0.0]),
        transmit_power_watts=5.0,
        antenna_gain=100.0
    )
    digital_twin.add_satellite(satellite_1)
    
    # 4. Run a simple simulation for 3 time steps
    print("\n=== STARTING SIMULATION ===")
    for time_step in range(3):
        # In a real simulation, we would use actual timestamps
        # For now, we'll use simple time steps
        digital_twin.run_time_step(time_step, f"2024-06-{10 + time_step} 12:00:00")
    
    print("\n=== SIMULATION COMPLETE ===")