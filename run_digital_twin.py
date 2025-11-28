# run_digital_twin.py
from mbse_model import Satellite, GroundStation, OpticalLink, AtmosphericCondition, Requirement
from skyfield.api import load, wgs84, EarthSatellite
from data_manager import SatelliteDataManager
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta
import time

class OrbitSimulator:
    def __init__(self):
        print("Initializing Orbit Simulator...")
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.data_manager = SatelliteDataManager()
        
    def load_real_satellites(self) -> List[Satellite]:
        """Load real satellites using our data manager"""
        print(" Loading real satellite data...")
        tle_data = self.data_manager.get_tle_data()
        
        real_satellites = []
        for name, line1, line2 in tle_data[:3]:  # Use first 3 for demo
            # Create Skyfield satellite object
            skyfield_sat = EarthSatellite(line1, line2, name, self.ts)
            
            # Create MBSE Satellite object with real data
            sat = Satellite(
                id=name[:20],  # Trim long names for display
                position=np.array([0, 0, 0]),  # Will be updated in real-time
                transmit_power_watts=5.0,
                antenna_gain=100.0,
                skyfield_object=skyfield_sat  # Store the real orbital object
            )
            real_satellites.append(sat)
            print(f"   Loaded: {name[:20]}...")
        
        print(f" Loaded {len(real_satellites)} real satellites")
        return real_satellites
    
    def calculate_satellite_visibility(self, ground_station: GroundStation, 
                                    satellite: Satellite, current_time) -> Tuple[float, float, np.ndarray]:
        """
        Calculate REAL elevation, distance, and position using Skyfield
        Returns: (elevation_degrees, distance_meters, position_array)
        """
        if hasattr(satellite, 'skyfield_object'):
            try:
                # Convert ground station location to Skyfield format
                gs_lat, gs_lon, gs_alt = ground_station.location
                ground_station_point = wgs84.latlon(gs_lat, gs_lon, gs_alt)
                
                # Calculate the difference vector
                difference = satellite.skyfield_object - ground_station_point
                topocentric = difference.at(current_time)
                
                # Get elevation, azimuth, and distance
                alt, az, distance_km = topocentric.altaz()
                elevation_deg = alt.degrees
                distance_m = distance_km * 1000
                
                # Get the actual satellite position in ECI coordinates (meters)
                geocentric = satellite.skyfield_object.at(current_time)
                position = geocentric.position.m
                
                return elevation_deg, distance_m, position
                
            except Exception as e:
                print(f"   Orbital calculation error for {satellite.id}: {e}")
                return 0.0, 10000000.0, np.array([0, 0, 0])
        else:
            # Fallback for simulated satellites
            return 25.0, 500000.0, np.array([7000000, 0, 0])

class DigitalTwinEngine:
    """
    The main engine that runs digital twin simulation with REAL orbital data.
    """
    def __init__(self):
        self.orbit_simulator = OrbitSimulator()
        self.ground_stations: List[GroundStation] = []
        self.satellites: List[Satellite] = []
        self.current_time = self.orbit_simulator.ts.now()
        
    def add_ground_station(self, ground_station: GroundStation):
        """Add a ground station to the simulation."""
        self.ground_stations.append(ground_station)
        
    def add_satellite(self, satellite: Satellite):
        """Add a satellite to the simulation."""
        self.satellites.append(satellite)
        
    def load_real_satellites(self):
        """Load real satellites from TLE data."""
        real_sats = self.orbit_simulator.load_real_satellites()
        self.satellites.extend(real_sats)
        
    def run_simulation(self, duration_hours: float = 2, time_step_minutes: float = 10):
        """
        Run the digital twin simulation with real orbital mechanics.
        """
        print(f"\n STARTING DIGITAL TWIN SIMULATION")
        print(f"   Duration: {duration_hours} hours")
        print(f"   Time Step: {time_step_minutes} minutes")
        print(f"   Start Time: {self.current_time.utc_strftime()}")
        print("=" * 60)
        
        start_time = self.current_time
        end_time = self.orbit_simulator.ts.utc(
            start_time.utc_datetime() + timedelta(hours=duration_hours)
        )
        
        time_step = timedelta(minutes=time_step_minutes)
        current_sim_time = start_time
        
        simulation_results = []
        
        while current_sim_time.tt < end_time.tt:
            # Convert Skyfield time to datetime for display
            current_dt = current_sim_time.utc_datetime()
            
            print(f"\n SIMULATION TIME: {current_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print("-" * 50)
            
            time_step_results = self._run_time_step(current_sim_time)
            simulation_results.extend(time_step_results)
            
            # Advance simulation time
            current_sim_time = self.orbit_simulator.ts.utc(
                current_sim_time.utc_datetime() + time_step
            )
        
        print(f"\n SIMULATION COMPLETE")
        print(f"   Analyzed {len(simulation_results)} communication opportunities")
        return simulation_results
    
    def _run_time_step(self, current_time):
        """Run one time step of the simulation."""
        time_step_results = []
        
        for ground_station in self.ground_stations:
            print(f"\n Ground Station: {ground_station.id}")
            print(f"   Location: {ground_station.location[0]:.4f}°, {ground_station.location[1]:.4f}°")
            
            visible_count = 0
            for satellite in self.satellites:
                # Calculate REAL satellite position and visibility
                elevation, distance, position = self.orbit_simulator.calculate_satellite_visibility(
                    ground_station, satellite, current_time
                )
                
                # Update satellite position with real data
                satellite.position = position
                
                # Check if it is possible to track this satellite
                if ground_station.can_track(elevation):
                    visible_count += 1
                    print(f"\n    TRACKING {satellite.id}")
                    print(f"      Elevation: {elevation:6.1f}° | Distance: {distance/1000:8.1f} km")
                    
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
                    req_002 = Requirement(
                        id="REQ-002", 
                        text="BER < 1e-6"
                    )
                    req_002.verify(ber < 1e-6)
                    
                    print(f"      Link Quality: {' OPERATIONAL' if req_002.verified else ' DEGRADED'}")
                    print(f"      BER: {ber:.2e} | Rx Power: {received_power:.2e} W")
                    
                    # Store results for analysis
                    result = {
                        'timestamp': current_time.utc_datetime(),
                        'ground_station': ground_station.id,
                        'satellite': satellite.id,
                        'elevation': elevation,
                        'distance_km': distance / 1000,
                        'ber': ber,
                        'received_power': received_power,
                        'link_operational': req_002.verified
                    }
                    time_step_results.append(result)
                    
                else:
                    if elevation > -5:  # Only show satellites that are somewhat close
                        print(f"    Below horizon: {satellite.id} (Elevation: {elevation:5.1f}°)")
            
            if visible_count == 0:
                print("    No satellites visible above minimum elevation")
        
        return time_step_results

def main():
    """Main function to run the complete digital twin simulation."""
    print("=" * 70)
    print("  OPTICAL SATELLITE COMMUNICATIONS DIGITAL TWIN")
    print("   MBSE + Real Orbital Mechanics + System Validation")
    print("=" * 70)
    
    # 1. Create the digital twin engine
    digital_twin = DigitalTwinEngine()
    
    # 2. Add ground stations
    print("\n INITIALIZING GROUND STATIONS")
    
    phi_gs = GroundStation(
        id="EINDHOVEN",
        location=[51.4416, 5.4697, 0.0],  # Eindhoven
        aperture_diameter=1.0,
        min_elevation=5.0
    )
    digital_twin.add_ground_station(phi_gs)
    print(f"    {phi_gs.id}")
    
    # Add another ground station for demonstration
    esa_gs = GroundStation(
        id="ESA-MADRID",
        location=[40.4310, -3.6780, 650.0],  # ESA Madrid space station
        aperture_diameter=2.0,  # Larger telescope
        min_elevation=10.0  # Higher requirement
    )
    digital_twin.add_ground_station(esa_gs)
    print(f"    {esa_gs.id}")
    
    # 3. Load real data from satellites
    print("\n  LOADING SATELLITE CONSTELLATION")
    digital_twin.load_real_satellites()
    
    # 4. Run the simulation
    print("\n STARTING REAL-TIME SIMULATION")
    results = digital_twin.run_simulation(
        duration_hours=1,  # Simulate 1 hour
        time_step_minutes=5  # 5-minute steps
    )
    
    # 5. Simulation Summary
    print("\n" + "=" * 70)
    print(" SIMULATION SUMMARY")
    print("=" * 70)
    
    if results:
        operational_links = [r for r in results if r['link_operational']]
        total_opportunities = len(results)
        operational_percentage = (len(operational_links) / total_opportunities * 100) if total_opportunities > 0 else 0
        
        print(f"Total communication opportunities: {total_opportunities}")
        print(f"Operational links (BER < 1e-6): {len(operational_links)}")
        print(f"System reliability: {operational_percentage:.1f}%")
        
        # Show best and worst links
        if operational_links:
            best_link = min(operational_links, key=lambda x: x['ber'])
            print(f"\n BEST LINK: {best_link['satellite']} -> {best_link['ground_station']}")
            print(f"   BER: {best_link['ber']:.2e} | Elevation: {best_link['elevation']:.1f}°")
    else:
        print("No communication opportunities found during simulation period.")
    
    print("\n DIGITAL TWIN EXECUTION COMPLETE")

if __name__ == "__main__":
    main()