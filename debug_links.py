# debug_links.py
from mbse_model import Satellite, GroundStation, OpticalLink, AtmosphericCondition
import numpy as np

def debug_link_calculation():
    """Debug why links are always degraded"""
    print(" DEBUGGING LINK CALCULATIONS")
    print("="*60)
    
    # Create test components
    test_satellite = Satellite(
        id="DEBUG-SAT",
        position=np.array([7000000, 0, 0]),
        transmit_power_watts=10.0,
        wavelength=1550e-9,
        antenna_gain=1000.0
    )
    
    test_ground_station = GroundStation(
        id="DEBUG-GS",
        location=[51.4416, 5.4697, 0.0],
        aperture_diameter=2.0,
        min_elevation=5.0
    )
    
    # Test different distances
    test_distances = [500000, 1000000, 2000000]  # 500km, 1000km, 2000km
    
    for distance in test_distances:
        print(f"\n--- Testing distance: {distance/1000:.0f} km ---")
        
        link = OpticalLink(
            satellite=test_satellite,
            ground_station=test_ground_station,
            atmospheric_condition=AtmosphericCondition.CLEAR,
            distance=distance
        )
        
        # Run link budget calculation
        received_power, ber = link.calculate_link_budget()
        
        # Check requirements
        requirement_met = ber < 1e-6
        print(f" Results:")
        print(f"   Received Power: {received_power:.2e} W")
        print(f"   BER: {ber:.2e}")
        print(f"   Requirement (BER < 1e-6): {' MET' if requirement_met else ' FAILED'}")
        
        # Debug individual components
        print(f" Component Analysis:")
        print(f"   Transmit Power: {test_satellite.transmit_power_watts} W")
        print(f"   Satellite Gain: {test_satellite.antenna_gain}")
        print(f"   Wavelength: {test_satellite.wavelength:.2e} m")
        print(f"   Aperture Diameter: {test_ground_station.aperture_diameter} m")
        print(f"   Distance: {distance} m")

def analyze_ber_sensitivity():
    """Analyze how different parameters affect BER"""
    print("\n" + "="*60)
    print(" BER SENSITIVITY ANALYSIS")
    print("="*60)
    
    base_satellite = Satellite(
        id="SENSITIVITY-SAT",
        position=np.array([7000000, 0, 0]),
        transmit_power_watts=10.0,
        wavelength=1550e-9,
        antenna_gain=1000.0
    )
    
    base_ground_station = GroundStation(
        id="SENSITIVITY-GS", 
        location=[51.4416, 5.4697, 0.0],
        aperture_diameter=2.0,
        min_elevation=5.0
    )
    
    # Test different parameter combinations
    power_levels = [5, 10, 20]  # Watts
    antenna_gains = [500, 1000, 2000]
    apertures = [1.0, 2.0, 3.0]  # meters
    
    print("\nTesting different configurations (distance = 1000 km):")
    print("Pwr(W) | Gain | Apert(m) | BER       | Status")
    print("-" * 50)
    
    for power in power_levels:
        for gain in antenna_gains:
            for aperture in apertures:
                base_satellite.transmit_power_watts = power
                base_satellite.antenna_gain = gain
                base_ground_station.aperture_diameter = aperture
                
                link = OpticalLink(
                    satellite=base_satellite,
                    ground_station=base_ground_station,
                    atmospheric_condition=AtmosphericCondition.CLEAR,
                    distance=1000000  # 1000 km
                )
                
                received_power, ber = link.calculate_link_budget()
                status = " OPERATIONAL" if ber < 1e-6 else " DEGRADED"
                
                print(f"{power:6.1f} | {gain:4} | {aperture:8.1f} | {ber:.2e} | {status}")

if __name__ == "__main__":
    debug_link_calculation()
    analyze_ber_sensitivity()