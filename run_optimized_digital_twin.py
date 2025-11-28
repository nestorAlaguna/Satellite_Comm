# run_optimized_digital_twin.py
from run_digital_twin import DigitalTwinEngine, OrbitSimulator
from mbse_model import Satellite, GroundStation, OpticalLink, AtmosphericCondition, Requirement
import numpy as np

class OptimizedDigitalTwin(DigitalTwinEngine):
    """
    Optimized version with realistic satellite parameters that meet requirements.
    """
    
    def __init__(self):
        super().__init__()
        
    def create_optimized_satellites(self):
        """Create satellites with realistic optical communication parameters"""
        print("\n CREATING OPTIMIZED SATELLITE CONSTELLATION")
        
        # Realistic optical communication satellite parameters
        optimized_satellites = [
            Satellite(
                id="OPTICAL-SAT-1",
                position=np.array([7000000, 0, 0]),
                transmit_power_watts=10.0,  # Increased from 5W
                wavelength=1550e-9,
                antenna_gain=1000.0,  # More realistic optical antenna gain
                description="High-power optical comm satellite"
            ),
            Satellite(
                id="OPTICAL-SAT-2", 
                position=np.array([0, 7000000, 0]),
                transmit_power_watts=8.0,
                wavelength=1550e-9,
                antenna_gain=800.0,
                description="Medium-power optical satellite"
            ),
            Satellite(
                id="OPTICAL-SAT-3",
                position=np.array([0, 0, 7000000]),
                transmit_power_watts=12.0,
                wavelength=1550e-9, 
                antenna_gain=1200.0,
                description="High-gain optical satellite"
            )
        ]
        
        for sat in optimized_satellites:
            self.add_satellite(sat)
            print(f"    {sat.id}: {sat.transmit_power_watts}W, Gain: {sat.antenna_gain}")
    
    def add_enhanced_ground_stations(self):
        """Add ground stations with better optical equipment"""
        print("\n ENHANCED GROUND STATIONS")
        
        enhanced_stations = [
            GroundStation(
                id="PLACE-EINDHOVEN-OPTICAL",
                location=[51.4416, 5.4697, 0.0],
                aperture_diameter=2.0,  # Larger telescope - 2 meter diameter
                min_elevation=5.0,
                description="Place in Eindhoven with 2m optical telescope"
            ),
            GroundStation(
                id="ESA-MADRID-OPTICAL", 
                location=[40.4310, -3.6780, 650.0],
                aperture_diameter=3.0,  # Even larger telescope
                min_elevation=10.0,
                description="ESA Madrid with 3m optical telescope"
            ),
            GroundStation(
                id="MIT-HAYSTACK-OPTICAL",
                location=[42.6234, -71.4912, 100.0], 
                aperture_diameter=1.5,
                min_elevation=8.0,
                description="MIT Haystack Observatory"
            )
        ]
        
        for gs in enhanced_stations:
            self.add_ground_station(gs)
            print(f"    {gs.id}: {gs.aperture_diameter}m telescope")
    
    def run_optimized_simulation(self):
        """Run simulation with optimized parameters and better analysis"""
        print("\n" + "="*70)
        print(" OPTIMIZED DIGITAL TWIN SIMULATION")
        print("   Realistic Optical Satellite Communications")
        print("=" + "="*70)
        
        # Add optimized components
        self.add_enhanced_ground_stations()
        self.create_optimized_satellites()
        
        # Load some real data from satellites for comparison
        real_sats = self.orbit_simulator.load_real_satellites()
        for sat in real_sats[:2]:  # Just 2 real ones for demo
            self.add_satellite(sat)
        
        # Run simulation
        results = self.run_simulation(
            duration_hours=0.5,  # 30 minutes for demo
            time_step_minutes=5
        )
        
        # Enhanced analysis
        self._analyze_system_performance(results)
        
        return results
    
    def _analyze_system_performance(self, results):
        """Provide detailed system performance analysis"""
        print("\n" + "="*70)
        print(" ENHANCED SYSTEM PERFORMANCE ANALYSIS")
        print("=" + "="*70)
        
        if not results:
            print(" No communication opportunities found")
            return
        
        # Categorize results
        operational_links = [r for r in results if r['link_operational']]
        degraded_links = [r for r in results if not r['link_operational']]
        
        total_opportunities = len(results)
        operational_percentage = (len(operational_links) / total_opportunities * 100) if total_opportunities > 0 else 0
        
        print(f" SYSTEM RELIABILITY: {operational_percentage:.1f}%")
        print(f"    Operational links: {len(operational_links)}")
        print(f"    Degraded links: {len(degraded_links)}")
        print(f"    Total opportunities: {total_opportunities}")
        
        # Best and worst performing links
        if operational_links:
            best_link = min(operational_links, key=lambda x: x['ber'])
            print(f"\n BEST PERFORMING LINK:")
            print(f"   {best_link['satellite']} → {best_link['ground_station']}")
            print(f"   BER: {best_link['ber']:.2e} | Elevation: {best_link['elevation']:.1f}°")
        
        if degraded_links:
            worst_link = max(degraded_links, key=lambda x: x['ber'])
            print(f"\n WORST PERFORMING LINK (needs optimization):")
            print(f"   {worst_link['satellite']} → {worst_link['ground_station']}")
            print(f"   BER: {worst_link['ber']:.2e} | Elevation: {worst_link['elevation']:.1f}°")
        
        # Performance by ground station
        print(f"\n PERFORMANCE BY GROUND STATION:")
        for gs in self.ground_stations:
            gs_links = [r for r in results if r['ground_station'] == gs.id]
            if gs_links:
                gs_operational = [r for r in gs_links if r['link_operational']]
                gs_percentage = (len(gs_operational) / len(gs_links) * 100) if gs_links else 0
                print(f"   {gs.id}: {gs_percentage:.1f}% reliable "
                      f"({len(gs_operational)}/{len(gs_links)} links)")

def main():
    """Run the optimized digital twin"""
    print("=" + "="*70)
    print("  OPTIMIZED OPTICAL SATELLITE COMMUNICATIONS DIGITAL TWIN")
    print("   MBSE + Realistic Parameters + System Performance Analysis")
    print("=" + "="*70)
    
    # Create and run optimized twin
    optimized_twin = OptimizedDigitalTwin()
    results = optimized_twin.run_optimized_simulation()
    
    print("\n OPTIMIZED SIMULATION COMPLETE")
    return results

if __name__ == "__main__":
    main()