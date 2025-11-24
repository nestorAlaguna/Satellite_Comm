# final_test.py
from run_optimized_digital_twin import OptimizedDigitalTwin
from visualization import SatelliteVisualizer

def final_verification():
    """Final test to verify operational links"""
    print(" FINAL VERIFICATION TEST")
    print("="*60)
    
    # Create optimized twin
    twin = OptimizedDigitalTwin()
    twin.add_enhanced_ground_stations()
    twin.create_optimized_satellites()
    
    # Run a quick simulation
    print("\n Running quick simulation...")
    results = twin.run_simulation(
        duration_hours=0.5,  # 30 minutes
        time_step_minutes=10  # 10-minute steps
    )
    
    # Analyze results
    if results:
        operational_links = [r for r in results if r['link_operational']]
        operational_percentage = (len(operational_links) / len(results)) * 100
        
        print(f"\n FINAL RESULTS:")
        print(f"   Total communication opportunities: {len(results)}")
        print(f"   Operational links: {len(operational_links)}")
        print(f"   System reliability: {operational_percentage:.1f}%")
        
        if operational_links:
            print(f"    SUCCESS: System has operational links!")
            
            # Show best link
            best_link = min(operational_links, key=lambda x: x['ber'])
            print(f"\n BEST LINK PERFORMANCE:")
            print(f"   {best_link['satellite']} → {best_link['ground_station']}")
            print(f"   BER: {best_link['ber']:.2e}")
            print(f"   Elevation: {best_link['elevation']:.1f}°")
            print(f"   Distance: {best_link['distance_km']:.1f} km")
            
            # Generate visualizations
            print(f"\n Generating final visualizations...")
            viz = SatelliteVisualizer()
            viz.plot_satellite_constellation(twin.satellites, twin.ground_stations)
            
            if len(results) >= 4:  # Only create dashboard if we have enough data
                viz.create_performance_dashboard(results)
                print(f" Performance dashboard created with operational data!")
            
        else:
            print(f"    Still no operational links - need further debugging")
    else:
        print(f"    No communication opportunities found")

if __name__ == "__main__":
    final_verification()