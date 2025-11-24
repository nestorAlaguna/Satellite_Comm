# showcase_project.py
from run_optimized_digital_twin import OptimizedDigitalTwin
from visualization import SatelliteVisualizer
from ml_predictor import FadePredictor
import time

def showcase_complete_project():
    """Run the complete project showcase """
    print("="*80)
    print(" OPTICAL SATELLITE COMMUNICATIONS DIGITAL TWIN - COMPLETE SHOWCASE")
    print("   MBSE + ML + Real-time Simulation + Professional Visualizations")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Initialize ML predictor
    print("\n1.  INITIALIZING MACHINE LEARNING MODULE")
    ml_predictor = FadePredictor()
    X, y = ml_predictor.generate_training_data()
    mae, r2 = ml_predictor.train_random_forest(X, y)
    print(f"    ML Model trained with R² = {r2:.3f}")
    
    # 2. Run optimized simulation
    print("\n2.   RUNNING OPTIMIZED DIGITAL TWIN SIMULATION")
    optimized_twin = OptimizedDigitalTwin()
    results = optimized_twin.run_optimized_simulation()
    
    # 3. Generate professional visualizations
    print("\n3.  CREATING PROFESSIONAL VISUALIZATIONS")
    viz = SatelliteVisualizer()
    
    # Constellation visualization
    viz.plot_satellite_constellation(optimized_twin.satellites, optimized_twin.ground_stations)
    
    # Performance dashboard (if we have results)
    if results:
        viz.create_performance_dashboard(results)
    
    # 4. Project summary
    execution_time = time.time() - start_time
    print("\n" + "="*80)
    print(" PROJECT SHOWCASE COMPLETE!")
    print("="*80)
    print(f" EXECUTION SUMMARY:")
    print(f"   • Total execution time: {execution_time:.1f} seconds")
    print(f"   • Satellite models: {len(optimized_twin.satellites)}")
    print(f"   • Ground stations: {len(optimized_twin.ground_stations)}") 
    print(f"   • ML model accuracy: R² = {r2:.3f}")
    if results:
        operational_links = sum(1 for r in results if r['link_operational'])
        print(f"   • System reliability: {operational_links}/{len(results)} links operational")
    
    print(f"\n FILES GENERATED:")
    print(f"   • satellite_constellation.png - System architecture visualization")
    print(f"   • performance_dashboard.png - Performance analysis")
    print(f"   • fade_prediction_examples.png - ML prediction examples")
    
    print(f"   This project demonstrates:")
    print(f"   • Model-Based Systems Engineering (MBSE) with Python")
    print(f"   • Optical satellite communication physics")
    print(f"   • Real orbital mechanics integration") 
    print(f"   • Machine Learning for predictive maintenance")
    print(f"   • Professional data visualization")
    print(f"   • System performance optimization")

if __name__ == "__main__":
    showcase_complete_project()