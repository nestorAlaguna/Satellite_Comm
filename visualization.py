# visualization.py - FIXED VERSION
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mbse_model import Satellite, GroundStation
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class SatelliteVisualizer:
    """Create visualizations for the satellite communication system"""
    
    def __init__(self):
        plt.style.use('default')  # Use default style for better compatibility
        self.fig = None
    
    def plot_satellite_constellation(self, satellites: List[Satellite], 
                                   ground_stations: List[GroundStation],
                                   save_path: str = "satellite_constellation.png"):
        """Create a 3D plot of satellites around Earth - FIXED VERSION"""
        print(" Creating satellite constellation visualization...")
        
        # Create figure with better layout control
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.patch.set_facecolor('white')
        
        # 3D plot - FIXED: Better Earth and layout
        ax1 = self.fig.add_subplot(221, projection='3d')
        self._plot_3d_constellation(ax1, satellites, ground_stations)
        
        # 2D polar plot (top-down view)
        ax2 = self.fig.add_subplot(222, polar=True)
        self._plot_polar_view(ax2, satellites, ground_stations)
        
        # Ground station coverage
        ax3 = self.fig.add_subplot(223)
        self._plot_ground_station_coverage(ax3, ground_stations)
        
        # System overview
        ax4 = self.fig.add_subplot(224)
        self._plot_system_overview(ax4, satellites, ground_stations)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout(pad=3.0)  # Increased padding
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Saved constellation visualization to {save_path}")
        plt.close()
    
    def _plot_3d_constellation(self, ax, satellites: List[Satellite], ground_stations: List[GroundStation]):
        """3D plot of satellites around Earth - FIXED with better Earth"""
        # Draw Earth with better texture approximation
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        r = 6371  # Earth radius in km
        
        # Create sphere
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Use a blue color that looks like Earth from space
        ax.plot_surface(x, y, z, color='#1f77b4', alpha=0.6, edgecolor='none')
        
        # Track plotted labels to avoid duplicates in legend
        sat_plotted = False
        gs_plotted = False
        
        # Plot satellites
        sat_positions = []
        for sat in satellites:
            if hasattr(sat, 'position') and sat.position is not None:
                # Ensure position is a numpy array and has 3 elements
                pos = np.array(sat.position)
                if pos.size >= 3:
                    pos = pos[:3] / 1000  # Convert to km, take first 3 elements
                    sat_positions.append(pos)
                    label = 'Satellites' if not sat_plotted else ""
                    ax.scatter(pos[0], pos[1], pos[2], color='red', s=60, 
                              label=label, alpha=0.8)
                    sat_plotted = True
        
        # Plot ground stations
        earth_radius = 6371
        for gs in ground_stations:
            if len(gs.location) >= 3:
                lat, lon, alt = gs.location[0], gs.location[1], gs.location[2]
                # Convert lat/lon to 3D coordinates on Earth surface
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad) 
                z = earth_radius * np.sin(lat_rad)
                
                label = 'Ground Stations' if not gs_plotted else ""
                ax.scatter(x, y, z, color='green', s=80, marker='^', 
                          label=label, alpha=0.8)
                gs_plotted = True
        
        ax.set_xlabel('X (km)', fontweight='bold')
        ax.set_ylabel('Y (km)', fontweight='bold')
        ax.set_zlabel('Z (km)', fontweight='bold')
        ax.set_title('3D Satellite Constellation', fontweight='bold', fontsize=12, pad=20)
        
        # Add legend only if we have items
        if sat_plotted or gs_plotted:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
        
        # Set equal aspect ratio
        max_range = 15000  # Extend view a bit beyond LEO
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def _plot_polar_view(self, ax, satellites: List[Satellite], ground_stations: List[GroundStation]):
        """Polar plot showing satellite positions"""
        earth_radius = 6371
        
        # Plot satellites
        for sat in satellites:
            if hasattr(sat, 'position') and sat.position is not None:
                pos = np.array(sat.position)
                if pos.size >= 3:
                    pos = pos[:3] / 1000  # Convert to km
                    # Calculate polar coordinates
                    r = np.sqrt(pos[0]**2 + pos[1]**2)
                    theta = np.arctan2(pos[1], pos[0])
                    ax.scatter(theta, r, color='red', s=40, alpha=0.7, label='Satellites' if not hasattr(self, '_sat_polar_plotted') else "")
                    self._sat_polar_plotted = True
        
        # Plot ground stations
        for gs in ground_stations:
            if len(gs.location) >= 2:
                lat, lon = gs.location[0], gs.location[1]
                theta = np.radians(lon)
                r = earth_radius  # On Earth's surface
                ax.scatter(theta, r, color='green', s=60, marker='^', alpha=0.8, 
                          label='Ground Stations' if not hasattr(self, '_gs_polar_plotted') else "")
                self._gs_polar_plotted = True
        
        ax.set_theta_zero_location("N")  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title('Top-Down Polar View', fontweight='bold', fontsize=12, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2])  # Only first two unique labels
    
    def _plot_ground_station_coverage(self, ax, ground_stations: List[GroundStation]):
        """World map with ground station locations"""
        lats = []
        lons = []
        names = []
        
        for gs in ground_stations:
            if len(gs.location) >= 2:
                lats.append(gs.location[0])
                lons.append(gs.location[1])
                names.append(gs.id)
        
        if lats and lons:
            # Simple world map background
            ax.scatter(lons, lats, c='red', s=100, alpha=0.7, edgecolors='black')
            
            # Add station names
            for i, name in enumerate(names):
                ax.annotate(name, (lons[i], lats[i]), xytext=(8, 8), 
                           textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Set reasonable limits for world view
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Longitude', fontweight='bold')
            ax.set_ylabel('Latitude', fontweight='bold')
            ax.set_title('Ground Station Network', fontweight='bold', fontsize=12, pad=20)
            ax.grid(True, alpha=0.3)
            
            # Add some map features
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Equator
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # Prime meridian
    
    def _plot_system_overview(self, ax, satellites: List[Satellite], ground_stations: List[GroundStation]):
        """System overview statistics"""
        stats = {
            'Total Satellites': len(satellites),
            'Ground Stations': len(ground_stations),
            'Total Links': len(satellites) * len(ground_stations),
            'Optical Systems': len([s for s in satellites if hasattr(s, 'wavelength')])
        }
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = ax.bar(range(len(stats)), list(stats.values()), color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(list(stats.keys()), rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('System Overview', fontweight='bold', fontsize=12, pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Set y-axis to start from 0 with some headroom
        max_val = max(stats.values()) if stats.values() else 1
        ax.set_ylim(0, max_val * 1.15)
    
    def create_performance_dashboard(self, simulation_results: List[Dict], 
                               save_path: str = "performance_dashboard.png"):
        """Create a comprehensive performance dashboard - PROPERLY FIXED LAYOUT"""
        print(" Creating performance dashboard...")
        
        if not simulation_results:
            print(" No results to visualize")
            return
        
        # Create figure with constrained layout for better control
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.patch.set_facecolor('white')
        
        # Set the main title with proper positioning
        self.fig.suptitle('Satellite Communication System Performance Dashboard', 
                        fontsize=16, fontweight='bold', y=0.99)
        
        # Remove any automatic percentage formatting on empty axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.ticklabel_format(useOffset=False)  # Prevent scientific notation issues
            # Clear any default y-axis formatters that show percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 1. BER distribution 
        bers = [r['ber'] for r in simulation_results if 'ber' in r and r['ber'] is not None]
        if bers:
            # Filter out extreme values for better visualization
            valid_bers = [ber for ber in bers if ber > 0 and ber <= 1]
            if valid_bers:
                log_bers = np.log10(valid_bers)
                counts, bins, patches = ax1.hist(log_bers, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(np.log10(1e-6), color='red', linestyle='--', linewidth=2, 
                        label='Requirement (1e-6)')
                ax1.set_xlabel('log10(BER)', fontweight='bold')
                ax1.set_ylabel('Frequency', fontweight='bold')
                ax1.set_title('Bit Error Rate Distribution', fontweight='bold', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Set proper y-axis limits to avoid percentage formatting
                max_count = max(counts) if len(counts) > 0 else 1
                ax1.set_ylim(0, max_count * 1.1)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            else:
                ax1.text(0.5, 0.5, 'No valid BER data', ha='center', va='center', 
                        transform=ax1.transAxes, fontweight='bold', fontsize=10)
                ax1.set_title('Bit Error Rate Distribution', fontweight='bold', fontsize=12)
                ax1.set_xticks([])
                ax1.set_yticks([])
        else:
            ax1.text(0.5, 0.5, 'No BER data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontweight='bold', fontsize=10)
            ax1.set_title('Bit Error Rate Distribution', fontweight='bold', fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
        
        # 2. Elevation vs BER 
        elevations = [r.get('elevation', 0) for r in simulation_results]
        if bers and elevations and len(bers) == len(elevations):
            # Filter to matching lengths
            valid_pairs = [(e, b) for e, b in zip(elevations, bers) if b > 0 and b <= 1]
            if valid_pairs:
                elev_vals, ber_vals = zip(*valid_pairs)
                scatter = ax2.scatter(elev_vals, np.log10(ber_vals), alpha=0.6, c=ber_vals, 
                                    cmap='viridis', s=50)
                ax2.set_xlabel('Elevation (degrees)', fontweight='bold')
                ax2.set_ylabel('log10(BER)', fontweight='bold')
                ax2.set_title('Link Quality vs Satellite Elevation', fontweight='bold', fontsize=12)
                
                # Add colorbar with proper positioning
                cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
                cbar.set_label('BER', fontweight='bold')
                
                ax2.grid(True, alpha=0.3)
                
                # Set proper y-axis formatter
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            else:
                ax2.text(0.5, 0.5, 'No valid elevation/BER data', ha='center', va='center', 
                        transform=ax2.transAxes, fontweight='bold', fontsize=10)
                ax2.set_title('Link Quality vs Satellite Elevation', fontweight='bold', fontsize=12)
                ax2.set_xticks([])
                ax2.set_yticks([])
        else:
            ax2.text(0.5, 0.5, 'No elevation/BER data', ha='center', va='center', 
                    transform=ax2.transAxes, fontweight='bold', fontsize=10)
            ax2.set_title('Link Quality vs Satellite Elevation', fontweight='bold', fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 3. Success rate by ground station 
        gs_names = list(set(r.get('ground_station', 'Unknown') for r in simulation_results))
        success_rates = []
        
        for gs in gs_names:
            gs_results = [r for r in simulation_results if r.get('ground_station') == gs]
            if gs_results:
                success_rate = sum(1 for r in gs_results if r.get('link_operational', False)) / len(gs_results) * 100
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        if success_rates and any(rate > 0 for rate in success_rates):
            colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#ff6b6b', '#ffa600']
            bars = ax3.bar(gs_names, success_rates, color=colors[:len(gs_names)], alpha=0.8, edgecolor='black')
            ax3.set_ylabel('Success Rate (%)', fontweight='bold')
            ax3.set_title('Ground Station Performance', fontweight='bold', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                if height > 0:  # Only label non-zero bars
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Set y-axis limits and formatter
            max_rate = max(success_rates) if success_rates else 100
            ax3.set_ylim(0, max_rate * 1.15)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        else:
            ax3.text(0.5, 0.5, 'No operational links found', ha='center', va='center', 
                    transform=ax3.transAxes, fontweight='bold', fontsize=10)
            ax3.set_title('Ground Station Performance', fontweight='bold', fontsize=12)
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # 4. Timeline of operational status 
        operational = [1 if r.get('link_operational', False) else 0 for r in simulation_results]
        
        if operational and any(op == 1 for op in operational):
            time_indices = range(len(operational))
            ax4.plot(time_indices, operational, 'o-', alpha=0.7, color='green', linewidth=2, markersize=4)
            ax4.set_xlabel('Time Step', fontweight='bold')
            ax4.set_ylabel('Operational Status', fontweight='bold')
            ax4.set_title('Link Availability Over Time', fontweight='bold', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['No', 'Yes'])
            ax4.set_xlim(-0.5, len(operational) - 0.5)
            
            # Add some statistics
            operational_percentage = (sum(operational) / len(operational)) * 100
            ax4.text(0.02, 0.98, f'Operational: {operational_percentage:.1f}%', 
                    transform=ax4.transAxes, fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        else:
            ax4.text(0.5, 0.5, 'No operational timeline data', ha='center', va='center', 
                    transform=ax4.transAxes, fontweight='bold', fontsize=10)
            ax4.set_title('Link Availability Over Time', fontweight='bold', fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # Use constrained_layout for better automatic spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        
        # Additional manual adjustment if needed
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, 
                        hspace=0.3, wspace=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                pad_inches=0.5)  # Reduced padding
        print(f"✅ Saved performance dashboard to {save_path}")
        plt.close()

# Example usage
if __name__ == "__main__":
    # Test with some sample data
    from run_optimized_digital_twin import OptimizedDigitalTwin
    
    print(" GENERATING VISUALIZATIONS ")
    
    # Create a sample system
    twin = OptimizedDigitalTwin()
    twin.add_enhanced_ground_stations()
    twin.create_optimized_satellites()
    
    # Create visualizer
    viz = SatelliteVisualizer()
    
    # Generate constellation plot
    viz.plot_satellite_constellation(twin.satellites, twin.ground_stations)
    


