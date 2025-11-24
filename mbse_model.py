# mbse_model.py

# IMPORTING
from pydantic import BaseModel, Field, ConfigDict  
from typing import Optional
import numpy as np
# import for atmospheric conditions
from enum import Enum

# DEFINING BLOCKS: 
# The Satellite 
class Satellite(BaseModel):
    # NEW: This is the model configuration. It tells Pydantic how to behave.
    # 'arbitrary_types_allowed=True' lets us use types like np.ndarray without error.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # These are the block's ATTRIBUTES. They define its state and properties.
    
    # 'id' is a string. It's required. This is the satellite's name.
    id: str
    
    # 'position' is a NumPy array of 3 numbers (x, y, z in space). It's required.
    # Because we set arbitrary_types_allowed=True, Pydantic won't complain about np.ndarray.
    position: np.ndarray
    
    # 'transmit_power_watts' is a float. It MUST be greater than 0 (gt=0).
    # 'Field' lets us add this extra validation. Default is 10.0 Watts.
    transmit_power_watts: float = Field(default=10.0, gt=0)
    
    # 'wavelength' is the light's color. 1550 nanometers is common for optics.
    # The default is 1550e-9 meters. It also must be > 0.
    wavelength: float = Field(default=1550e-9, gt=0)
    
    # 'antenna_gain' represents how well the satellite's telescope focuses light.
    antenna_gain: float = Field(default=1.0, ge=0) # ge=0 means 'greater than or equal to zero'

    # Adding a new attribute. 'bool' is a standard type, so no special config is needed.
    # We give it a default value of True.
    is_operational: bool = True

    # --- Requirement Definitions (Our "Requirements Model") ---
class Requirement(BaseModel):
    """A simple class to represent a system requirement."""
    id: str  # e.g., "REQ-001"
    text: str  # The descriptive text of the requirement
    verified: bool = False  # This will be set to True once we verify it

    def verify(self, condition: bool):
        """A simple method to verify a requirement."""
        self.verified = condition
        return condition

# Ground Station
class GroundStation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    # Let's use a simple list for lat, lon, alt. It's easier for now than a np.array.
    # This represents the ground station's fixed location on Earth.
    # Latitude, Longitude, Altitude (in meters)
    location: list[float] = Field(..., min_items=3, max_items=3)  # '...' means this field is required and has no default.

    # NEW CONCEPT: This is from your optical background.
    # The aperture diameter (in meters) of the receiving telescope.
    # A larger aperture collects more light, improving the link.
    aperture_diameter: float = Field(default=0.5, gt=0)  # Default 50 cm telescope

    # This is a system REQUIREMENT embedded as a model parameter.
    # The ground station cannot talk to satellites below this angle to the horizon.
    min_elevation: float = Field(default=5.0, ge=0)  # ge=0 means 'greater than or equal to zero'

    # Let's add a method (a "behavior") to this block.
    def can_track(self, satellite_elevation: float) -> bool:
        """
        Method to check if the ground station can track a satellite based on elevation.
        This is a simple behavioral model.
        """
        can_track = satellite_elevation >= self.min_elevation
        return can_track

# Enum for atmospheric conditions.

class AtmosphericCondition(str, Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    TURBULENT = "turbulent"

# Optical link


class OpticalLink(BaseModel):
    """
    The OpticalLink is a 'Parametric Model' that connects a Satellite and GroundStation.
    UPDATED: More realistic optical communication parameters
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    satellite: Satellite
    ground_station: GroundStation
    atmospheric_condition: AtmosphericCondition
    distance: float = Field(..., gt=0)
    received_power: Optional[float] = Field(default=None, description="Calculated received power in Watts.")
    bit_error_rate: Optional[float] = Field(default=None, description="Calculated Bit Error Rate.")

    def calculate_link_budget(self):
        """Implements the Free-Space Optical Link Budget - UPDATED for realism"""
                
        # 1. Calculate Free Space Path Loss (FSPL)
        # FSPL = (λ/(4πd))^2
        fspl = (self.satellite.wavelength / (4 * np.pi * self.distance)) ** 2
        
        # 2. Calculate Pointing Loss 
        pointing_loss_factor = 0.9  # UPDATED: 10% loss instead of 20%

        # 3. Calculate Atmospheric Loss 
        atmospheric_loss_map = {
            AtmosphericCondition.CLEAR: 0.95,     # UPDATED: 5% loss in clear weather
            AtmosphericCondition.CLOUDY: 0.3,     # 70% loss in clouds
            AtmosphericCondition.TURBULENT: 0.7   # 30% loss in turbulence
        }
        atmospheric_loss_factor = atmospheric_loss_map[self.atmospheric_condition]

        # 4. Calculate Receiver Gain 
        # For optical systems, the receiver gain is: G_rx = (π * D / λ)^2
        receiver_gain = (np.pi * self.ground_station.aperture_diameter / self.satellite.wavelength) ** 2

        # 5. TOTAL RECEIVED POWER (UPDATED: more realistic calculation)
        self.received_power = (self.satellite.transmit_power_watts *
                              self.satellite.antenna_gain *
                              fspl *
                              pointing_loss_factor *
                              atmospheric_loss_factor *
                              receiver_gain)

        # 6. Calculate Bit Error Rate 
        # For optical OOK systems with avalanche photodiodes
        # Assuming a more sensitive receiver
        
        # Receiver sensitivity (photons per bit)
        # Real optical systems can detect much lower power levels
        required_photons_per_bit = 100  # UPDATED: More sensitive receiver
        
        # Calculate received photons per second
        energy_per_photon = 6.626e-34 * 3e8 / self.satellite.wavelength  # E = hc/λ
        photons_per_second = self.received_power / energy_per_photon
        
        # Assume data rate of 1 Gbps
        data_rate = 1e9  # 1 Gbps
        photons_per_bit = photons_per_second / data_rate
        
        # BER calculation for optical OOK
        if photons_per_bit > required_photons_per_bit:
            # Good link - use quantum-limited BER approximation
            self.bit_error_rate = 0.5 * np.exp(-photons_per_bit / 2)
        else:
            # Poor link - high BER
            self.bit_error_rate = 0.1  # 10% BER when insufficient photons
        
        # Ensure BER is within reasonable bounds
        self.bit_error_rate = max(1e-12, min(0.5, self.bit_error_rate))
        
        return self.received_power, self.bit_error_rate


# ------------------------------- Tests -------------------------------
# 3. LET'S TEST IT: This code only runs if we execute this file directly.
if __name__ == "__main__":
    print("=== Testing the Fixed Satellite Model ===")
    # Let's create an INSTANCE of our Satellite block.
    # This is like taking the blueprint (the class) and building a real satellite object.
    my_satellite = Satellite(
        id="ASTRA-1",
        position=np.array([1000000, 2000000, 3000000.0]) # A random position in meters. Using floats is good practice.
        # We are using the defaults for power, wavelength, and gain.
    )
    

    # Now, let's print our satellite to see what we made.
    print("\n1. My Satellite Object:")
    print(my_satellite)
    
    print("\n2. Accessing its attributes directly:")
    print(f"  Satellite ID: {my_satellite.id}")
    print(f"  Transmit Power: {my_satellite.transmit_power_watts} W")
    print(f"  Position Array: {my_satellite.position}")
    print(f"  Type of 'position': {type(my_satellite.position)}") # Let's confirm it's a NumPy array

    print("\n3. Testing Validation (this should FAIL):")
    try:
        bad_satellite = Satellite(
            id="BAD-ASTRA",
            position=np.array([0, 0, 0]),
            transmit_power_watts=-5.0  # This violates our 'gt=0' rule!
        )
    except Exception as e:
        print(f"   Good! Validation caught an error: {e}")

    
    
    print("\n4. Testing the new 'is_operational' attribute:")
    operational_sat = Satellite(
        id="OPERATIONAL-ASTRA",
        position=np.array([0, 0, 0.0]),
        is_operational=True
    )
    broken_sat = Satellite(
        id="BROKEN-ASTRA",
        position=np.array([0, 0, 0.0]),
        is_operational=False
    )
    print(f"   {operational_sat.id} is operational: {operational_sat.is_operational}")
    print(f"   {broken_sat.id} is operational: {broken_sat.is_operational}")

    print("\n" + "="*50)
    print("TESTING THE GROUND STATION BLOCK")
    print("="*50)

    # Let's create a ground station. We'll use a location in The Netherlands.
    eindhoven_gs = GroundStation(
        id="EINDHOVEN",
        location=[51.4416, 5.4697, 0.0],  # Latitude, Longitude, Altitude
        aperture_diameter=1.0,  # A 1-meter telescope
        min_elevation=10.0
    )

    print("\n1. My Ground Station Object:")
    print(eindhoven_gs)

    print("\n2. Testing the 'can_track' behavior:")
    # Simulate a satellite at a low elevation
    low_elevation = 3.0
    # Simulate a satellite at a high elevation
    high_elevation = 45.0

    can_track_low = eindhoven_gs.can_track(low_elevation)
    can_track_high = eindhoven_gs.can_track(high_elevation)

    print(f"   Can track satellite at {low_elevation}°? {can_track_low}")
    print(f"   Can track satellite at {high_elevation}°? {can_track_high}")

    print("\n3. Testing Requirement Verification:")
    # Now let's use our Requirement class.
    # This is REQ-001 from our initial plan: "The ground station shall only track satellites above min_elevation."
    req_001 = Requirement(
        id="REQ-001",
        text=f"The ground station shall only track satellites above {eindhoven_gs.min_elevation}° elevation."
    )

    # We verify the requirement by checking the behavior we just tested.
    # The requirement is satisfied if the station correctly REJECTS the low-elevation satellite.
    # We verify that it CANNOT track the low one.
    req_001.verify(condition=not can_track_low)

    print(f"   {req_001.id}: {req_001.text}")
    print(f"   Verified: {req_001.verified}")

    print("\n" + "="*50)
    print("TESTING THE OPTICAL LINK - THE FULL SYSTEM")
    print("="*50)

    # 1. Create our system blocks
    my_satellite = Satellite(
        id="ASTRA-1",
        position=np.array([1000000, 2000000, 3000000.0]),
        transmit_power_watts=5.0,  # 5 Watts of optical power
        antenna_gain=100.0  # The satellite's telescope gain
    )

    my_ground_station = GroundStation(
        id="EINDHOVEN",
        location=[51.4416, 5.4697, 0.0],
        aperture_diameter=0.1,  # 1-meter telescope
        min_elevation=5.0
    )

    # 2. Check if we can even establish a link (using the ground station's behavior)
    # Let's assume for this test that the satellite is visible (elevation > 5°)
    print("1. System Pre-Check:")
    print(f"   Ground Station can track: {my_ground_station.can_track(satellite_elevation=25.0)}")

    # 3. CREATE THE LINK between the blocks
    print("\n2. Creating Optical Link...")
    my_link = OpticalLink(
        satellite=my_satellite,
        ground_station=my_ground_station,
        atmospheric_condition=AtmosphericCondition.CLEAR,
        distance=500_000  # 500 km
    )

    # 4. RUN THE PARAMETRIC ANALYSIS (The core physics)
    print("\n3. Running Link Budget Calculation...")
    received_power, ber = my_link.calculate_link_budget()

    # 5. VERIFY SYSTEM REQUIREMENTS
    print("\n4. Verifying System Requirements:")
    req_002 = Requirement(
        id="REQ-002",
        text="The link shall maintain a Bit Error Rate (BER) better than 1e-6."
    )
    # Verify the requirement using our calculated BER
    req_002.verify(ber < 1e-6)
    print(f"   {req_002.id}: {req_002.text}")
    print(f"   Requirement MET: {req_002.verified} (Actual BER: {ber:.2e})")

    # Let's see what happens in BAD weather
    print("\n" + "-"*30)
    print("SIMULATING CLOUDY CONDITIONS...")
    bad_link = OpticalLink(
        satellite=my_satellite,
        ground_station=my_ground_station,
        atmospheric_condition=AtmosphericCondition.CLOUDY,  # CLOUDY now!
        distance=500_000
    )
    received_power_bad, ber_bad = bad_link.calculate_link_budget()

    req_002_bad = Requirement(id="REQ-002", text="BER < 1e-6")
    req_002_bad.verify(ber_bad < 1e-6)
    print(f"   Requirement MET in cloudy weather: {req_002_bad.verified} (Actual BER: {ber_bad:.2e})")