# data_manager.py
import requests
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os
from typing import Optional, List, Tuple

# SQLAlchemy setup
Base = declarative_base()

class TLEEntry(Base):
    """SQL model for storing TLE data"""
    __tablename__ = 'tle_data'
    
    id = Column(Integer, primary_key=True)
    satellite_name = Column(String, unique=True, nullable=False)
    tle_line1 = Column(String, nullable=False)
    tle_line2 = Column(String, nullable=False)
    last_updated = Column(DateTime, nullable=False)
    source_url = Column(String, nullable=False)

class SatelliteDataManager:
    """
    Manages satellite TLE data with hybrid caching strategy.
    Demonstrates professional data pipeline architecture.
    """
    
    def __init__(self, database_url: str = "sqlite:///satellite_data.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # TLE sources (using Celestrak - the professional standard)
        self.tle_sources = {
            'active': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
            'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
            'weather': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle'
        }
    
    def fetch_fresh_tle_data(self, group: str = 'active') -> List[Tuple[str, str, str]]:
        """
        Fetch fresh TLE data from Celestrak.
        Returns: List of (satellite_name, tle_line1, tle_line2)
        """
        print(f" Fetching fresh TLE data for {group} satellites...")
        try:
            response = requests.get(self.tle_sources[group])
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            tle_data = []
            
            # TLE data comes in groups of 3 lines: name, line1, line2
            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    name = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    # Basic data validation (like your data_pipeline project!)
                    if (name and line1.startswith('1 ') and line2.startswith('2 ')):
                        tle_data.append((name, line1, line2))
                        # Update or insert into database
                        self._update_tle_in_db(name, line1, line2, self.tle_sources[group])
            
            print(f" Successfully fetched {len(tle_data)} satellites")
            return tle_data
            
        except requests.RequestException as e:
            print(f" Failed to fetch TLE data: {e}")
            print(" Falling back to cached database data...")
            return self.get_cached_tle_data()
    
    def _update_tle_in_db(self, name: str, line1: str, line2: str, source: str):
        """Update or insert TLE data in the database"""
        existing = self.session.query(TLEEntry).filter_by(satellite_name=name).first()
        
        if existing:
            # Update existing entry
            existing.tle_line1 = line1
            existing.tle_line2 = line2
            existing.last_updated = datetime.utcnow()
            existing.source_url = source
        else:
            # Create new entry
            new_entry = TLEEntry(
                satellite_name=name,
                tle_line1=line1,
                tle_line2=line2,
                last_updated=datetime.utcnow(),
                source_url=source
            )
            self.session.add(new_entry)
        
        self.session.commit()
    
    def get_cached_tle_data(self, max_age_hours: int = 24) -> List[Tuple[str, str, str]]:
        """
        Get TLE data from cache if it's fresh enough.
        This enables offline operation and better performance.
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        fresh_entries = self.session.query(TLEEntry).filter(
            TLEEntry.last_updated >= cutoff_time
        ).all()
        
        if fresh_entries:
            print(f" Using cached TLE data ({len(fresh_entries)} fresh satellites)")
            return [(entry.satellite_name, entry.tle_line1, entry.tle_line2) for entry in fresh_entries]
        else:
            print(" No fresh cached data available")
            return []
    
    def get_tle_data(self, force_refresh: bool = False) -> List[Tuple[str, str, str]]:
        """
        Main method to get TLE data using hybrid strategy.
        First tries cache, falls back to fresh fetch if needed.
        """
        if not force_refresh:
            cached_data = self.get_cached_tle_data()
            if cached_data:
                return cached_data
        
        # If no fresh cache or force_refresh=True, fetch fresh data
        return self.fetch_fresh_tle_data()
    
    def get_specific_satellite(self, satellite_name: str) -> Optional[Tuple[str, str, str]]:
        """Get TLE data for a specific satellite by name"""
        entry = self.session.query(TLEEntry).filter_by(satellite_name=satellite_name).first()
        if entry:
            return (entry.satellite_name, entry.tle_line1, entry.tle_line2)
        return None

# Test the data manager
if __name__ == "__main__":
    print("=== TESTING SATELLITE DATA MANAGER ===")
    data_manager = SatelliteDataManager()
    
    # Test the hybrid approach
    tle_data = data_manager.get_tle_data()
    print(f"\nRetrieved {len(tle_data)} satellites")
    
    # Show first 3 satellites as example
    for i, (name, line1, line2) in enumerate(tle_data[:3]):
        print(f"\n{i+1}. {name}")
        print(f"   Line1: {line1[:50]}...")
        print(f"   Line2: {line2[:50]}...")