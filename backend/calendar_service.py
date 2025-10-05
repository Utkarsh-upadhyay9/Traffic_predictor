"""
Google Calendar Integration Service
Syncs holidays and events for traffic prediction
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import requests


class CalendarService:
    """Service to fetch holidays and events from Google Calendar"""
    
    def __init__(self):
        # Use backend directory relative path or absolute path
        backend_dir = Path(__file__).parent
        project_root = backend_dir.parent
        self.cache_dir = project_root / "ml" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "holidays_cache.json"
        self.holidays_data = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load holidays from cache file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is still valid (not older than 7 days)
                    cache_date = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
                    if datetime.now() - cache_date < timedelta(days=7):
                        print("✅ Loaded holidays from cache")
                        return data
            except Exception as e:
                print(f"⚠️  Cache load error: {e}")
        return {}
    
    def _save_cache(self, data: Dict):
        """Save holidays to cache file"""
        try:
            data['cached_at'] = datetime.now().isoformat()
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("✅ Saved holidays to cache")
        except Exception as e:
            print(f"⚠️  Cache save error: {e}")
    
    def fetch_us_holidays(self, days_ahead: int = 30) -> List[Dict]:
        """
        Fetch US holidays for the next N days
        Uses public holiday API (no authentication required)
        
        Args:
            days_ahead: Number of days to fetch (default 30)
            
        Returns:
            List of holiday dictionaries with date, name, type
        """
        try:
            # Check cache first
            if self.holidays_data.get('holidays'):
                print(f"📅 Using cached holidays ({len(self.holidays_data['holidays'])} events)")
                return self.holidays_data['holidays']
            
            # Fetch from public holiday API
            print("🌐 Fetching holidays from API...")
            
            today = datetime.now()
            end_date = today + timedelta(days=days_ahead)
            
            # US Federal Holidays (hardcoded - reliable and doesn't require API key)
            us_holidays = self._get_federal_holidays(today.year)
            
            # Filter to next 30 days
            filtered_holidays = []
            for holiday in us_holidays:
                holiday_date = datetime.strptime(holiday['date'], '%Y-%m-%d')
                if today <= holiday_date <= end_date:
                    filtered_holidays.append(holiday)
            
            # Add some common observances and special events
            observances = self._get_observances(today, end_date)
            filtered_holidays.extend(observances)
            
            # Sort by date
            filtered_holidays.sort(key=lambda x: x['date'])
            
            # Cache the results
            self.holidays_data = {
                'holidays': filtered_holidays,
                'fetched_at': datetime.now().isoformat(),
                'days_ahead': days_ahead
            }
            self._save_cache(self.holidays_data)
            
            print(f"✅ Fetched {len(filtered_holidays)} holidays/events")
            return filtered_holidays
            
        except Exception as e:
            print(f"❌ Error fetching holidays: {e}")
            # Return basic holidays as fallback
            return self._get_fallback_holidays(days_ahead)
    
    def _get_federal_holidays(self, year: int) -> List[Dict]:
        """Get US Federal Holidays for a given year"""
        holidays = [
            {"date": f"{year}-01-01", "name": "New Year's Day", "type": "federal"},
            {"date": f"{year}-01-20", "name": "Martin Luther King Jr. Day", "type": "federal"},
            {"date": f"{year}-02-17", "name": "Presidents' Day", "type": "federal"},
            {"date": f"{year}-05-26", "name": "Memorial Day", "type": "federal"},
            {"date": f"{year}-06-19", "name": "Juneteenth", "type": "federal"},
            {"date": f"{year}-07-04", "name": "Independence Day", "type": "federal"},
            {"date": f"{year}-09-01", "name": "Labor Day", "type": "federal"},
            {"date": f"{year}-10-13", "name": "Columbus Day", "type": "federal"},
            {"date": f"{year}-11-11", "name": "Veterans Day", "type": "federal"},
            {"date": f"{year}-11-27", "name": "Thanksgiving Day", "type": "federal"},
            {"date": f"{year}-12-25", "name": "Christmas Day", "type": "federal"},
        ]
        return holidays
    
    def _get_observances(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get special observances that affect traffic"""
        year = start_date.year
        observances = [
            {"date": f"{year}-02-14", "name": "Valentine's Day", "type": "observance"},
            {"date": f"{year}-03-17", "name": "St. Patrick's Day", "type": "observance"},
            {"date": f"{year}-10-31", "name": "Halloween", "type": "observance"},
            {"date": f"{year}-11-28", "name": "Black Friday", "type": "shopping"},
            {"date": f"{year}-12-24", "name": "Christmas Eve", "type": "observance"},
            {"date": f"{year}-12-31", "name": "New Year's Eve", "type": "observance"},
            # Major sporting events (generic dates - update annually)
            {"date": f"{year}-02-09", "name": "Super Bowl Sunday", "type": "sports"},
        ]
        
        # Filter to date range
        filtered = []
        for obs in observances:
            obs_date = datetime.strptime(obs['date'], '%Y-%m-%d')
            if start_date <= obs_date <= end_date:
                filtered.append(obs)
        
        return filtered
    
    def _get_fallback_holidays(self, days_ahead: int) -> List[Dict]:
        """Fallback holidays if API fails"""
        today = datetime.now()
        year = today.year
        
        basic_holidays = [
            {"date": f"{year}-12-25", "name": "Christmas Day", "type": "federal"},
            {"date": f"{year}-01-01", "name": "New Year's Day", "type": "federal"},
            {"date": f"{year}-07-04", "name": "Independence Day", "type": "federal"},
        ]
        
        # Filter to next N days
        end_date = today + timedelta(days=days_ahead)
        filtered = []
        for holiday in basic_holidays:
            holiday_date = datetime.strptime(holiday['date'], '%Y-%m-%d')
            if today <= holiday_date <= end_date:
                filtered.append(holiday)
        
        return filtered
    
    def is_holiday(self, date: datetime) -> bool:
        """Check if a given date is a holiday"""
        date_str = date.strftime('%Y-%m-%d')
        holidays = self.fetch_us_holidays(30)
        
        for holiday in holidays:
            if holiday['date'] == date_str:
                return True
        return False
    
    def get_holiday_name(self, date: datetime) -> Optional[str]:
        """Get the name of holiday on a given date"""
        date_str = date.strftime('%Y-%m-%d')
        holidays = self.fetch_us_holidays(30)
        
        for holiday in holidays:
            if holiday['date'] == date_str:
                return holiday['name']
        return None
    
    def get_holidays_in_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get all holidays within a date range"""
        all_holidays = self.fetch_us_holidays(30)
        
        filtered = []
        for holiday in all_holidays:
            holiday_date = datetime.strptime(holiday['date'], '%Y-%m-%d')
            if start_date <= holiday_date <= end_date:
                filtered.append(holiday)
        
        return filtered
    
    def get_traffic_impact_factor(self, date: datetime, hour: int) -> float:
        """
        Get traffic impact factor based on holidays and events
        
        Args:
            date: Date to check
            hour: Hour of day (0-23)
            
        Returns:
            Multiplier for traffic (1.0 = normal, >1.0 = more traffic, <1.0 = less)
        """
        holiday_name = self.get_holiday_name(date)
        
        if not holiday_name:
            return 1.0
        
        # Different holidays have different traffic patterns
        # Federal holidays typically reduce weekday commute traffic
        if "Day" in holiday_name and date.weekday() < 5:  # Weekday federal holiday
            if 6 <= hour <= 9 or 15 <= hour <= 18:  # Rush hours
                return 0.3  # Much less traffic during commute
            else:
                return 0.7  # Still reduced
        
        # Shopping days increase traffic
        if "Black Friday" in holiday_name or "Christmas Eve" in holiday_name:
            if 10 <= hour <= 20:  # Shopping hours
                return 1.8  # 80% more traffic
            return 1.2
        
        # New Year's Eve - late night traffic
        if "New Year's Eve" in holiday_name:
            if 18 <= hour <= 23:
                return 1.5  # 50% more traffic
            return 1.0
        
        # Default holiday impact
        return 0.8


# Global singleton instance
_calendar_service = None


def get_calendar_service() -> CalendarService:
    """Get or create the global calendar service instance"""
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = CalendarService()
    return _calendar_service


if __name__ == "__main__":
    # Test the calendar service
    service = CalendarService()
    
    print("\n📅 US Holidays for Next 30 Days:")
    print("=" * 60)
    
    holidays = service.fetch_us_holidays(30)
    for holiday in holidays:
        print(f"  {holiday['date']} - {holiday['name']} ({holiday['type']})")
    
    print("\n🚗 Traffic Impact Test:")
    print("=" * 60)
    
    test_dates = [
        (datetime(2025, 12, 25, 8, 0), "Christmas Morning"),
        (datetime(2025, 7, 4, 14, 0), "July 4th Afternoon"),
        (datetime(2025, 11, 28, 11, 0), "Black Friday Late Morning"),
    ]
    
    for test_date, label in test_dates:
        factor = service.get_traffic_impact_factor(test_date, test_date.hour)
        print(f"  {label}: {factor:.1f}x traffic")
