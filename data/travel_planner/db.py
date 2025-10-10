import os
from typing import Any, Dict, List
import sys

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# Add TravelPlanner to path
travel_planner_path = os.path.join(os.path.dirname(__file__), "reward_utils")
if travel_planner_path not in sys.path:
    sys.path.append(travel_planner_path)

from tp_utils.func import (
    extract_before_parenthesis,
    get_valid_name_city,
)

class TravelDB:
    """
    A database of travel-related information including restaurants, attractions,
    accommodations, and flights.
    """

    def __init__(self):
        """Initialize the travel database by loading all CSV files."""
        try:
            import pandas as pd

            # Load restaurants
            restaurants_path = os.path.join(ROOT_PATH, "restaurants.csv")
            self.restaurants_df = pd.read_csv(restaurants_path)

            # Load attractions
            attractions_path = os.path.join(ROOT_PATH, "attractions.csv")
            self.attractions_df = pd.read_csv(attractions_path)

            # Load accommodations
            accommodations_path = os.path.join(ROOT_PATH, "accommodations.csv")
            self.accommodations_df = pd.read_csv(accommodations_path)

            # Load flights
            flights_path = os.path.join(ROOT_PATH, "flights.csv")
            self.flights_df = pd.read_csv(flights_path)

            # Load city-state mapping
            city_state_path = os.path.join(ROOT_PATH, "city_state.csv")
            self.city_state_df = pd.read_csv(city_state_path)

            print(
                f"TravelDB loaded: {len(self.restaurants_df)} restaurants, {len(self.attractions_df)} attractions, {len(self.accommodations_df)} accommodations, {len(self.flights_df)} flights"
            )

        except Exception as e:
            print(f"Error loading TravelDB: {e}")
            # Initialize empty dataframes if loading fails
            self.restaurants_df = pd.DataFrame()
            self.attractions_df = pd.DataFrame()
            self.accommodations_df = pd.DataFrame()
            self.flights_df = pd.DataFrame()
            self.city_state_df = pd.DataFrame()

    def get_restaurant_info(self, restaurant_name: str, city: str) -> Dict[str, Any]:
        """Get restaurant information by name and city."""
        city = extract_before_parenthesis(city)
        try:
            mask = (
                self.restaurants_df["name"].str.lower() == restaurant_name.lower()
            ) & (self.restaurants_df["city"].str.lower() == city.lower())
            if mask.any():
                return self.restaurants_df[mask].iloc[0].to_dict()
        except Exception as e:
            print(f"Error looking up restaurant {restaurant_name} in {city}: {e}")
        return None

    def get_attraction_info(self, attraction_name: str, city: str) -> Dict[str, Any]:
        """Get attraction information by name and city."""
        city = extract_before_parenthesis(city)
        try:
            mask = (
                self.attractions_df["name"].str.lower() == attraction_name.lower()
            ) & (self.attractions_df["city"].str.lower() == city.lower())
            if mask.any():
                return self.attractions_df[mask].iloc[0].to_dict()
        except Exception as e:
            print(f"Error looking up attraction {attraction_name} in {city}: {e}")
        return None

    def get_accommodation_info(self, hotel_name: str, city: str) -> Dict[str, Any]:
        """Get accommodation information by name and city."""
        city = extract_before_parenthesis(city)
        try:
            mask = (
                self.accommodations_df["name"].str.lower() == hotel_name.lower()
            ) & (self.accommodations_df["city"].str.lower() == city.lower())
            if mask.any():
                return self.accommodations_df[mask].iloc[0].to_dict()
        except Exception as e:
            print(f"Error looking up accommodation {hotel_name} in {city}: {e}")
        return None

    def get_flight_info(
        self, departure_city: str, arrival_city: str
    ) -> List[Dict[str, Any]]:
        """Get flight information between two cities."""
        departure_city = extract_before_parenthesis(departure_city)
        arrival_city = extract_before_parenthesis(arrival_city)
        try:
            mask = (
                self.flights_df["departure_city"].str.lower() == departure_city.lower()
            ) & (self.flights_df["arrival_city"].str.lower() == arrival_city.lower())
            if mask.any():
                return self.flights_df[mask].to_dict("records")
        except Exception as e:
            print(
                f"Error looking up flights from {departure_city} to {arrival_city}: {e}"
            )
        return []

    def get_city_state(self, city: str) -> str:
        """Get state for a given city."""
        city = extract_before_parenthesis(city)
        try:
            mask = self.city_state_df["city"].str.lower() == city.lower()
            if mask.any():
                return self.city_state_df[mask].iloc[0]["state"]
        except Exception as e:
            print(f"Error looking up state for city {city}: {e}")
        return None

    def get_travel_item_by_name(self, name_and_city: str) -> Dict[str, Any]:
        """
        Get a travel item (restaurant, attraction, or accommodation) by name and city.
        Returns the first match found, or None if not found.
        """
        try:
            name, city = get_valid_name_city(name_and_city)

            # Try to find as restaurant first
            restaurant_info = self.get_restaurant_info(name, city)
            if restaurant_info:
                return {"type": "restaurant", "info": restaurant_info}
            
            # Try to find as attraction
            attraction_info = self.get_attraction_info(name, city)
            if attraction_info:
                return {"type": "attraction", "info": attraction_info}
            
            # Try to find as accommodation
            accommodation_info = self.get_accommodation_info(name, city)
            if accommodation_info:
                return {"type": "accommodation", "info": accommodation_info}
            
            return None
        except Exception as e:
            print(f"Error looking up travel item {name_and_city}: {e}")
            return None
