"""
Streamlit search interface components for the travel planner database.

Provides:
- BM25-based natural language search for restaurants, attractions, and accommodations
- Filter components (city, type, price, etc.)
- Search results display
- Liked items footer (all types in the same list)

Dependencies:
- rank_bm25: Install with `pip install rank-bm25`
"""

from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "liked_travel_items",
    "travel_searcher",
    "last_restaurants_search_query",
    "last_restaurants_search_results",
    "last_restaurants_search_filters",
    "last_attractions_search_query",
    "last_attractions_search_results",
    "last_attractions_search_filters",
    "last_accommodations_search_query",
    "last_accommodations_search_results",
    "last_accommodations_search_filters",
]

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required. Install it with: pip install rank-bm25"
    )
import re
from data.travel_planner.db import TravelDB
from data.travel_planner.streamlit_render import (
    _restaurant_to_dialog_content,
    _attraction_to_dialog_content,
    _accommodation_to_dialog_content,
    _get_dollar_signs,
)


class TravelSearcher:
    """Handles BM25 search and filtering for travel items (restaurants, attractions, accommodations)."""
    
    def __init__(self, travel_db: TravelDB):
        self.travel_db = travel_db
        
        # Combine all dataframes with a type identifier
        dfs = []
        
        # Restaurants
        if not travel_db.restaurants_df.empty:
            restaurants_df = travel_db.restaurants_df.copy()
            restaurants_df["item_type"] = "restaurant"
            restaurants_df["item_id"] = restaurants_df.apply(
                lambda row: f"restaurant_{row['name']}_{row['city']}", axis=1
            )
            dfs.append(restaurants_df)
        
        # Attractions
        if not travel_db.attractions_df.empty:
            attractions_df = travel_db.attractions_df.copy()
            attractions_df["item_type"] = "attraction"
            attractions_df["item_id"] = attractions_df.apply(
                lambda row: f"attraction_{row['name']}_{row['city']}", axis=1
            )
            dfs.append(attractions_df)
        
        # Accommodations
        if not travel_db.accommodations_df.empty:
            accommodations_df = travel_db.accommodations_df.copy()
            accommodations_df["item_type"] = "accommodation"
            accommodations_df["item_id"] = accommodations_df.apply(
                lambda row: f"accommodation_{row['name']}_{row['city']}", axis=1
            )
            dfs.append(accommodations_df)
        
        if not dfs:
            self.df = pd.DataFrame()
            self.tokenized_docs = []
            self.bm25 = None
            return
        
        # Combine all dataframes
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Create search_area with all text columns for indexing
        exclude_cols = {"item_type", "item_id", "price", "average_cost", "rating", 
                        "aggregate_rating", "review_rate_number", "num_reviews", 
                        "review_count", "minimum_nights", "maximum_occupancy"}
        searchable_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols and self.df[col].dtype == "object"
        ]
        
        # Handle list columns (like cuisines)
        for col in self.df.columns:
            if col not in exclude_cols:
                sample_val = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
                if isinstance(sample_val, list) or (isinstance(sample_val, str) and sample_val.startswith('[')):
                    try:
                        self.df[f"{col}_str"] = self.df[col].apply(
                            lambda x: " ".join(eval(x)) if isinstance(x, str) and x.startswith('[') 
                                      else (" ".join(x) if isinstance(x, list) else str(x))
                        )
                        if f"{col}_str" not in searchable_cols:
                            searchable_cols.append(f"{col}_str")
                    except:
                        pass
        
        self.df["search_area"] = (
            self.df[searchable_cols]
            .apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
            .str.lower()
        )
        
        # Tokenize documents for BM25
        self.tokenized_docs = [
            self._tokenize(text) for text in self.df["search_area"]
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Cache for unique values
        self._unique_values_cache = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Search travel items using BM25 and apply filters.
        
        Args:
            query: Natural language search query
            filters: Dictionary of filter criteria
            top_k: Number of top results to return
            
        Returns:
            DataFrame with search results sorted by relevance
        """
        if not query or not query.strip():
            results_df = self.df.copy()
        else:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            results_df = self.df.copy()
            results_df["bm25_score"] = scores
            results_df = results_df.sort_values("bm25_score", ascending=False)
            results_df = results_df.head(top_k)
        
        # Apply filters
        if filters:
            results_df = self._apply_filters(results_df, filters)
        
        return results_df.reset_index(drop=True)
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe."""
        filtered_df = df.copy()
        
        # Type filter (restaurant, attraction, accommodation) - required
        if "item_type" in filters and filters["item_type"]:
            filtered_df = filtered_df[filtered_df["item_type"] == filters["item_type"]]
        
        # City filter
        if "city" in filters and filters["city"]:
            filtered_df = filtered_df[filtered_df["city"] == filters["city"]]
        
        # Price filter (for restaurants and accommodations)
        if "max_price" in filters and filters["max_price"] is not None:
            item_type = filters.get("item_type", "")
            if item_type == "restaurant" and "average_cost" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["average_cost"] <= filters["max_price"]]
            elif item_type == "accommodation" and "price" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["price"] <= filters["max_price"]]
        
        # Rating filter
        if "min_rating" in filters and filters["min_rating"] is not None:
            item_type = filters.get("item_type", "")
            if item_type == "restaurant" and "aggregate_rating" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["aggregate_rating"] >= filters["min_rating"]]
            elif item_type == "accommodation" and "review_rate_number" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["review_rate_number"] >= filters["min_rating"]]
        
        # Attraction type filter (handles lists)
        if "attraction_type" in filters and filters["attraction_type"]:
            selected_type = filters["attraction_type"]
            # Check if the selected type is in the attraction_type list
            def has_attraction_type(row):
                types_list = row.get("attraction_type")
                if pd.isna(types_list):
                    return False
                if isinstance(types_list, list):
                    return selected_type in types_list
                elif isinstance(types_list, str):
                    # Try to parse if it's a string representation of a list
                    try:
                        import ast
                        parsed = ast.literal_eval(types_list)
                        if isinstance(parsed, list):
                            return selected_type in parsed
                        else:
                            return selected_type == types_list
                    except:
                        return selected_type == types_list
                else:
                    return selected_type == str(types_list)
            
            mask = filtered_df.apply(has_attraction_type, axis=1)
            filtered_df = filtered_df[mask]
        
        # Room type filter
        if "room_type" in filters and filters["room_type"]:
            filtered_df = filtered_df[filtered_df["room_type"] == filters["room_type"]]
        
        # Restaurant attributes filter (checkboxes)
        if "attributes" in filters and filters["attributes"]:
            item_type = filters.get("item_type", "")
            if item_type == "restaurant":
                # Filter restaurants that have ALL selected attributes set to True
                for attr in filters["attributes"]:
                    if attr in filtered_df.columns:
                        # Handle both boolean and string representations
                        filtered_df = filtered_df[
                            (filtered_df[attr] == True) | 
                            (filtered_df[attr] == "True") | 
                            (filtered_df[attr] == "true") |
                            (filtered_df[attr] == 1)
                        ]
        
        return filtered_df
    
    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column (with caching)."""
        if column not in self._unique_values_cache:
            if column in self.df.columns:
                unique_vals = sorted(self.df[column].dropna().unique().tolist())
                self._unique_values_cache[column] = unique_vals
            else:
                self._unique_values_cache[column] = []
        return self._unique_values_cache[column]
    
    def get_price_range(self) -> Tuple[float, float]:
        """Get min and max price across restaurants and accommodations."""
        prices = []
        if "average_cost" in self.df.columns:
            prices.extend(self.df["average_cost"].dropna().tolist())
        if "price" in self.df.columns:
            prices.extend(self.df["price"].dropna().tolist())
        if not prices:
            return 0.0, 1000.0
        return float(min(prices)), float(max(prices))


def render_restaurant_filters(searcher: TravelSearcher, city: Optional[str] = None) -> Dict[str, Any]:
    """Render filters specific to restaurants."""
    filters = {"item_type": "restaurant"}
    
    # City is pre-set, not user-selectable
    if city:
        filters["city"] = city
    
    restaurant_df = searcher.df[searcher.df["item_type"] == "restaurant"]
    if city:
        restaurant_df = restaurant_df[restaurant_df["city"] == city]
    
    with st.expander("Filters", expanded=True):
        # Max price filter (average_cost for restaurants)
        if not restaurant_df.empty and "average_cost" in restaurant_df.columns:
            cost_min = float(restaurant_df["average_cost"].min())
            cost_max = float(restaurant_df["average_cost"].max())
            max_cost = st.slider(
                "Max Cost ($)",
                min_value=float(cost_min),
                max_value=float(cost_max),
                value=float(cost_max),
                step=5.0,
                key="restaurant_max_cost_filter"
            )
            filters["max_price"] = max_cost
        
        # Min rating filter
        if not restaurant_df.empty and "aggregate_rating" in restaurant_df.columns:
            min_rating = st.slider(
                "Min Rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                key="restaurant_min_rating_filter"
            )
            if min_rating > 0:
                filters["min_rating"] = min_rating
        
        # Attributes checkboxes - split into two columns
        st.markdown("**Attributes:**")
        attribute_options = [
            ("good_for_groups", "Good for Groups"),
            ("good_for_kids", "Good for Kids"),
            ("has_takeout", "Has Takeout"),
            ("has_delivery", "Has Delivery"),
            ("has_reservations", "Takes Reservations"),
            ("has_outdoor_seating", "Outdoor Seating"),
            ("has_wifi", "Has WiFi"),
            ("wheelchair_accessible", "Wheelchair Accessible"),
            ("accepts_credit_card", "Accepts Credit Card"),
            ("has_table_service", "Table Service"),
        ]
        
        # Split attributes into two columns
        attr_col1, attr_col2 = st.columns(2)
        selected_attributes = []
        
        with attr_col1:
            for attr_key, attr_label in attribute_options[:5]:  # First 5 attributes
                if not restaurant_df.empty and attr_key in restaurant_df.columns:
                    if st.checkbox(attr_label, key=f"restaurant_attr_{attr_key}"):
                        selected_attributes.append(attr_key)
        
        with attr_col2:
            for attr_key, attr_label in attribute_options[5:]:  # Last 5 attributes
                if not restaurant_df.empty and attr_key in restaurant_df.columns:
                    if st.checkbox(attr_label, key=f"restaurant_attr_{attr_key}"):
                        selected_attributes.append(attr_key)
        
        if selected_attributes:
            filters["attributes"] = selected_attributes
    
    return filters


def render_attraction_filters(searcher: TravelSearcher, city: Optional[str] = None) -> Dict[str, Any]:
    """Render filters specific to attractions."""
    filters = {"item_type": "attraction"}
    
    # City is pre-set, not user-selectable
    if city:
        filters["city"] = city
    
    attraction_df = searcher.df[searcher.df["item_type"] == "attraction"]
    if city:
        attraction_df = attraction_df[attraction_df["city"] == city]
    
    with st.expander("Filters", expanded=True):
        # Attraction type filter
        if not attraction_df.empty and "attraction_type" in attraction_df.columns:
                # Extract all unique attraction types from lists
                all_types = set()
                for types_list in attraction_df["attraction_type"].dropna():
                    if isinstance(types_list, list):
                        all_types.update(types_list)
                    elif isinstance(types_list, str):
                        # Try to parse if it's a string representation of a list
                        try:
                            import ast
                            parsed = ast.literal_eval(types_list)
                            if isinstance(parsed, list):
                                all_types.update(parsed)
                            else:
                                all_types.add(types_list)
                        except:
                            all_types.add(types_list)
                    else:
                        all_types.add(str(types_list))
                
                if all_types:
                    attraction_types = sorted(list(all_types))
                    selected_type = st.selectbox(
                        "Attraction Type",
                        options=["All"] + attraction_types,
                        key="attraction_type_filter"
                    )
                    if selected_type != "All":
                        filters["attraction_type"] = selected_type
    
    return filters


def render_accommodation_filters(searcher: TravelSearcher, city: Optional[str] = None) -> Dict[str, Any]:
    """Render filters specific to accommodations."""
    filters = {"item_type": "accommodation"}
    
    # City is pre-set, not user-selectable
    if city:
        filters["city"] = city
    
    accommodation_df = searcher.df[searcher.df["item_type"] == "accommodation"]
    if city:
        accommodation_df = accommodation_df[accommodation_df["city"] == city]
    
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Max price filter (price for accommodations)
            if not accommodation_df.empty and "price" in accommodation_df.columns:
                price_min = float(accommodation_df["price"].min())
                price_max = float(accommodation_df["price"].max())
                max_price = st.slider(
                    "Max Price per Night ($)",
                    min_value=float(price_min),
                    max_value=float(price_max),
                    value=float(price_max),
                    step=10.0,
                    key="accommodation_max_price_filter"
                )
                filters["max_price"] = max_price
        
        with col2:
            # Room type filter
            if not accommodation_df.empty and "room_type" in accommodation_df.columns:
                room_types = sorted(accommodation_df["room_type"].dropna().unique().tolist())
                if room_types:
                    selected_room_type = st.selectbox(
                        "Room Type",
                        options=["All"] + room_types,
                        key="accommodation_room_type_filter"
                    )
                    if selected_room_type != "All":
                        filters["room_type"] = selected_room_type
    
    return filters


def _get_item_display_name(row: pd.Series) -> str:
    """Get display name for a travel item."""
    item_type = row.get("item_type", "unknown")
    name = row.get("name", "Unknown")
    
    if item_type == "restaurant":
        return f":material/restaurant: {name}"
    elif item_type == "attraction":
        icon = row.get("icon", ":material/attractions:")
        return f"{icon} {name}"
    elif item_type == "accommodation":
        return f":material/hotel: {name}"
    return name


def _get_item_details_markdown(row: pd.Series) -> str:
    """Get markdown details for a travel item."""
    item_type = row.get("item_type", "unknown")
    item_dict = row.to_dict()
    
    if item_type == "restaurant":
        return _restaurant_to_dialog_content(item_dict)
    elif item_type == "attraction":
        return _attraction_to_dialog_content(item_dict)
    elif item_type == "accommodation":
        return _accommodation_to_dialog_content(item_dict)
    return str(item_dict)


def render_travel_results(results_df: pd.DataFrame, travel_db: TravelDB, max_results: int = 50):
    """
    Render search results for travel items.
    
    Args:
        results_df: DataFrame with search results
        travel_db: TravelDB instance
        max_results: Maximum number of results to display
    """
    if results_df.empty:
        st.subheader("Search Results (0 items)")
        st.write("No items found matching your search criteria.")
        return
    
    # Initialize liked list in session state if not present
    if "liked_travel_items" not in st.session_state:
        st.session_state.liked_travel_items = set()
    
    num_results = min(len(results_df), max_results)
    st.subheader(f"Search Results ({num_results} of {len(results_df)} items)")
    
    # Use horizontal flex container that adapts to screen width
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for item_idx in range(num_results):
            row = results_df.iloc[item_idx]
            item_id = row["item_id"]
            item_type = row.get("item_type", "unknown")
            item_name = row.get("name", "Unknown")
            
            # Each item card container with fixed width and height
            with st.container(border=True, width=500, height=500):
                # Plus button to add to liked list (at the top)
                is_liked = item_id in st.session_state.liked_travel_items
                button_label = ":material/heart_check:" if is_liked else ":material/heart_plus:"
                button_type = "primary" if is_liked else "secondary"
                
                if st.button(
                    button_label,
                    key=f"like_button_{item_id}",
                    type=button_type,
                    use_container_width=True,
                    help="Add to liked list"
                ):
                    if is_liked:
                        st.session_state.liked_travel_items.remove(item_id)
                    else:
                        st.session_state.liked_travel_items.add(item_id)
                    st.rerun()
                
                # Item details
                display_name = _get_item_display_name(row)
                st.markdown(f"### {display_name}")
                
                # Basic info
                if item_type == "restaurant":
                    city = row.get("city", "Unknown")
                    cost = row.get("average_cost", 0)
                    rating = row.get("aggregate_rating", 0)
                    review_count = row.get("review_count", 0)
                    st.markdown(f"**City:** {city}")
                    if cost > 0:
                        st.markdown(f"**Cost:** {_get_dollar_signs(cost)}")
                    if rating > 0:
                        st.markdown(f"**Rating:** {rating:.1f}⭐")
                    if review_count > 0:
                        st.markdown(f"**Reviews:** {int(review_count)}")
                    
                    # Show restaurant attributes
                    attributes = []
                    attribute_labels = {
                        "good_for_groups": "Good for Groups",
                        "good_for_kids": "Good for Kids",
                        "has_takeout": "Takeout",
                        "has_delivery": "Delivery",
                        "has_reservations": "Reservations",
                        "has_outdoor_seating": "Outdoor Seating",
                        "has_wifi": "WiFi",
                        "wheelchair_accessible": "Wheelchair Accessible",
                        "accepts_credit_card": "Credit Cards",
                        "has_table_service": "Table Service",
                    }
                    
                    for attr_key, attr_label in attribute_labels.items():
                        if attr_key in row:
                            attr_value = row[attr_key]
                            # Check if attribute is True (handle different representations)
                            if (attr_value == True or attr_value == "True" or attr_value == "true" or attr_value == 1):
                                attributes.append(attr_label)
                    
                    if attributes:
                        st.markdown(f"**Features:** {', '.join(attributes)}")
                    
                    # Show cuisine/tags if available
                    if "tags" in row and pd.notna(row["tags"]):
                        try:
                            tags = row["tags"]
                            if isinstance(tags, str):
                                # Try to parse if it's a string representation
                                import ast
                                tags = ast.literal_eval(tags)
                            if isinstance(tags, list) and tags:
                                st.markdown(f"**Cuisine/Tags:** {', '.join(tags[:5])}")  # Show first 5 tags
                        except:
                            pass
                elif item_type == "attraction":
                    city = row.get("city", "Unknown")
                    attraction_type = row.get("attraction_type", "")
                    activity_level = row.get("activity_level", "")
                    st.markdown(f"**City:** {city}")
                    if attraction_type:
                        st.markdown(f"**Type:** {attraction_type}")
                    if activity_level:
                        st.markdown(f"**Activity Level:** {activity_level}")
                elif item_type == "accommodation":
                    city = row.get("city", "Unknown")
                    price = row.get("price", 0)
                    room_type = row.get("room_type", "")
                    st.markdown(f"**City:** {city}")
                    if price > 0:
                        st.markdown(f"**Price per night:** ${price:.2f}")
                    if room_type:
                        st.markdown(f"**Room Type:** {room_type}")
                
                # Description if available (no truncation for attractions)
                if "description" in row and pd.notna(row["description"]):
                    desc = str(row["description"])
                    st.markdown(f"*{desc}*")
                
                # BM25 score if available
                if "bm25_score" in row:
                    st.caption(f"Relevance Score: {row['bm25_score']:.4f}")


def render_liked_travel_items_footer(liked_list: List[str], travel_db: TravelDB, searcher: TravelSearcher):
    """
    Render a floating footer showing liked travel items with clickable buttons.
    
    Args:
        liked_list: List of item_ids that are liked
        travel_db: TravelDB instance for getting item information
        searcher: TravelSearcher instance for getting item details
    """
    if not liked_list:
        return
    
    # Inject CSS for floating footer using the container key
    st.markdown("""
    <style>
    /* Style the footer container using its key */
    .st-key-liked-travel-items-footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-color: #fff !important;
        border-top: 2px solid #EAC9BD !important;
        padding: 1rem !important;
        z-index: 999999 !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1) !important;
        max-height: 200px !important;
        overflow-y: auto !important;
    }
    /* Add bottom padding to main content area when footer is present */
    .main .block-container {
        padding-bottom: 220px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create footer container with a key for CSS styling
    with st.container(key="liked-travel-items-footer", border=False):
        st.markdown(f":material/favorite: Liked Items ({len(liked_list)})")
        
        # Use horizontal flex container for buttons
        with st.container(horizontal=True, horizontal_alignment="left", gap="small"):
            for item_id in liked_list[:20]:  # Limit to first 20 to prevent wrapping
                try:
                    # Find the item in the searcher's dataframe
                    item_row = searcher.df[searcher.df["item_id"] == item_id]
                    if item_row.empty:
                        st.button(
                            f"{item_id} (not found)",
                            key=f"footer_like_{item_id}",
                            disabled=True,
                            width=200
                        )
                        continue
                    
                    row = item_row.iloc[0]
                    item_type = row.get("item_type", "unknown")
                    item_name = row.get("name", "Unknown")
                    city = row.get("city", "")
                    
                    # Truncate long names
                    display_name = item_name
                    if len(display_name) > 30:
                        display_name = display_name[:27] + "..."
                    
                    # Add type icon
                    if item_type == "restaurant":
                        display_name = f":material/restaurant: {display_name}"
                    elif item_type == "attraction":
                        icon = row.get("icon", ":material/attractions:")
                        display_name = f"{icon} {display_name}"
                    elif item_type == "accommodation":
                        display_name = f":material/hotel: {display_name}"
                    
                    # Create dialog function for this item
                    def make_dialog(item_row: pd.Series):
                        item_dict = item_row.to_dict()
                        item_type = item_dict.get("item_type", "unknown")
                        item_name = item_dict.get("name", "Unknown")
                        
                        @st.dialog(f"{item_name} ({item_type})", width="large")
                        def show_item_dialog():
                            if item_type == "restaurant":
                                st.markdown(_restaurant_to_dialog_content(item_dict), unsafe_allow_html=True)
                            elif item_type == "attraction":
                                st.markdown(_attraction_to_dialog_content(item_dict), unsafe_allow_html=True)
                            elif item_type == "accommodation":
                                st.markdown(_accommodation_to_dialog_content(item_dict), unsafe_allow_html=True)
                            
                            # Remove from liked list button
                            item_id = item_dict.get("item_id")
                            if st.button("Remove from Liked", type="primary", key=f"remove_{item_id}"):
                                st.session_state.liked_travel_items.remove(item_id)
                                st.rerun()
                        
                        return show_item_dialog
                    
                    dialog_fn = make_dialog(row)
                    
                    st.button(
                        display_name,
                        key=f"footer_like_{item_id}",
                        on_click=dialog_fn,
                        width=200,
                        type="primary"
                    )
                except Exception as e:
                    # Item not found - show disabled button
                    st.button(
                        f"{item_id} (error)",
                        key=f"footer_like_{item_id}",
                        disabled=True,
                        width=200
                    )
        
        if len(liked_list) > 20:
            st.caption(f"... and {len(liked_list) - 20} more items")


def render_search_interface(travel_db: Optional[TravelDB] = None, city: Optional[str] = None, num_results: int = 30):
    """
    Main function to render the complete travel search interface as a reusable component.
    
    Args:
        travel_db: Optional TravelDB instance. If None, creates a new one.
        city: Required city name to filter all searches. If None, searches all cities.
        num_results: Fixed number of results to show (default: 21)
    """
    if travel_db is None:
        travel_db = TravelDB()
    
    # Initialize searcher
    if "travel_searcher" not in st.session_state:
        with st.spinner("Initializing search index..."):
            st.session_state.travel_searcher = TravelSearcher(travel_db)
    
    searcher = st.session_state.travel_searcher
    
    if searcher.df.empty:
        st.error("No travel data available. Please check the database.")
        return
    
    # Initialize liked items list
    if "liked_travel_items" not in st.session_state:
        st.session_state.liked_travel_items = set()
    
    # Main search interface
    if city:
        st.markdown(f"Search through restaurants, attractions, and accommodations in **{city}**, and add items you like to your liked list.")
    else:
        st.markdown("Search through restaurants, attractions, and accommodations, and add items you like to your liked list.")
    
    # Create tabs for each item type
    tab1, tab2, tab3 = st.tabs([":material/restaurant: Restaurants", ":material/attractions: Attractions", ":material/hotel: Accommodations"])
    
    # Initialize session state for each tab
    for tab_name in ["restaurants", "attractions", "accommodations"]:
        if f"last_{tab_name}_search_query" not in st.session_state:
            st.session_state[f"last_{tab_name}_search_query"] = None
        if f"last_{tab_name}_search_results" not in st.session_state:
            st.session_state[f"last_{tab_name}_search_results"] = None
        if f"last_{tab_name}_search_filters" not in st.session_state:
            st.session_state[f"last_{tab_name}_search_filters"] = None
    
    # Restaurant tab
    with tab1:
        filters = render_restaurant_filters(searcher, city=city)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                ":material/search: Search Query",
                placeholder="e.g., 'Italian restaurant', 'pizza', 'sushi'",
                key="restaurant_search_query",
                label_visibility="collapsed"
            )
        with col2:
            search_button = st.button("Search", type="primary", use_container_width=True, key="restaurant_search_button")
        
        should_search = search_button or (query and query != st.session_state.last_restaurants_search_query)
        
        if should_search:
            with st.spinner("Searching restaurants..."):
                results_df = searcher.search(query, filters=filters, top_k=100)
                st.session_state.last_restaurants_search_query = query
                st.session_state.last_restaurants_search_results = results_df
                st.session_state.last_restaurants_search_filters = filters
        elif st.session_state.last_restaurants_search_results is not None:
            results_df = st.session_state.last_restaurants_search_results
            if filters != st.session_state.last_restaurants_search_filters:
                results_df = searcher.search(
                    st.session_state.last_restaurants_search_query or "",
                    filters=filters,
                    top_k=100
                )
                st.session_state.last_restaurants_search_results = results_df
                st.session_state.last_restaurants_search_filters = filters
        else:
            results_df = None
        
        if results_df is not None:
            render_travel_results(results_df, travel_db, max_results=num_results)
    
    # Attraction tab
    with tab2:
        filters = render_attraction_filters(searcher, city=city)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                ":material/search: Search Query",
                placeholder="e.g., 'museum', 'park', 'zoo'",
                key="attraction_search_query",
                label_visibility="collapsed"
            )
        with col2:
            search_button = st.button("Search", type="primary", use_container_width=True, key="attraction_search_button")
        
        should_search = search_button or (query and query != st.session_state.last_attractions_search_query)
        
        if should_search:
            with st.spinner("Searching attractions..."):
                results_df = searcher.search(query, filters=filters, top_k=100)
                st.session_state.last_attractions_search_query = query
                st.session_state.last_attractions_search_results = results_df
                st.session_state.last_attractions_search_filters = filters
        elif st.session_state.last_attractions_search_results is not None:
            results_df = st.session_state.last_attractions_search_results
            if filters != st.session_state.last_attractions_search_filters:
                results_df = searcher.search(
                    st.session_state.last_attractions_search_query or "",
                    filters=filters,
                    top_k=100
                )
                st.session_state.last_attractions_search_results = results_df
                st.session_state.last_attractions_search_filters = filters
        else:
            results_df = None
        
        if results_df is not None:
            render_travel_results(results_df, travel_db, max_results=num_results)
    
    # Accommodation tab
    with tab3:
        filters = render_accommodation_filters(searcher, city=city)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                ":material/search: Search Query",
                placeholder="e.g., 'budget hotel', 'apartment', 'downtown'",
                key="accommodation_search_query",
                label_visibility="collapsed"
            )
        with col2:
            search_button = st.button("Search", type="primary", use_container_width=True, key="accommodation_search_button")
        
        should_search = search_button or (query and query != st.session_state.last_accommodations_search_query)
        
        if should_search:
            with st.spinner("Searching accommodations..."):
                results_df = searcher.search(query, filters=filters, top_k=100)
                st.session_state.last_accommodations_search_query = query
                st.session_state.last_accommodations_search_results = results_df
                st.session_state.last_accommodations_search_filters = filters
        elif st.session_state.last_accommodations_search_results is not None:
            results_df = st.session_state.last_accommodations_search_results
            if filters != st.session_state.last_accommodations_search_filters:
                results_df = searcher.search(
                    st.session_state.last_accommodations_search_query or "",
                    filters=filters,
                    top_k=100
                )
                st.session_state.last_accommodations_search_results = results_df
                st.session_state.last_accommodations_search_filters = filters
        else:
            results_df = None
        
        if results_df is not None:
            render_travel_results(results_df, travel_db, max_results=num_results)
    
    # Display liked items in a floating footer at the bottom (shared across all tabs)
    if st.session_state.liked_travel_items:
        liked_list = sorted(list(st.session_state.liked_travel_items))
        render_liked_travel_items_footer(liked_list, travel_db, searcher)


def render_liked_items(liked_items: set, travel_db: Optional[TravelDB] = None):
    """
    Render liked items in travel item cards for the final comparison.
    
    Args:
        liked_items: Set of item IDs that are liked
        travel_db: Optional TravelDB instance. If None, creates a new one.
    """
    if travel_db is None:
        travel_db = TravelDB()
    
    if not liked_items or len(liked_items) == 0:
        st.markdown("*No items were liked during exploration.*")
        return
    
    st.markdown(f"You liked {len(liked_items)} item(s) during exploration.")
    liked_list = list(liked_items)
    
    # Get all items from travel_db
    all_items_df = pd.concat([
        travel_db.restaurants_df.assign(item_type="restaurant"),
        travel_db.attractions_df.assign(item_type="attraction"),
        travel_db.accommodations_df.assign(item_type="accommodation")
    ], ignore_index=True)
    
    # Render liked travel items in cards
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for item_id in liked_list:
            try:
                item_row = all_items_df[all_items_df["item_id"] == item_id]
                if item_row.empty:
                    st.write(f"Item {item_id} (not found)")
                    continue
                
                row = item_row.iloc[0]
                item_type = row.get("item_type", "unknown")
                
                with st.container(border=True, width=500, height=500):
                    # Item details
                    display_name = _get_item_display_name(row)
                    st.markdown(f"### {display_name}")
                    
                    # Basic info
                    if item_type == "restaurant":
                        city = row.get("city", "Unknown")
                        cost = row.get("average_cost", 0)
                        rating = row.get("aggregate_rating", 0)
                        review_count = row.get("review_count", 0)
                        st.markdown(f"**City:** {city}")
                        if cost > 0:
                            st.markdown(f"**Cost:** {_get_dollar_signs(cost)}")
                        if rating > 0:
                            st.markdown(f"**Rating:** {rating:.1f}⭐")
                        if review_count > 0:
                            st.markdown(f"**Reviews:** {int(review_count)}")
                        
                        # Show restaurant attributes
                        attributes = []
                        attribute_labels = {
                            "good_for_groups": "Good for Groups",
                            "good_for_kids": "Good for Kids",
                            "has_takeout": "Takeout",
                            "has_delivery": "Delivery",
                            "has_reservations": "Reservations",
                            "has_outdoor_seating": "Outdoor Seating",
                            "has_wifi": "WiFi",
                            "wheelchair_accessible": "Wheelchair Accessible",
                            "accepts_credit_card": "Credit Cards",
                            "has_table_service": "Table Service",
                        }
                        
                        for attr_key, attr_label in attribute_labels.items():
                            if attr_key in row:
                                attr_value = row[attr_key]
                                if (attr_value is True or attr_value == "True" or attr_value == "true" or attr_value == 1):
                                    attributes.append(attr_label)
                        
                        if attributes:
                            st.markdown(f"**Features:** {', '.join(attributes)}")
                        
                        # Show cuisine/tags if available
                        if "tags" in row and pd.notna(row["tags"]):
                            try:
                                tags = row["tags"]
                                if isinstance(tags, str):
                                    import ast
                                    tags = ast.literal_eval(tags)
                                if isinstance(tags, list) and len(tags) > 0:
                                    tags_display = ", ".join(tags[:5])
                                    st.markdown(f"**Cuisine/Tags:** {tags_display}")
                            except Exception:
                                pass
                    
                    elif item_type == "attraction":
                        city = row.get("city", "Unknown")
                        st.markdown(f"**City:** {city}")
                        if "description" in row and pd.notna(row["description"]):
                            st.markdown(f"**Description:** {row['description']}")
                    
                    elif item_type == "accommodation":
                        city = row.get("city", "Unknown")
                        price = row.get("price_per_night", 0)
                        room_type = row.get("room_type", "Unknown")
                        st.markdown(f"**City:** {city}")
                        if price > 0:
                            st.markdown(f"**Price per Night:** ${price:.2f}")
                        st.markdown(f"**Room Type:** {room_type}")
            except (ValueError, KeyError, AttributeError, TypeError):
                st.write(f"Item {item_id} (error loading)")

