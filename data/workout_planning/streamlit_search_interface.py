"""
Streamlit search interface components for the workout planning exercise database.

Provides:
- BM25-based natural language search for exercises
- Filter components (difficulty, equipment, muscle group, etc.)
- Search results display
- Liked exercises footer

Dependencies:
- rank_bm25: Install with `pip install rank-bm25`
"""

from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "liked_exercises",
    "exercise_searcher",
    "last_exercise_search_query",
    "last_exercise_search_results",
    "last_exercise_search_filters",
]

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required. Install it with: pip install rank-bm25"
    )
import re
from data.workout_planning.db import ExerciseDB
from data.workout_planning.streamlit_render import _render_exercise_details


class ExerciseSearcher:
    """Handles BM25 search and filtering for exercises."""
    
    def __init__(self, exercise_db: ExerciseDB):
        self.exercise_db = exercise_db
        self.df = exercise_db.tables['exercises'].copy()
        
        # Create search_area with all text columns for indexing
        exclude_cols = {
            "num_primary_items", "num_secondary_items", "num_sets", 
            "time_per_set", "rest_time", "total_time_seconds", "num_reps_per_set"
        }
        searchable_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols and self.df[col].dtype == "object"
        ]
        
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
        Search exercises using BM25 and apply filters.
        
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
        
        # Difficulty level filter (supports multiple values)
        if "difficulty_level" in filters and filters["difficulty_level"]:
            selected_levels = filters["difficulty_level"]
            if isinstance(selected_levels, list):
                filtered_df = filtered_df[filtered_df["difficulty_level"].isin(selected_levels)]
            else:
                filtered_df = filtered_df[filtered_df["difficulty_level"] == selected_levels]
        
        # Primary equipment filter (supports multiple values)
        if "primary_equipment" in filters and filters["primary_equipment"]:
            selected_equipment = filters["primary_equipment"]
            if isinstance(selected_equipment, list):
                filtered_df = filtered_df[filtered_df["primary_equipment"].isin(selected_equipment)]
            else:
                filtered_df = filtered_df[filtered_df["primary_equipment"] == selected_equipment]
        
        # Target muscle group filter (supports multiple values)
        if "target_muscle_group" in filters and filters["target_muscle_group"]:
            selected_muscles = filters["target_muscle_group"]
            if isinstance(selected_muscles, list):
                filtered_df = filtered_df[filtered_df["target_muscle_group"].isin(selected_muscles)]
            else:
                filtered_df = filtered_df[filtered_df["target_muscle_group"] == selected_muscles]
        
        # Body region filter (supports multiple values)
        if "body_region" in filters and filters["body_region"]:
            selected_regions = filters["body_region"]
            if isinstance(selected_regions, list):
                filtered_df = filtered_df[filtered_df["body_region"].isin(selected_regions)]
            else:
                filtered_df = filtered_df[filtered_df["body_region"] == selected_regions]
        
        # Primary exercise classification filter (supports multiple values)
        if "primary_exercise_classification" in filters and filters["primary_exercise_classification"]:
            selected_classifications = filters["primary_exercise_classification"]
            if isinstance(selected_classifications, list):
                filtered_df = filtered_df[filtered_df["primary_exercise_classification"].isin(selected_classifications)]
            else:
                filtered_df = filtered_df[filtered_df["primary_exercise_classification"] == selected_classifications]
        
        # Max total time filter (in seconds)
        if "max_total_time_seconds" in filters and filters["max_total_time_seconds"] is not None:
            filtered_df = filtered_df[filtered_df["total_time_seconds"] <= filters["max_total_time_seconds"]]
        
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
    
    def get_total_time_range(self) -> Tuple[float, float]:
        """Get min and max total_time_seconds from exercises."""
        return float(self.df["total_time_seconds"].min()), float(self.df["total_time_seconds"].max())


def render_exercise_filters(searcher: ExerciseSearcher) -> Dict[str, Any]:
    """
    Render filter UI components in an expandable and return filter values.
    
    Returns:
        Dictionary of filter values
    """
    filters = {}
    
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Difficulty level filter (multiselect)
            difficulty_levels = searcher.get_unique_values("difficulty_level")
            if difficulty_levels:
                selected_difficulties = st.multiselect(
                    "Select Difficulty Level(s)",
                    options=difficulty_levels,
                    key="difficulty_filter"
                )
                if selected_difficulties:
                    filters["difficulty_level"] = selected_difficulties
            
            # Target muscle group filter (multiselect)
            muscle_groups = searcher.get_unique_values("target_muscle_group")
            if muscle_groups:
                selected_muscles = st.multiselect(
                    "Select Target Muscle Group(s)",
                    options=muscle_groups,
                    key="muscle_group_filter"
                )
                if selected_muscles:
                    filters["target_muscle_group"] = selected_muscles
        
        with col2:
            # Primary equipment filter (multiselect)
            equipment = searcher.get_unique_values("primary_equipment")
            if equipment:
                selected_equipment = st.multiselect(
                    "Select Primary Equipment",
                    options=equipment,
                    key="equipment_filter"
                )
                if selected_equipment:
                    filters["primary_equipment"] = selected_equipment
            
            # Body region filter (multiselect)
            body_regions = searcher.get_unique_values("body_region")
            if body_regions:
                selected_regions = st.multiselect(
                    "Select Body Region(s)",
                    options=body_regions,
                    key="body_region_filter"
                )
                if selected_regions:
                    filters["body_region"] = selected_regions
        
        with col3:
            # Primary exercise classification filter (multiselect)
            classifications = searcher.get_unique_values("primary_exercise_classification")
            if classifications:
                selected_classifications = st.multiselect(
                    "Select Exercise Type(s)",
                    options=classifications,
                    key="classification_filter"
                )
                if selected_classifications:
                    filters["primary_exercise_classification"] = selected_classifications
            
            # Max total time filter (in minutes)
            time_min, time_max = searcher.get_total_time_range()
            max_time_minutes = st.slider(
                "Max Total Time (minutes)",
                min_value=int(time_min / 60),
                max_value=int(time_max / 60),
                value=int(time_max / 60),
                step=5,
                key="max_time_filter"
            )
            filters["max_total_time_seconds"] = max_time_minutes * 60
    
    return filters


def render_exercise_results(results_df: pd.DataFrame, exercise_db: ExerciseDB, max_results: int = 50):
    """
    Render search results for exercises, grouped by exercise name.
    
    Args:
        results_df: DataFrame with search results
        exercise_db: ExerciseDB instance
        max_results: Maximum number of unique exercises to display
    """
    if results_df.empty:
        st.subheader("Search Results (0 exercises)")
        st.write("No exercises found matching your search criteria.")
        return
    
    # Initialize liked list in session state if not present
    if "liked_exercises" not in st.session_state:
        st.session_state.liked_exercises = set()
    
    # Group by exercise_name to get unique exercises
    unique_exercises = results_df.groupby("exercise_name").apply(
        lambda group: group.to_dict("records")
    ).to_dict()
    
    num_results = min(len(unique_exercises), max_results)
    st.subheader(f"Search Results ({num_results} unique exercises)")
    
    # Use horizontal flex container that adapts to screen width
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for exercise_idx, (exercise_name, variations) in enumerate(list(unique_exercises.items())[:num_results]):
            # Use exercise_name as the identifier (not variation-specific)
            exercise_id = exercise_name
            
            # Each exercise card container with fixed width and height
            with st.container(border=True, width=500, height=700):
                # Plus button to add to liked list (at the top)
                is_liked = exercise_id in st.session_state.liked_exercises
                button_label = ":material/heart_check:" if is_liked else ":material/heart_plus:"
                button_type = "primary" if is_liked else "secondary"
                
                if st.button(
                    button_label,
                    key=f"like_button_{exercise_id}_{exercise_idx}",
                    type=button_type,
                    use_container_width=True,
                    help="Add to liked list"
                ):
                    if is_liked:
                        st.session_state.liked_exercises.remove(exercise_id)
                    else:
                        st.session_state.liked_exercises.add(exercise_id)
                    st.rerun()
            
                # Render exercise with all variations
                _render_exercise_with_variations(exercise_name, variations, exercise_idx)


def _render_exercise_with_variations(exercise_name: str, variations: List[Dict], exercise_idx: int):
    """
    Render an exercise card showing the exercise name and all its variations.
    
    Args:
        exercise_name: Name of the exercise
        variations: List of variation dictionaries
        exercise_idx: Index for unique keys
    """
    # Show exercise name as header
    st.markdown(f"### {exercise_name}")
    
    # Show number of variations
    st.markdown(f"*{len(variations)} variation(s) available*")
    
    # Use tabs to show different variations
    if len(variations) > 1:
        tab_names = [f"Variation {i+1}" for i in range(len(variations))]
        tabs = st.tabs(tab_names)
        for i, (tab, variation) in enumerate(zip(tabs, variations)):
            with tab:
                exercise_markdown = _render_exercise_details(i, variation)
                st.markdown(exercise_markdown, unsafe_allow_html=True)
    else:
        # Single variation, just show it directly
        exercise_markdown = _render_exercise_details(0, variations[0])
        st.markdown(exercise_markdown, unsafe_allow_html=True)


def render_liked_exercises_footer(liked_list: List[str], exercise_db: ExerciseDB):
    """
    Render a floating footer showing liked exercises with clickable buttons.
    
    Args:
        liked_list: List of exercise names that are liked
        exercise_db: ExerciseDB instance for getting exercise information
    """
    if not liked_list:
        return
    
    # Inject CSS for floating footer using the container key
    st.markdown("""
    <style>
    /* Style the footer container using its key */
    .st-key-liked-exercises-footer {
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
    with st.container(key="liked-exercises-footer", border=False):
        st.markdown(f":material/favorite: Liked Exercises ({len(liked_list)})")
        
        # Use horizontal flex container for buttons
        with st.container(horizontal=True, horizontal_alignment="left", gap="small"):
            for exercise_id in liked_list[:20]:  # Limit to first 20 to prevent wrapping
                try:
                    # exercise_id is now just the exercise_name (no variation)
                    exercise_name = exercise_id
                    
                    # Get all variations of this exercise
                    all_exercises = exercise_db.get_all_exercises_by_name(exercise_name)
                    
                    if not all_exercises:
                        st.button(
                            f"{exercise_name} (not found)",
                            key=f"footer_like_{exercise_id}",
                            disabled=True,
                            width=200
                        )
                        continue
                    
                    # Truncate long exercise names
                    display_name = exercise_name
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."
                    
                    # Create dialog function for this exercise with all variations
                    def make_dialog(exercise_name: str, variations: List[Dict], exercise_id: str):
                        @st.dialog(f"{exercise_name} - {len(variations)} Variations", width="large")
                        def show_exercise_dialog():
                            if len(variations) > 1:
                                tabs = st.tabs([f"Variation {i+1}" for i in range(len(variations))])
                                for i, (tab, variation) in enumerate(zip(tabs, variations)):
                                    with tab:
                                        st.markdown(_render_exercise_details(i, variation), unsafe_allow_html=True)
                            else:
                                st.markdown(_render_exercise_details(0, variations[0]), unsafe_allow_html=True)
                            
                            # Remove from liked list button
                            if st.button("Remove from Liked", type="primary", key=f"remove_{exercise_id}"):
                                st.session_state.liked_exercises.remove(exercise_id)
                                st.rerun()
                        
                        return show_exercise_dialog
                    
                    dialog_fn = make_dialog(exercise_name, all_exercises, exercise_id)
                    
                    st.button(
                        display_name,
                        key=f"footer_like_{exercise_id}",
                        on_click=dialog_fn,
                        width=200,
                        type="primary"
                    )
                except Exception:
                    # Exercise not found - show disabled button
                    st.button(
                        f"{exercise_id} (error)",
                        key=f"footer_like_{exercise_id}",
                        disabled=True,
                        width=200
                    )
        
        if len(liked_list) > 20:
            st.caption(f"... and {len(liked_list) - 20} more exercises")


def render_search_interface(exercise_db: Optional[ExerciseDB] = None, num_results: int = 30):
    """
    Main function to render the complete exercise search interface as a reusable component.
    
    Args:
        exercise_db: Optional ExerciseDB instance. If None, creates a new one.
        num_results: Fixed number of results to show (default: 21)
    """
    if exercise_db is None:
        exercise_db = ExerciseDB()
    
    # Initialize searcher
    if "exercise_searcher" not in st.session_state:
        with st.spinner("Initializing search index..."):
            st.session_state.exercise_searcher = ExerciseSearcher(exercise_db)
    
    searcher = st.session_state.exercise_searcher
    
    # Initialize liked exercises list
    if "liked_exercises" not in st.session_state:
        st.session_state.liked_exercises = set()
    
    # Main search interface
    st.markdown("Search through the exercise database. Find exercises you like and add them to your liked list.")
    
    # Render filters in expandable at the top
    filters = render_exercise_filters(searcher)
    
    # Search query input and button
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            ":material/search: Search Query",
            placeholder="Search query e.g., 'push up', 'core strength', 'bodyweight legs'",
            key="exercise_search_query",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Store search state in session state
    if "last_exercise_search_query" not in st.session_state:
        st.session_state.last_exercise_search_query = None
    if "last_exercise_search_results" not in st.session_state:
        st.session_state.last_exercise_search_results = None
    if "last_exercise_search_filters" not in st.session_state:
        st.session_state.last_exercise_search_filters = None
    
    # Perform search if query changed or search button clicked
    should_search = search_button or (query and query != st.session_state.last_exercise_search_query)
    
    if should_search:
        with st.spinner("Searching exercises..."):
            results_df = searcher.search(query, filters=filters, top_k=100)
            st.session_state.last_exercise_search_query = query
            st.session_state.last_exercise_search_results = results_df
            st.session_state.last_exercise_search_filters = filters
    elif st.session_state.last_exercise_search_results is not None:
        results_df = st.session_state.last_exercise_search_results
        if filters != st.session_state.last_exercise_search_filters:
            results_df = searcher.search(
                st.session_state.last_exercise_search_query or "",
                filters=filters,
                top_k=100
            )
            st.session_state.last_exercise_search_results = results_df
            st.session_state.last_exercise_search_filters = filters
    else:
        results_df = None
    
    # Display results if available
    if results_df is not None:
        render_exercise_results(results_df, exercise_db, max_results=num_results)
    
    # Display liked exercises in a floating footer at the bottom
    if st.session_state.liked_exercises:
        liked_list = sorted(list(st.session_state.liked_exercises))
        render_liked_exercises_footer(liked_list, exercise_db)


def render_liked_items(liked_items: set, exercise_db: Optional[ExerciseDB] = None):
    """
    Render liked items in exercise cards for the final comparison.
    
    Args:
        liked_items: Set of exercise IDs (exercise_name::variation_name) that are liked
        exercise_db: Optional ExerciseDB instance. If None, creates a new one.
    """
    if exercise_db is None:
        exercise_db = ExerciseDB()
    
    if not liked_items or len(liked_items) == 0:
        st.markdown("*No items were liked during exploration.*")
        return
    
    st.markdown(f"You liked {len(liked_items)} item(s) during exploration.")
    liked_list = list(liked_items)
    
    # Render liked exercises in cards
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for exercise_id in liked_list:
            try:
                # exercise_id is now just the exercise_name (no variation)
                exercise_name = exercise_id
                
                # Get all variations of this exercise
                all_exercises = exercise_db.get_all_exercises_by_name(exercise_name)
                
                if all_exercises:
                    with st.container(border=True, width=600, height=700):
                        _render_exercise_with_variations(exercise_name, all_exercises, 0)
                else:
                    st.write(f"Exercise: {exercise_id} (not found)")
            except (ValueError, AttributeError, TypeError):
                st.write(f"Exercise: {exercise_id} (error loading)")

