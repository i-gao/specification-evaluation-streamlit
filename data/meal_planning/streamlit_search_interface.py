"""
Streamlit search interface components for the meal planning recipe database.

Provides:
- BM25-based natural language search for recipes
- Filter components (cuisine, diet, food type, etc.)
- Search results display
- Liked recipes footer

Dependencies:
- rank_bm25: Install with `pip install rank-bm25`
"""

from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "liked_recipes",
    "recipe_searcher",
    "last_recipe_search_query",
    "last_recipe_search_results",
    "last_recipe_search_filters",
]

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is required. Install it with: pip install rank-bm25")
import re
from data.meal_planning.db import RecipeDB, Recipe
from data.meal_planning.streamlit_render import _recipe_details


class RecipeSearcher:
    """Handles BM25 search and filtering for recipes."""

    def __init__(self, recipe_db: RecipeDB):
        self.recipe_db = recipe_db
        self.df = recipe_db.tables["recipes"].copy()

        # Create search_area with all text columns for indexing
        exclude_cols = {
            "rating",
            "num_reviews",
            "calories",
            "protein",
            "total_fat",
            "total_carbohydrate",
            "total_time",
            "num_servings",
        }
        searchable_cols = [
            col
            for col in self.df.columns
            if col not in exclude_cols and self.df[col].dtype == "object"
        ]
        # Also include list columns as strings
        for col in ["ingredients", "instructions", "diet", "intolerances", "equipment"]:
            if col in self.df.columns:
                self.df[f"{col}_str"] = self.df[col].apply(
                    lambda x: " ".join(x) if isinstance(x, list) else str(x)
                )
                searchable_cols.append(f"{col}_str")

        self.df["search_area"] = (
            self.df[searchable_cols]
            .apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
            .str.lower()
        )

        # Tokenize documents for BM25
        self.tokenized_docs = [self._tokenize(text) for text in self.df["search_area"]]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Cache for unique values
        self._unique_values_cache = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 20
    ) -> pd.DataFrame:
        """
        Search recipes using BM25 and apply filters.

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

        # Cuisine filter
        if "cuisine" in filters and filters["cuisine"]:
            filtered_df = filtered_df[filtered_df["cuisine"] == filters["cuisine"]]

        # Food type filter
        if "food_type" in filters and filters["food_type"]:
            filtered_df = filtered_df[filtered_df["food_type"] == filters["food_type"]]

        # Diet filter (check if diet list contains the filter value)
        if "diet" in filters and filters["diet"]:
            filtered_df = filtered_df[
                filtered_df["diet"].apply(
                    lambda x: filters["diet"] in x if isinstance(x, list) else False
                )
            ]

        # Time filter
        if "max_time" in filters and filters["max_time"] is not None:
            filtered_df = filtered_df[filtered_df["total_time"] <= filters["max_time"]]

        # Calories filter
        if "max_calories" in filters and filters["max_calories"] is not None:
            filtered_df = filtered_df[
                filtered_df["calories"] <= filters["max_calories"]
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

    def get_time_range(self) -> Tuple[int, int]:
        """Get min and max total_time from recipes."""
        return int(self.df["total_time"].min()), int(self.df["total_time"].max())

    def get_calories_range(self) -> Tuple[float, float]:
        """Get min and max calories from recipes."""
        return float(self.df["calories"].min()), float(self.df["calories"].max())


def render_recipe_filters(searcher: RecipeSearcher) -> Dict[str, Any]:
    """
    Render filter UI components in an expandable and return filter values.

    Returns:
        Dictionary of filter values
    """
    filters = {}

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Cuisine filter
            cuisines = searcher.get_unique_values("cuisine")
            if cuisines:
                selected_cuisine = st.selectbox(
                    "Select Cuisine", options=["All"] + cuisines, key="cuisine_filter"
                )
                if selected_cuisine != "All":
                    filters["cuisine"] = selected_cuisine

            # Food type filter
            food_types = searcher.get_unique_values("food_type")
            if food_types:
                selected_food_type = st.selectbox(
                    "Select Food Type",
                    options=["All"] + [ft for ft in food_types if ft],
                    key="food_type_filter",
                )
                if selected_food_type != "All":
                    filters["food_type"] = selected_food_type

        with col2:
            # Diet filter
            all_diets = set()
            for diet_list in searcher.df["diet"]:
                if isinstance(diet_list, list):
                    all_diets.update(diet_list)
            if all_diets:
                selected_diet = st.selectbox(
                    "Select Diet",
                    options=["All"] + sorted(list(all_diets)),
                    key="diet_filter",
                )
                if selected_diet != "All":
                    filters["diet"] = selected_diet

            # Max time filter
            time_min, time_max = searcher.get_time_range()
            max_time = st.slider(
                "Max Cooking Time (minutes)",
                min_value=int(time_min),
                max_value=int(time_max),
                value=int(time_max),
                step=15,
                key="max_time_filter",
            )
            filters["max_time"] = max_time

        with col3:
            # Max calories filter
            cal_min, cal_max = searcher.get_calories_range()
            max_calories = st.slider(
                "Max Calories",
                min_value=float(cal_min),
                max_value=float(cal_max),
                value=float(cal_max),
                step=50.0,
                key="max_calories_filter",
            )
            filters["max_calories"] = max_calories

    return filters


def render_recipe_results(
    results_df: pd.DataFrame, recipe_db: RecipeDB, max_results: int = 50
):
    """
    Render search results for recipes.

    Args:
        results_df: DataFrame with search results
        recipe_db: RecipeDB instance
        max_results: Maximum number of results to display
    """
    if results_df.empty:
        st.subheader("Search Results (0 recipes)")
        st.write("No recipes found matching your search criteria.")
        return

    # Initialize liked list in session state if not present
    if "liked_recipes" not in st.session_state:
        st.session_state.liked_recipes = set()

    num_results = min(len(results_df), max_results)
    st.subheader(f"Search Results ({num_results} of {len(results_df)} recipes)")

    # Use horizontal flex container that adapts to screen width
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for product_idx in range(num_results):
            row = results_df.iloc[product_idx]
            # Filter row dict to only include valid Recipe fields
            row_dict = row.to_dict()
            # Get valid Recipe field names
            recipe_fields = {f.name for f in Recipe.__dataclass_fields__.values()}
            filtered_dict = {k: v for k, v in row_dict.items() if k in recipe_fields}
            recipe = Recipe(**filtered_dict)
            recipe_title = recipe.title

            # Each recipe card container with fixed width and height
            with st.container(border=True, width=500, height=700):
                # Plus button to add to liked list (at the top)
                is_liked = recipe_title in st.session_state.liked_recipes
                button_label = (
                    ":material/heart_check:" if is_liked else ":material/heart_plus:"
                )
                button_type = "primary" if is_liked else "secondary"

                if st.button(
                    button_label,
                    key=f"like_button_{recipe_title}",
                    type=button_type,
                    use_container_width=True,
                    help="Add to liked list",
                ):
                    if is_liked:
                        st.session_state.liked_recipes.remove(recipe_title)
                    else:
                        st.session_state.liked_recipes.add(recipe_title)
                    st.rerun()

                # Recipe details
                recipe_markdown = _recipe_details(recipe)
                st.markdown(recipe_markdown, unsafe_allow_html=True)


def render_liked_recipes_footer(liked_list: List[str], recipe_db: RecipeDB):
    """
    Render a floating footer showing liked recipes with clickable buttons.

    Args:
        liked_list: List of recipe titles that are liked
        recipe_db: RecipeDB instance for getting recipe information
    """
    if not liked_list:
        return

    # Inject CSS for floating footer using the container key
    st.markdown(
        """
    <style>
    /* Style the footer container using its key */
    .st-key-liked-recipes-footer {
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
    """,
        unsafe_allow_html=True,
    )

    # Create footer container with a key for CSS styling
    with st.container(key="liked-recipes-footer", border=False):
        st.markdown(f":material/favorite: Liked Recipes ({len(liked_list)})")

        # Use horizontal flex container for buttons
        with st.container(horizontal=True, horizontal_alignment="left", gap="small"):
            for recipe_title in liked_list[
                :20
            ]:  # Limit to first 20 to prevent wrapping
                try:
                    recipe = recipe_db.get_recipe_by_name(recipe_title)
                    if recipe is None:
                        st.button(
                            f"{recipe_title} (not found)",
                            key=f"footer_like_{recipe_title}",
                            disabled=True,
                            width=200,
                        )
                        continue

                    # Truncate long recipe names
                    display_name = recipe_title
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."

                    # Create dialog function for this recipe
                    def make_dialog(recipe: Recipe):
                        @st.dialog(recipe.title, width="large")
                        def show_recipe_dialog():
                            st.markdown(_recipe_details(recipe), unsafe_allow_html=True)

                            # Remove from liked list button
                            if st.button(
                                "Remove from Liked",
                                type="primary",
                                key=f"remove_{recipe.title}",
                            ):
                                st.session_state.liked_recipes.remove(recipe.title)
                                st.rerun()

                        return show_recipe_dialog

                    dialog_fn = make_dialog(recipe)

                    st.button(
                        display_name,
                        key=f"footer_like_{recipe_title}",
                        on_click=dialog_fn,
                        width=200,
                        type="primary",
                    )
                except Exception:
                    # Recipe not found - show disabled button
                    st.button(
                        f"{recipe_title} (error)",
                        key=f"footer_like_{recipe_title}",
                        disabled=True,
                        width=200,
                    )

        if len(liked_list) > 20:
            st.caption(f"... and {len(liked_list) - 20} more recipes")


def render_search_interface(
    recipe_db: Optional[RecipeDB] = None, num_results: int = 30
):
    """
    Main function to render the complete recipe search interface as a reusable component.

    Args:
        recipe_db: Optional RecipeDB instance. If None, creates a new one.
        num_results: Fixed number of results to show (default: 21)
    """
    if recipe_db is None:
        recipe_db = RecipeDB()

    # Initialize searcher
    if "recipe_searcher" not in st.session_state:
        with st.spinner("Initializing search index..."):
            st.session_state.recipe_searcher = RecipeSearcher(recipe_db)

    searcher = st.session_state.recipe_searcher

    # Initialize liked recipes list
    if "liked_recipes" not in st.session_state:
        st.session_state.liked_recipes = set()

    # Main search interface
    st.markdown(
        "Search through the AllRecipes database. Find recipes you like and add them to your liked list."
    )

    # Render filters in expandable at the top
    filters = render_recipe_filters(searcher)

    # Search query input and button
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            ":material/search: Search Query",
            placeholder="Search query e.g., 'chicken pasta', 'vegetarian dessert', 'quick breakfast'",
            key="recipe_search_query",
            label_visibility="collapsed",
        )
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Store search state in session state
    if "last_recipe_search_query" not in st.session_state:
        st.session_state.last_recipe_search_query = None
    if "last_recipe_search_results" not in st.session_state:
        st.session_state.last_recipe_search_results = None
    if "last_recipe_search_filters" not in st.session_state:
        st.session_state.last_recipe_search_filters = None

    # Perform search if query changed or search button clicked
    should_search = search_button or (
        query and query != st.session_state.last_recipe_search_query
    )

    if should_search:
        with st.spinner("Searching recipes..."):
            results_df = searcher.search(query, filters=filters, top_k=100)
            st.session_state.last_recipe_search_query = query
            st.session_state.last_recipe_search_results = results_df
            st.session_state.last_recipe_search_filters = filters
    elif st.session_state.last_recipe_search_results is not None:
        results_df = st.session_state.last_recipe_search_results
        if filters != st.session_state.last_recipe_search_filters:
            results_df = searcher.search(
                st.session_state.last_recipe_search_query or "",
                filters=filters,
                top_k=100,
            )
            st.session_state.last_recipe_search_results = results_df
            st.session_state.last_recipe_search_filters = filters
    else:
        results_df = None

    # Display results if available
    if results_df is not None:
        render_recipe_results(results_df, recipe_db, max_results=num_results)

    # Display liked recipes in a floating footer at the bottom
    if st.session_state.liked_recipes:
        liked_list = sorted(list(st.session_state.liked_recipes))
        render_liked_recipes_footer(liked_list, recipe_db)


def render_liked_items(liked_items: set, recipe_db: Optional[RecipeDB] = None):
    """
    Render liked items in recipe cards for the final comparison.

    Args:
        liked_items: Set of recipe titles that are liked
        recipe_db: Optional RecipeDB instance. If None, creates a new one.
    """
    if recipe_db is None:
        recipe_db = RecipeDB()

    if not liked_items or len(liked_items) == 0:
        st.markdown("*No items were liked during exploration.*")
        return

    st.markdown(f"You liked {len(liked_items)} item(s) during exploration.")
    liked_list = list(liked_items)

    # Render liked recipes in cards
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for recipe_title in liked_list:
            try:
                recipe = recipe_db.get_recipe_by_name(recipe_title)
                if recipe:
                    with st.container(border=True, width=500, height=700):
                        recipe_markdown = _recipe_details(recipe)
                        st.markdown(recipe_markdown, unsafe_allow_html=True)
                else:
                    st.write(f"Recipe: {recipe_title} (not found)")
            except (ValueError, AttributeError, TypeError):
                st.write(f"Recipe: {recipe_title} (error loading)")
