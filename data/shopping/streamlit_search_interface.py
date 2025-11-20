"""
Streamlit search interface components for the shopping catalog.

Provides:
- BM25-based natural language search
- Filter components (price, department, color, product type, etc.)
- Search results display

Dependencies:
- rank_bm25: Install with `pip install rank-bm25`
"""

from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "liked_products",
    "searcher",
    "last_search_query",
    "last_search_results",
    "last_search_filters",
]

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required. Install it with: pip install rank-bm25"
    )
import re
from data.shopping.db import Catalog
from data.shopping.streamlit_render import (
    _get_product_image_html,
    _product_details_to_markdown,
)


class CatalogSearcher:
    """Handles BM25 search and filtering for the catalog."""
    
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.df = catalog.df.copy()
        
        # Create search_area with all text columns for indexing (overwrite existing if present)
        # Exclude numeric columns (article_id, price) and search_area itself
        exclude_cols = {"article_id", "price", "search_area"}
        searchable_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols and self.df[col].dtype == "object"
        ]
        self.df["search_area"] = (
            self.df[searchable_cols]
            .apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
            .str.lower()
        )
        
        # Tokenize documents for BM25 using the search_area column
        self.tokenized_docs = [
            self._tokenize(text) for text in self.df["search_area"]
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Cache for unique values
        self._unique_values_cache = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        # Simple tokenization: split on whitespace and remove punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Search the catalog using BM25 and apply filters.
        
        Args:
            query: Natural language search query
            filters: Dictionary of filter criteria
            top_k: Number of top results to return
            
        Returns:
            DataFrame with search results sorted by relevance
        """
        if not query or not query.strip():
            # If no query, return all products (filtered if filters are provided)
            results_df = self.df.copy()
        else:
            # Tokenize query
            tokenized_query = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Create results dataframe with scores
            results_df = self.df.copy()
            results_df["bm25_score"] = scores
            
            # Sort by score and take top_k
            results_df = results_df.sort_values("bm25_score", ascending=False)
            results_df = results_df.head(top_k)
        
        # Apply filters
        if filters:
            results_df = self._apply_filters(results_df, filters)
        
        return results_df.reset_index(drop=True)
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe."""
        filtered_df = df.copy()
        
        # Price range filter
        if "price_min" in filters and filters["price_min"] is not None:
            filtered_df = filtered_df[filtered_df["price"] >= filters["price_min"]]
        if "price_max" in filters and filters["price_max"] is not None:
            filtered_df = filtered_df[filtered_df["price"] <= filters["price_max"]]
        
        # Department filter
        if "department" in filters and filters["department"]:
            filtered_df = filtered_df[filtered_df["department_name"] == filters["department"]]
        
        # Section filter
        if "section" in filters and filters["section"]:
            filtered_df = filtered_df[filtered_df["section_name"] == filters["section"]]
        
        # Product type filter
        if "product_type" in filters and filters["product_type"]:
            filtered_df = filtered_df[filtered_df["product_type_name"] == filters["product_type"]]
        
        # Product group filter
        if "product_group" in filters and filters["product_group"]:
            filtered_df = filtered_df[filtered_df["product_group_name"] == filters["product_group"]]
        
        # Color filter
        if "color" in filters and filters["color"]:
            filtered_df = filtered_df[filtered_df["colour_group_name"] == filters["color"]]
        
        # Color master filter
        if "color_master" in filters and filters["color_master"]:
            filtered_df = filtered_df[filtered_df["perceived_colour_master_name"] == filters["color_master"]]
        
        # Index group filter (brand)
        if "index_group" in filters and filters["index_group"]:
            filtered_df = filtered_df[filtered_df["index_group_name"] == filters["index_group"]]
        
        return filtered_df
    
    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column (with caching)."""
        if column not in self._unique_values_cache:
            unique_vals = sorted(self.df[column].dropna().unique().tolist())
            self._unique_values_cache[column] = unique_vals
        return self._unique_values_cache[column]
    
    def get_price_range(self) -> Tuple[float, float]:
        """Get min and max price from catalog."""
        return float(self.df["price"].min()), float(self.df["price"].max())


def render_search_filters(searcher: CatalogSearcher) -> Dict[str, Any]:
    """
    Render filter UI components in an expandable and return filter values.
    
    Returns:
        Dictionary of filter values
    """
    filters = {}
    
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Price range filter
            price_min, price_max = searcher.get_price_range()
            price_range = st.slider(
                "Price ($)",
                min_value=float(price_min),
                max_value=float(price_max),
                value=(float(price_min), float(price_max)),
                step=10.0,
                key="price_filter"
            )
            filters["price_min"] = price_range[0]
            filters["price_max"] = price_range[1]
                        
            # Index group (brand) filter
            index_groups = searcher.get_unique_values("index_group_name")
            if index_groups:
                selected_index_group = st.selectbox(
                    "Select Demographic/Collection",
                    options=["All"] + index_groups,
                    key="index_group_filter"
                )
                if selected_index_group != "All":
                    filters["index_group"] = selected_index_group
        
        with col2:
            # Product type filter
            product_types = searcher.get_unique_values("product_type_name")
            if product_types:
                selected_type = st.selectbox(
                    "Select Product Type",
                    options=["All"] + product_types[:50],  # Limit to first 50 for performance
                    key="product_type_filter"
                )
                if selected_type != "All":
                    filters["product_type"] = selected_type
            
            # Color filter
            colors = searcher.get_unique_values("colour_group_name")
            if colors:
                selected_color = st.selectbox(
                    "Select Color",
                    options=["All"] + colors[:50],
                    key="color_filter"
                )
                if selected_color != "All":
                    filters["color"] = selected_color
        
        with col3:
            # Pattern filter
            patterns = searcher.get_unique_values("graphical_appearance_name")
            if patterns:
                selected_pattern = st.selectbox(
                    "Select Pattern",
                    options=["All"] + patterns,
                    key="pattern_filter"
                )
                if selected_pattern != "All":
                    filters["pattern"] = selected_pattern
            
    
    return filters


def render_liked_products_footer(liked_list: List[int], catalog: Catalog):
    """
    Render a floating footer showing liked products with clickable buttons.
    
    Args:
        liked_list: List of article IDs that are liked
        catalog: Catalog instance for getting product information
    """
    if not liked_list:
        return
    
    # Inject CSS for floating footer using the container key
    st.markdown("""
    <style>
    /* Style the footer container using its key */
    .st-key-liked-products-footer {
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
    with st.container(key="liked-products-footer", border=False):
        st.markdown(f":material/favorite: Liked Products ({len(liked_list)})")
        
        # Use horizontal flex container for buttons
        with st.container(horizontal=True, horizontal_alignment="left", gap="small"):
            for liked_id in liked_list[:20]:  # Limit to first 20 to prevent wrapping
                try:
                    product = catalog.get_row_by_article_id(liked_id)
                    product_series = product
                    product_name = product_series.get("prod_name", f"Product {liked_id}")
                    
                    # Truncate long product names
                    if len(product_name) > 20:
                        product_name = product_name[:17] + "..."
                    
                    # Create dialog function for this product (using closure to capture values)
                    def make_dialog(article_id: int, prod: pd.Series):
                        @st.dialog(product_name, width="large")
                        def show_product_dialog():
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Product image
                                try:
                                    img_html = _get_product_image_html(prod, catalog, max_width=300)
                                    st.markdown(img_html, unsafe_allow_html=True)
                                except Exception:
                                    st.write("No image available")
                            
                            with col2:
                                # Product details
                                product_dict = prod.to_dict()
                                details_markdown = _product_details_to_markdown(0, product_dict)
                                st.markdown(details_markdown, unsafe_allow_html=True)
                                
                                # Price
                                st.markdown(f"**Price:** ${prod['price']:.2f}")
                                
                                # Remove from liked list button
                                if st.button("Remove from Liked", type="primary", key=f"remove_{article_id}"):
                                    st.session_state.liked_products.remove(article_id)
                                    st.rerun()
                        
                        return show_product_dialog
                    
                    dialog_fn = make_dialog(liked_id, product_series)
                    
                    # Button with fixed width
                    st.button(
                        product_name,
                        key=f"footer_like_{liked_id}",
                        on_click=dialog_fn,
                        width=200,
                        type="primary"
                    )
                except ValueError:
                    # Product not found - show disabled button with ID
                    st.button(
                        f"Product {liked_id} (not found)",
                        key=f"footer_like_{liked_id}",
                        disabled=True,
                        width=200
                    )
        
        if len(liked_list) > 30:
            st.caption(f"... and {len(liked_list) - 30} more products")


def render_search_results(results_df: pd.DataFrame, catalog: Catalog, max_results: int = 50):
    """
    Render search results in a nice format, reusing components from streamlit_render.py.
    Includes a "plus" button to add products to liked list.
    
    Args:
        results_df: DataFrame with search results
        catalog: Catalog instance for getting images
        max_results: Maximum number of results to display
    """
    if results_df is None or results_df.empty:
        st.subheader("Search Results (0 products)")
        st.write("No products found matching your search criteria.")
        return
    
    # Initialize liked list in session state if not present
    if "liked_products" not in st.session_state:
        st.session_state.liked_products = set()
    
    num_results = min(len(results_df), max_results)
    st.subheader(f"Search Results ({num_results} of {len(results_df)} products)")
    
    # Use a horizontal flex container that adapts to screen width
    # Each product card has a fixed width and height for consistency
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for product_idx in range(num_results):
            row = results_df.iloc[product_idx]
            product_series = row  # row is already a Series
            article_id = int(product_series["article_id"])
            
            # Each product card container with fixed width and height (600px height for consistency)
            # Content will scroll if it overflows
            with st.container(border=True, width=500, height=700):
                # Plus button to add to liked list (at the top)
                is_liked = article_id in st.session_state.liked_products
                button_label = ":material/heart_check:" if is_liked else ":material/heart_plus:"
                button_type = "primary" if is_liked else "secondary"
                
                if st.button(
                    button_label,
                    key=f"like_button_{article_id}",
                    type=button_type,
                    use_container_width=True,
                    help="Add to liked list"
                ):
                    if is_liked:
                        st.session_state.liked_products.remove(article_id)
                    else:
                        st.session_state.liked_products.add(article_id)
                    st.rerun()
            
                # Image
                try:
                    img_html = _get_product_image_html(product_series, catalog, max_width=200)
                    st.markdown(f"<center>{img_html}</center>", unsafe_allow_html=True)
                except Exception:
                    st.write("No image available")
                
                # Product details
                product_dict = product_series.to_dict()
                details_markdown = _product_details_to_markdown(product_idx + 1, product_dict)
                st.markdown(details_markdown, unsafe_allow_html=True)
                
                # Price
                st.markdown(f"**Price:** ${product_series['price']:.2f}")
                
                # BM25 score if available
                if "bm25_score" in product_series:
                    st.caption(f"Relevance Score: {product_series['bm25_score']:.4f}")
                


def render_search_interface(catalog: Optional[Catalog] = None, num_results: int = 30):
    """
    Main function to render the complete search interface as a reusable component.
    
    Args:
        catalog: Optional Catalog instance. If None, creates a new one.
        num_results: Fixed number of results to show (default: 20)
    """
    if catalog is None:
        catalog = Catalog()
    
    # Initialize searcher
    if "searcher" not in st.session_state:
        with st.spinner("Initializing search index..."):
            st.session_state.searcher = CatalogSearcher(catalog)
    
    searcher = st.session_state.searcher
    
    # Initialize liked products list
    if "liked_products" not in st.session_state:
        st.session_state.liked_products = set()
    
    # Main search interface
    st.markdown("Search through the H&M product catalog. Find items you like for the task and add them to your liked list.")
    
    # Render filters in expandable at the top
    filters = render_search_filters(searcher)
    
    # Search query input and button
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            ":material/search: Search Query",
            placeholder="Search query e.g., 'blue hoodie', 'summer dress', 'casual jeans'",
            key="search_query",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    
    # Store search state in session state
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = None
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None
    if "last_search_filters" not in st.session_state:
        st.session_state.last_search_filters = None
    
    # Perform search if query changed or search button clicked
    should_search = search_button or (query and query != st.session_state.last_search_query)
    if should_search:
        with st.spinner("Searching catalog..."):
            results_df = searcher.search(query, filters=filters, top_k=100)
            # Store results in session state
            st.session_state.last_search_query = query
            st.session_state.last_search_results = results_df
            st.session_state.last_search_filters = filters
    elif st.session_state.last_search_results is not None:
        # Restore previous results
        results_df = st.session_state.last_search_results
        # Reapply filters if they changed
        if filters != st.session_state.last_search_filters:
            results_df = searcher.search(
                st.session_state.last_search_query or "",
                filters=filters,
                top_k=100
            )
            st.session_state.last_search_results = results_df
            st.session_state.last_search_filters = filters
    else:
        results_df = None
    
    # Display results if available
    render_search_results(results_df, catalog, max_results=num_results)
    
    # Display liked products in a floating footer at the bottom
    if st.session_state.liked_products:
        liked_list = sorted(list(st.session_state.liked_products))
        render_liked_products_footer(liked_list, catalog)


def render_liked_items(liked_items: set, catalog: Optional[Catalog] = None):
    """
    Render liked items in product cards for the final comparison.
    
    Args:
        liked_items: Set of article IDs that are liked
        catalog: Optional Catalog instance. If None, creates a new one.
    """
    if catalog is None:
        catalog = Catalog()
    
    if not liked_items or len(liked_items) == 0:
        st.markdown("*No items were liked during exploration.*")
        return
    
    st.markdown(f"You liked {len(liked_items)} item(s) during exploration.")
    liked_list = list(liked_items)
    
    # Render liked products in cards
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        for idx, article_id in enumerate(liked_list):
            try:
                product_row = catalog.get_row_by_article_id(int(article_id))
                product_series = pd.Series(product_row)
                
                with st.container(border=True, width=500, height=700):
                    # Image
                    try:
                        img_html = _get_product_image_html(product_series, catalog, max_width=200)
                        st.markdown(f"<center>{img_html}</center>", unsafe_allow_html=True)
                    except Exception:
                        st.write("No image available")
                    
                    # Product details
                    product_dict = product_series.to_dict()
                    details_markdown = _product_details_to_markdown(idx + 1, product_dict)
                    st.markdown(details_markdown, unsafe_allow_html=True)
                    
                    # Price
                    st.markdown(f"**Price:** ${product_series['price']:.2f}")
            except (ValueError, KeyError, AttributeError, TypeError):
                st.write(f"Product {article_id} (error loading)")

