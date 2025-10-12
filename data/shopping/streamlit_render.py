from typing import Any, List, Tuple, Dict, Literal
import streamlit as st
import pandas as pd
from PIL import Image
from utils.streamlit_types import FormElement, form_element_to_streamlit
import random
from evaluation.qualitative_eval import COMPARISON_LIKERT, INSTRUMENT_LIKERT
import inflect


def comparison_to_md(
    y1: str, y2: str, db: Any, validity_fn=None, validity_kwargs=None
) -> str:
    y1_products, y1_total_cost, y1_num_valid, y1_has_invalid = (
        _process_cart_with_invalid_tracking(y1, db)
    )
    y2_products, y2_total_cost, y2_num_valid, y2_has_invalid = (
        _process_cart_with_invalid_tracking(y2, db)
    )

    if len(y1_products) == 0 and len(y2_products) == 0:
        return "No items in either cart"

    y1_summary, y1_error = _get_validation_strings(
        y1, y1_has_invalid, validity_fn, validity_kwargs
    )
    y2_summary, y2_error = _get_validation_strings(
        y2, y2_has_invalid, validity_fn, validity_kwargs
    )

    markdown_lines: List[str] = []
    markdown_lines.append("### Shopping Cart Comparison")
    markdown_lines.append("")
    markdown_lines.append("| | | | | | |")
    markdown_lines.append("|---|---|---|---|---|---|")
    markdown_lines.append(
        "| **🛍️ Shopping Cart A** | | Price | **🛍️ Shopping Cart B** | | Price |"
    )
    markdown_lines.append(f"| |{y1_summary} | |  | {y2_summary} | |")

    max_rows = max(len(y1_products), len(y2_products))
    if max_rows == 0:
        markdown_lines.append(
            "| *No items in either cart* | | | *No items in either cart* | | |"
        )
        return "\n".join(markdown_lines)

    for i in range(max_rows):
        if i < len(y1_products):
            y1_item = y1_products[i]
            if y1_item["is_valid"]:
                y1_image_html = _get_product_image_html(y1_item["product"], db)
                y1_details = _product_details_to_markdown(
                    i + 1, y1_item["product"].to_dict()
                )
                y1_price = f"\${y1_item['product']['price']:.2f}"
            else:
                y1_image_html = ":material/error:"
                y1_details = (
                    f":red-background[*Invalid product (ID: {y1_item['article_id']})*]"
                )
                y1_price = "*N/A*"
        else:
            y1_image_html = ""
            y1_details = ""
            y1_price = ""

        if i < len(y2_products):
            y2_item = y2_products[i]
            if y2_item["is_valid"]:
                y2_image_html = _get_product_image_html(y2_item["product"], db)
                y2_details = _product_details_to_markdown(
                    i + 1, y2_item["product"].to_dict()
                )
                y2_price = f"\${y2_item['product']['price']:.2f}"
            else:
                y2_image_html = ":material/error:"
                y2_details = (
                    f":red-background[*Invalid product (ID: {y2_item['article_id']})*]"
                )
                y2_price = "*N/A*"
        else:
            y2_image_html = ""
            y2_details = ""
            y2_price = ""

        markdown_lines.append(
            f"| {y1_image_html} | {y1_details} | {y1_price} | {y2_image_html} | {y2_details} | {y2_price} |"
        )

    y1_total_str = f"**Total: \${y1_total_cost:.2f}**"
    y2_total_str = f"**Total: \${y2_total_cost:.2f}**"
    markdown_lines.append(f"| | | {y1_total_str} | | | {y2_total_str} |")
    markdown_lines.append("")
    return "\n".join(markdown_lines)


def _shopping_recommendations_to_markdown(
    cart: str,
    catalog: Any,
    header_type: Literal["cart", "mention", None] = "cart",
) -> str:
    products, total_cost, num_valid, has_invalid = _process_cart_with_invalid_tracking(
        cart, catalog
    )
    if len(products) == 0:
        return ""

    markdown_lines: List[str] = []
    if header_type == "cart":
        markdown_lines.append("### 🛍️ Items in Shopping Cart")
        markdown_lines.append(
            f"Total items: {len(products)}"
            + (
                f" ({len(products) - num_valid} invalid)"
                if len(products) != num_valid
                else ""
            )
            + f" | **Total cost: \${total_cost:.2f}**"
        )
    elif header_type == "mention":
        markdown_lines.append("")

    markdown_lines.append("| |  | Price |")
    markdown_lines.append("|---|---|---|")
    for i, product in enumerate(products, 1):
        if product["is_valid"]:
            img_html = _get_product_image_html(
                product["product"], catalog, max_width=200
            )
            markdown_lines.append(
                f"| {img_html} | {_product_details_to_markdown(i, product['product'].to_dict())} | \${product['product']['price']:.2f} |"
            )
        else:
            markdown_lines.append(
                f"| :material/error: | :red-background[*Invalid product (ID: {product['article_id']})*] | *N/A* |"
            )

    markdown_lines.append(f"|||Total ({len(products)} items): **\${total_cost:.2f}**|")
    return "\n".join(markdown_lines)


def _get_validation_strings(y, y_has_invalid, validity_fn, validity_kwargs):
    if validity_fn is not None and validity_kwargs is not None:
        y_valid, y_metadata = validity_fn(
            y, **(validity_kwargs or {}), raise_errors=False
        )
        y_error = y_metadata.get("violated_constraints", [])
    else:
        y_valid = not y_has_invalid
        y_error = ["Invalid products in the shopping cart"] if y_has_invalid else []

    if y_valid:
        y_summary = ":small[:green[:material/check: valid cart, passes basic checks]]"
    else:
        y_summary = f" :small[:red[:material/close: invalid cart]]"
        y_summary += ": " + ", ".join(
            [
                f":small[:red[{constraint}]]"
                for constraint in y_error
                if constraint is not None
            ]
        )
    return y_summary, y_error


def _process_cart_with_invalid_tracking(
    cart: str, catalog: Any
) -> Tuple[List[Dict], float, int, bool]:
    from utils.misc import parse_for_answer_tags  # local import to avoid cycles

    article_ids = parse_for_answer_tags(
        cart, keyword="cart", return_none_if_not_found=True
    )
    if article_ids is None:
        return [], 0.0, 0, True

    article_ids = [
        int(id.strip()) if id.strip().isdigit() else id.strip()
        for id in article_ids.split(",")
    ]

    products_with_validity: List[Dict] = []
    total_cost = 0.0
    num_valid = 0
    has_invalid = False

    for article_id in article_ids:
        try:
            product = catalog.get_row_by_article_id(article_id)
            products_with_validity.append(
                {"product": product, "is_valid": True, "article_id": article_id}
            )
            total_cost += float(product["price"])
            num_valid += 1
        except ValueError:
            products_with_validity.append(
                {"product": None, "is_valid": False, "article_id": article_id}
            )
            has_invalid = True

    return products_with_validity, total_cost, num_valid, has_invalid


def _get_product_image_html(
    product: pd.Series, catalog: Any, max_width: int = 200
) -> str:
    try:
        image = catalog.get_image_by_article_id(product["article_id"])
        if image:
            import base64
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f'<img src="data:image/jpeg;base64,{img_str}" alt="Product {product["article_id"]}" style="max-width: {max_width}px; height: auto;" />'
        else:
            return "*Image not available*"
    except Exception:
        return "*Image not available*"


def _product_details_to_markdown(i: int, product: Dict[str, Any]) -> str:
    lines: List[str] = []
    basic_info: List[str] = []
    if product.get("product_group_name"):
        basic_info.append(product["product_group_name"])
    if product.get("product_type_name"):
        basic_info.append(product["product_type_name"])

    brand_info: List[str] = []
    if product.get("department_name"):
        brand_info.append(product["department_name"])
    if product.get("index_group_name"):
        brand_info.append(product["index_group_name"])
    if product.get("section_name"):
        brand_info.append(product["section_name"])

    style_info: List[str] = []
    if product.get("graphical_appearance_name"):
        style_info.append(product["graphical_appearance_name"])
    if product.get("perceived_colour_value_name") and product.get("colour_group_name"):
        style_info.append(
            f"{product['colour_group_name']} ({product['perceived_colour_value_name']})"
        )

    lines.append(f"<h4>{product['prod_name']}</h4>")
    lines.append("**Catalog ID:** " + str(product["article_id"]))
    if style_info:
        lines.append("**Style:** " + " > ".join(style_info))
    if product.get("detail_desc"):
        lines.append("")
        lines.append(product["detail_desc"])
        lines.append("")
    if basic_info:
        lines.append("**Category:** " + " > ".join(basic_info))
    if brand_info:
        lines.append("**Section:** " + " > ".join(brand_info))
    return "<br>".join(lines)


def render_eval(
    *, final_prediction: str, y0: List[str] | None, db
) -> tuple[bool, dict | None]:
    """
    Render evaluation UI for shopping custom specs and return (completed, feedback).
    """
    if "ranking" not in st.session_state.form_results["final_evaluation"]:
        render_eval_first_page(final_prediction=final_prediction, y0=y0, db=db)
        return False, None

    return render_eval_second_page(final_prediction=final_prediction, y0=y0, db=db)


def render_eval_first_page(*, final_prediction: str, y0: List[str], db):
    st.markdown("### Review the assistant's recommendations")
    st.markdown(
        "The assistant has generated several recommendations for you. Select your favorite below. **Note that you must buy all the items in the cart together.**"
    )

    carousel_completed = (
        st.session_state.form_results["final_evaluation"].get(
            "carousel_selection_index", None
        )
        is not None
    )

    # Carousel
    from evaluation.app.components import carousel

    if not carousel_completed:
        options = y0 + [final_prediction]
        random.shuffle(options)
        print(options)
    else:
        options = list(
            st.session_state.form_results["final_evaluation"]["index_to_cart"].values()
        )

    def on_completion(index):
        print(f"Selected cart {index}")
        st.session_state.form_results["final_evaluation"].update(
            {
                "carousel_selection": options[index],
                "carousel_selection_index": index,
                "index_to_cart": {i: y for i, y in enumerate(options)},
            }
        )
        st.rerun()

    def display_fn(i):
        st.markdown(
            _shopping_recommendations_to_markdown(options[i], db, header_type="cart"),
            unsafe_allow_html=True,
        )

    carousel(
        [lambda i=i: display_fn(i) for i in range(len(options))],
        include_select_button=not carousel_completed,
        select_on_click=on_completion,
    )
    if carousel_completed:
        st.write(
            f"<center><em>Selected cart {st.session_state.form_results['final_evaluation']['carousel_selection_index'] + 1}</em></center>",
            unsafe_allow_html=True,
        )

        # Ask for ranking
        with st.form(key="shopping_ranking_form"):
            styles_favorites = st.multiselect(
                "Select the cart(s) whose **styles** you like most",
                [i for i in range(len(options))],
                format_func=lambda x: f"Cart {x + 1}",
            )
            prices_favorites = st.multiselect(
                "Select the cart(s) whose **prices** you like most",
                [i for i in range(len(options))],
                format_func=lambda x: f"Cart {x + 1}",
            )
            colors_favorites = st.multiselect(
                "Select the cart(s) whose **colors** you like most",
                [i for i in range(len(options))],
                format_func=lambda x: f"Cart {x + 1}",
            )
            must_haves_nice_to_haves = st.text_area(
                "What are your **must haves** vs. **nice to haves**? How do the carts compare?",
                height=120,
            )
            rank = st.multiselect(
                "Rank the carts above from MOST to LEAST preferred, taking in consideration all of the factors above.",
                [i for i in range(len(options))],
                default=[
                    st.session_state.form_results["final_evaluation"][
                        "carousel_selection_index"
                    ]
                ],
                format_func=lambda x: f"Cart {x + 1}",
            )
            submit = st.form_submit_button("Submit", type="primary")
            if submit:
                if len(rank) != len(options):
                    st.error("Please rank all carts")
                    return
                if (
                    rank[0]
                    != st.session_state.form_results["final_evaluation"][
                        "carousel_selection_index"
                    ]
                ):
                    st.error("Please rank the selected cart first")
                    return
                if (
                    len(styles_favorites) == 0
                    or len(colors_favorites) == 0
                    or len(prices_favorites) == 0
                ):
                    st.error("Please select at least one option for each category")
                    return
                if len(must_haves_nice_to_haves) == 0:
                    st.error("Please fill out all fields")
                    return
                final_prediction_index = next(
                    k
                    for k, v in st.session_state.form_results["final_evaluation"][
                        "index_to_cart"
                    ].items()
                    if v == final_prediction
                )
                final_prediction_rank = rank.index(final_prediction_index)
                st.session_state.form_results["final_evaluation"].update(
                    {
                        "ranking": rank,
                        "score": len(options) - final_prediction_rank,
                        "styles_favorites": styles_favorites,
                        "colors_favorites": colors_favorites,
                        "prices_favorites": prices_favorites,
                        "must_haves_nice_to_haves": must_haves_nice_to_haves,
                    }
                )


def render_eval_second_page(*, final_prediction: str, y0: List[str], db):
    st.write("### Evaluate a specific cart")
    # Display the cart
    p = inflect.engine()
    ranking = (
        len(st.session_state.form_results["final_evaluation"]["index_to_cart"])
        - st.session_state.form_results["final_evaluation"]["score"]
        + 1
    )
    st.write(
        f"<center><em>You ranked this cart {p.ordinal(ranking)} out of {len(st.session_state.form_results['final_evaluation']['index_to_cart'])}</em></center>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, height=800):
        st.markdown(
            _shopping_recommendations_to_markdown(
                final_prediction, db, header_type="cart"
            ),
            unsafe_allow_html=True,
        )

    # Display the form
    _, total_cost, _, _ = _process_cart_with_invalid_tracking(final_prediction, db)
    form_elements = [
        FormElement(
            input_type="stars",
            label="Rate this cart.",
        ),
        FormElement(
            input_type="radio",
            label=f'How much do you agree with this statement: "I would rather purchase this cart as is (at \${total_cost:.2f}) instead of continuing my search with the assistant for 10 more minutes."',
            options=["-"] + INSTRUMENT_LIKERT,
        ),
        FormElement(
            input_type="text_area",
            label="If you were to continue your search with the assistant for 10 more minutes, what would you want it to change?",
            height=120,
        )
    ]

    with st.form(key="shopping_custom_eval_form"):
        feedback: dict = {}
        for element in form_elements:
            st_fn, st_kwargs, required = form_element_to_streamlit(element)
            value = st_fn(**st_kwargs)
            label = element.get("label", "question")
            feedback[label] = value
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            for element in form_elements:
                if element.get("required", False):
                    label = element.get("label")
                    if (
                        not feedback.get(label)
                        or feedback.get(label) == ""
                        or feedback.get(label) == "-"
                    ):
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None
