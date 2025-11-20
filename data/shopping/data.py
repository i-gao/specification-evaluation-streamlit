import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from langchain_core.tools import tool
import pandas as pd
import json
from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
    LinearFixedSpecification,
)
from data.actions import Action, get_jupyter_actions
from collections import Counter
import inflect
from data.reward import linear_reward
import math
from utils.misc import (
    hash,
    add_section,
    parse_for_answer_tags,
    replace_tags_with_link,
)
from utils.streamlit_types import FormElement
from data.reward import Constraint
from data.shopping.reward_utils.helpers import clip_score
import data.shopping.streamlit_render as renderer
import streamlit as st
from typing import Callable  # noqa: F401

# extractors used in local helper below
import data.shopping.extractors as extractors

from data.shopping.db import Catalog

DEV_FRAC = 0.3
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
VISUAL_SCORE_WEIGHT = 0.5

FEATURES_OF_INTEREST = [
    "product_group_name",
    "product_type_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_group_name",
    "index_name",
    "section_name",
    "garment_group_name",
]

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to buy clothes for a client from H&M.** The client has some shopping goal they have delegated to you, and they have specified a budget. Your job is to work with the assistant to find the best products for the client.

When the chat session starts, you will see some information about what items the client is looking for. 

### The tricky part
Some of the client's preferences may be missing. For example, they may not have specified what styles of fashions they like.

You will need to use the tools on the side panel to get more information about the client.

To maximize your score, you may have to recommend different products and ask the client to evaluate them. The client's score will be between 0 and 100.
"""

COMMONSENSE_DESCRIPTION = "Recommend products from the given catalog. All products must come from the catalog. You can assume that all sizes are in stock."

PREDICTION_FMT_INSTRUCTIONS = "Return the article_ids of the products to recommend to the customer, separated by commas and wrapped in <cart></cart>, e.g.: '<cart>123456,123457,123458</cart>'."

MSG_FMT_INSTRUCTIONS = "Communicate with the user in language. Always mention the article_id's of products and wrap these in <item></item>, e.g.: '<item>123456</item>'. This will append a widget describing the product at the end of your message so the user can view the product. You should always do this by default."


def render_fixed_task_explanation():
    """Render the fixed task explanation for shopping."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


class ShoppingDataset(SpecificationCollection):
    @property
    def dataset_name(self) -> str:
        return "shopping"

    @property
    def dataset_pretty_name(self) -> str:
        return "Fashion Shopping"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **shop for clothes from H&M Online.**"

    @property
    def assets_file_id(self) -> str:
        return "1-y6KOwyRSWD_rinyE-MnRGl7ZAYVF2Hs"

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return [
            {
                "image_name": "jupyter_docker_image",
                "dockerfile_path": "utils/jupyter_docker_image/Dockerfile",
                "build_context": "utils/jupyter_docker_image",
                "description": "Docker image for Jupyter notebook",
            },
            {
                "image_name": "h_m_shopping",
                "dockerfile_path": "data/shopping/reward_utils/Dockerfile",
                "build_context": "data/shopping",
                "description": "Docker image for Shopping code evaluation",
            },
        ]

    def _create_user_expertise_form(self, is_custom: bool = False, product_types: List[str] = None) -> List[FormElement]:
        """
        Create user expertise form elements for fashion and shopping knowledge.
        
        Args:
            is_custom: If True, adds questions about the specific product types (for custom specs).
            product_types: Optional list of product types being shopped for (e.g., ["hoodie", "pants"]).
        """
        form_elements = [
            FormElement(
                input_type="radio",
                label="How frequently do you read/watch about fashion OR browse in-person/online for clothes?",
                options=[
                    "I have never shopped for clothes or read about fashion",
                    "A few times a year",
                    "Once or twice a month",
                    "Weekly",
                    "Almost every day",
                ],
                default="Once or twice a month",  # Default to middle option
                required=True,
                help="This helps us understand your experience level with fashion and shopping",
            )
        ]
        
        # Add product-specific questions for custom specs (one question per product type)
        if is_custom and product_types is not None:
            # Get unique product types
            unique_product_types = list(set(product_types))
            p = inflect.engine()
            for product_type in unique_product_types:
                plural_product_type = p.plural(product_type)
                form_elements.append(
                    FormElement(
                        input_type="radio",
                        label=f"How familiar are you with types of {plural_product_type}?",
                        options=[
                            "I have never shopped for this type of product",
                            "I have some experience with this type of product",
                            "I am familiar with this type of product",
                            "I am very familiar with this type of product",
                            "I shop for this type of product regularly",
                        ],
                        required=True,
                        help=f"This helps us understand your familiarity with {plural_product_type}",
                    )
                )
        
        return form_elements

    def _create_user_specification_form_final(
        self, intent_data: Dict = None
    ) -> List[FormElement]:
        """
        Create final form elements for budget and demographics.
        """
        return [
            FormElement(
                input_type="slider",
                label="How much are you willing to spend? (Budget in dollars)",
                required=True,
                help=f"Your maximum budget for this shopping trip (minimum: \${40:.2f})",
                min_value=40,
                value=150,
                step=10,
                max_value=300,
            ),
            FormElement(
                input_type="radio",
                label="What section would you like to shop in?",
                options=["Mens", "Womens", "Boys", "Girls"],
                default="Mens",
                required=True,
                help="Select the H&M department you'd like to browse",
            ),
        ]

    def _create_user_evaluation_form(self) -> List[FormElement]:
        """Create the user evaluation form for shopping."""
        return [
            FormElement(
                input_type="radio",
                label="Compare the **styles** of the individual products in the shopping cart A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **colors** of the products in the shopping cart A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **prices** of the products in the shopping cart A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Which set of products are you personally more likely to purchase in real life?",
                options=["A", "neutral", "B"],
                required=True,
            ),
        ]

    def __init__(
        self,
        dev: bool = False,
        docker_image: str = None,
        fixed_indexes: Optional[List[int]] = None,
        custom_indexes: Optional[List[int]] = None,
        persist_docker_container: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load the profiles
        js: Dict[str, str] = json.load(open(f"{DATASET_ROOT}/assets/intents.json"))
        intents = {int(k): v for k, v in js.items()}
        self._intents = intents
        self.fixed_length = len(intents)

        js: List[Dict] = json.load(open(f"{DATASET_ROOT}/assets/custom_intents.json"))
        self._custom_intents = js
        self.custom_length = len(js)
        self._docker_image = docker_image
        self._persist_docker_container = persist_docker_container
        # Import extractors and build lookup
        import data.shopping.extractors as extractors_mod

        self._extractor_lookup = {
            name: func
            for name, func in extractors_mod.__dict__.items()
            if callable(func)
        }
        # Load the catalog to get column information
        self._catalog = Catalog()
        # Use the Database's description format
        self._desc_json = {
            "filename": "catalog.csv",
            "description": self._catalog.table_descriptions["catalog"],
            "columns": self._catalog._list_columns("catalog"),
        }

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)
        if custom_indexes is not None:
            self.load_custom_specs(indexes=custom_indexes)

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}
        # convert to specs
        specs = {}
        for ix in indexes:
            intent = self._intents[ix]
            customer_info, items_df = load_transaction_data(ix)
            ystar = (
                "<cart>"
                + ",".join(items_df["article_id"].astype(str).tolist())
                + "</cart>"
            )
            budget = math.ceil(items_df["price"].sum() / 10) * 10

            signature = f"You are a customer shopping on H&M's website. You want the assistant to recommend products matching your needs.\n**Your shopping intent**: ``{intent}``."
            product_type_counts = items_df["product_type_name"].value_counts()
            p_eng = inflect.engine()
            phr = ", ".join(
                [
                    f"{int(v)} {p_eng.plural(str(k), int(v))}"
                    for k, v in product_type_counts.items()
                ]
            )
            signature += "\n**Products you want to buy**: " + phr

            # Build LinearFixedSpecification features and weights
            if self._persist_docker_container and self._docker_image is not None:
                from llm_sandbox import SandboxSession

                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=self._desc_json,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Build all constraints (hard + soft) via helper
            features_dicts = _build_shopping_constraints(
                items_df, self._catalog, FEATURES_OF_INTEREST, budget=budget
            )

            features: List[Constraint] = [
                Constraint.from_dict(fd, extractor_lookup=self._extractor_lookup)
                for fd in features_dicts
            ]

            # Weights: hard constraints get strong penalty; soft weights via helper based on consistency in true cart
            # All weights should be positive; linear_reward will handle sign correction for penalty constraints
            HARD_PENALTY = 999999.0
            importance = _compute_shopping_feature_importances(
                items_df, self._catalog, features
            )
            # All weights should be positive (absolute value)
            weights: List[float] = []
            for c, imp in zip(features, importance):
                if c.is_hard:
                    weights.append(HARD_PENALTY)
                else:
                    weights.append(
                        abs(float(imp))
                    )  # Use absolute value, sign handled by linear_reward

            spec = LinearFixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                initial_specification=signature,
                commonsense_description=COMMONSENSE_DESCRIPTION,
                features=features,
                weights=weights,
                parse_solutions_fn=_parse_shopping_solutions,
                parse_solutions_and_options_fn=_parse_shopping_solutions_and_options,
                parse_y_fn=lambda yhat, raise_errors: _parse_y_fn(yhat, self._catalog, raise_errors),
                validity_fn_tool_name="check_shopping_cart_validity",
                validity_fn_tool_description="Check if the shopping cart is valid and within budget",
                reward_fn_tool_name="score_shopping_cart",
                reward_fn_tool_description="Score the shopping cart based on feature matches",
                ystar=ystar,
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=actions + get_actions(self._catalog, items_df),
                msg_fmt_instructions=(
                    PREDICTION_FMT_INSTRUCTIONS + " " + MSG_FMT_INSTRUCTIONS
                ),
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_fn_txt=output_to_txt,
                render_msg_kwargs=["db"],
                db=self._catalog,
                name=f"shopping_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(is_custom=False),
            )
            specs[ix] = spec
        return specs

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        """
        Create custom shopping specifications with different shopping prompts.
        """
        if indexes is None:
            return {}

        p = inflect.engine()

        specs = {}
        for ix in indexes:
            if self._persist_docker_container and self._docker_image is not None:
                from llm_sandbox import SandboxSession

                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            # Get Jupyter actions for the environment
            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=self._desc_json,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            custom_intent = self._custom_intents[ix]
            prompt: List[str] = custom_intent["intent"]

            # Start with basic constraints that will be updated by the callback
            initial_constraints = [
                Constraint.create_boolean_penalize_false_constraint(
                    description="Shopping cart must match the specified prompt requirements",
                    extractor="matches_prompt",
                    extractor_kwargs={"catalog": self._catalog, "prompt": prompt},
                    is_hard=True,
                )
            ]
            initial_constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in initial_constraints
            ]

            prompt_as_str = Counter(prompt)
            prompt_as_str = " ".join(
                [f"{v} {p.plural(k)}" for k, v in prompt_as_str.items()]
            )

            # Import search interface before creating spec
            from data.shopping.streamlit_search_interface import (
                render_search_interface,
                render_liked_items,
            )
            
            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification=f"Buy {prompt_as_str} from H&M for the person you have in mind. You can assume that all products are available in all sizes.",
                current_specification=f"Buy {prompt_as_str} from H&M for the person you have in mind. You can assume that all products are available in all sizes.",
                commonsense_description=COMMONSENSE_DESCRIPTION,
                user_specification_form_final=self._create_user_specification_form_final(
                    custom_intent
                ),
                user_specification_callback=user_specification_callback,
                user_specification_callback_kwargs=[
                    "_validity_kwargs",
                    "_y0_mapping",
                    "_extractor_lookup",
                    "_prompt",
                    "initial_specification",
                    "_render_evaluation_kwargs",
                ],
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": initial_constraints,
                    "catalog": self._catalog,
                },
                validity_fn_tool_name="check_shopping_cart_validity",
                validity_fn_tool_description="Check if the shopping cart is valid and within budget",
                y0=None,  # Not provided
                render_task_explanation=self._render_custom_task_explanation,
                actions=actions,
                msg_fmt_instructions=MSG_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=lambda msg, db: output_to_streamlit(msg, db),
                render_msg_kwargs=["db"],
                db=self._catalog,
                render_comparison_fn=output_to_streamlit_comparison,
                name=f"custom_shopping_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(is_custom=True, product_types=prompt),
                _y0_mapping=custom_intent["y0"],
                _prompt=prompt,
                _extractor_lookup=self._extractor_lookup,
                render_evaluation_fn=lambda **kwargs: renderer.render_eval(
                    **kwargs,
                    db=self._catalog,
                ),
                render_search_interface_fn=render_search_interface,
                render_search_interface_kwargs={"catalog": self._catalog},
                render_liked_items_fn=render_liked_items,
                render_liked_items_kwargs={"catalog": self._catalog},
            )
            specs[ix] = spec
        return specs

    def _render_custom_task_explanation(self):
        """Render the custom task explanation for shopping."""

        st.markdown("### What you need to prompt the assistant to do")
        st.markdown(
            "In this task, **your goal is to get the assistant to shop for clothes for you from H&M.** Imagine that you will purchase everything the assistant recommends. Therefore, you should make sure the purchase is within your budget and matches your needs / style."
        )

        with st.container(border=True):
            # Example shopping cart with valid products
            example_cart = "<cart>422106014,569974017</cart>"
            st.info("*Example:* Shopping for 2 hoodies with a budget of \$150")
            try:
                st.markdown(
                    renderer._shopping_recommendations_to_markdown(
                        example_cart, self._catalog, header_type="cart"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    "Example shopping cart with 3 items (2 hoodies, 1 women's top)"
                )

        st.markdown(
            "Think about who you're shopping for and what your budget and style preferences are. The assistant should personalize the shopping recommendations to your needs, picking products that match your taste and budget."
        )
        st.markdown("### Making sure your shopping cart is valid")
        st.markdown(
            "To successfully complete this task, your shopping cart must *be valid.* A valid cart must:"
        )
        st.markdown(
            "* ONLY include real products from the H&M catalog. Using made-up products is not allowed."
        )

        with st.container(border=True):
            # Example with invalid product
            invalid_cart = "<cart>422106014,999999999</cart>"
            st.info(
                ":red[:material/close: *Example:* This is an invalid cart because it includes a made-up product (ID: 999999999), designated by the :material/error: icon]"
            )
            try:
                st.markdown(
                    renderer._shopping_recommendations_to_markdown(
                        invalid_cart, self._catalog, header_type="cart"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    "Example shopping cart with 1 valid item and 1 invalid item (marked with error icon)"
                )

        st.markdown("* Stay within your specified budget.")

        with st.container(border=True):
            # Example that exceeds budget
            over_budget_cart = "<cart>802444001,802553001</cart>"
            st.info(
                ":red[:material/close: *Example:* This cart exceeds the budget of \$150]"
            )
            try:
                st.markdown(
                    renderer._shopping_recommendations_to_markdown(
                        over_budget_cart, self._catalog, header_type="cart"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    "Example shopping cart with 5 items that would exceed a $150 budget"
                )

        st.markdown(
            "* Match the shopping requirements you specified (e.g., if you asked for 2 hoodies, include 2 hoodies)."
        )

        with st.container(border=True):
            # Example that doesn't match requirements
            wrong_items_cart = "<cart>112679048,118458003</cart>"
            st.info(
                ":red[:material/close: *Example:* This cart doesn't match the request for 2 hoodies - it contains a baby sweatshirt and men's joggers instead]"
            )
            try:
                st.markdown(
                    renderer._shopping_recommendations_to_markdown(
                        wrong_items_cart, self._catalog, header_type="cart"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    "Example shopping cart with items that don't match the hoodie request"
                )

        st.markdown(
            "You can assume that all products are available in all sizes, so there is no need to worry about size."
        )
        st.markdown(
            "You should not need to access any external websites. The AI only needs to show you a final recommendation; it does not have the ability to do anything else (e.g. place orders in the real world)."
        )


def load_transaction_data(example_idx):
    """Load customer info and items for a given example index."""
    # Load customer info
    customer_file = f"{DATASET_ROOT}/assets/transactions/{example_idx}_customer.json"
    with open(customer_file, "r") as f:
        customer_info = json.load(f)

    # Load items
    items_file = f"{DATASET_ROOT}/assets/transactions/{example_idx}_items.csv"
    items_df = pd.read_csv(items_file, index_col=0).reset_index(drop=True)

    return customer_info, items_df


def format_customer_info(customer_info):
    """Format customer information for the prompt."""
    info_parts = []
    if customer_info.get("age"):
        info_parts.append(f"Age: {customer_info['age']}")
    info_parts.append(
        f"Club Member Status: {customer_info.get('club_member_status', 'NOT A MEMBER')}"
    )
    subscription = customer_info.get("FN", False)
    subscription = True if subscription == 1 else False
    info_parts.append(f"Fashion Newsletter subscription: {subscription}")
    info_parts.append(
        f"Fashion News Frequency: {customer_info.get('fashion_news_frequency', 'NEVER')}"
    )
    active = customer_info.get("Active", False)
    active = True if active == 1 else False
    info_parts.append(f"Subscribed to other communications: {active}")

    return "\n".join(info_parts)


def format_items(
    items_df: pd.DataFrame,
    features_of_interest: List[str],
    column_descriptions: Dict[str, str],
):
    """Format items information, omitting the specific article_id and prod_name and detail_desc."""
    if items_df.empty:
        return "No items purchased"

    assert "article_id" not in features_of_interest
    assert "prod_name" not in features_of_interest
    assert "detail_desc" not in features_of_interest

    item_descriptions = []
    for _, item in items_df.iterrows():
        desc_parts = []
        for feature in features_of_interest:
            if pd.notna(item.get(feature)):
                desc_parts.append(f"{column_descriptions[feature]}: {item[feature]}")

        item_descriptions.append("\n".join(desc_parts))

    return "\n\n".join(
        [
            f"<hr>- Item {i + 1} <hr>--\n{desc}"
            for i, desc in enumerate(item_descriptions)
        ]
    )

    """Legacy reward function is disabled. Use LinearFixedSpecification instead."""
    raise NotImplementedError(
        "Legacy reward_fn disabled; use LinearFixedSpecification for scoring."
    )


def get_actions(catalog: Catalog, true_products: pd.DataFrame) -> List[Action]:
    true_images = [
        catalog.get_image_by_article_id(article_id)
        for article_id in true_products["article_id"].tolist()
    ]

    @tool
    def describe_how_close(article_id: int) -> str:
        """
        Describes how close the predicted product is to a true product.
        If there are multiple true products, this describes the closest one.
        """
        test_img = catalog.get_image_by_article_id(article_id)
        scores = [clip_score(test_img, x) for x in true_images]
        best_match = np.max(scores)
        if best_match <= 0.1:
            return "not what I want at all"
        elif best_match <= 0.25:
            return "not what I want"
        elif best_match <= 0.5:
            return "not really what I want"
        elif best_match <= 0.8:
            return "somewhat similar to what I want"
        elif best_match <= 0.95:
            return "very close to what I want"
        else:
            return "basically exactly what I want"

    return [
        Action(
            fn=describe_how_close,
            is_public=False,
            is_human=False,
            name="Describe how close",
        )
    ]


def output_to_txt(
    msg: str,
    db: Catalog,
    render_cart: bool = True,
    render_items: bool = True,
) -> str:
    """
    Returns the rendered message in a text format.
    All items in <item> tags and all products in <cart> tags are rendered as JSONs from the catalog.
    """
    predicted_products = (
        parse_for_answer_tags(
            msg, keyword="cart", return_none_if_not_found=True, return_all=True
        )
        or []
    )
    mentioned_products = (
        parse_for_answer_tags(
            msg, keyword="item", return_all=True, return_none_if_not_found=True
        )
        or []
    )
    all_products = [p.split(",") for p in predicted_products + mentioned_products]
    all_products = [p for sublist in all_products for p in sublist]
    all_products = [int(p.strip()) for p in all_products if p.strip().isdigit()]
    all_products = list(dict.fromkeys(all_products))
    all_products_jsons = []
    for p in all_products:
        try:
            all_products_jsons.append(db.get_row_by_article_id(p).to_dict())
        except ValueError:
            all_products_jsons.append(
                {"article_id": p, "name": "Invalid product (not in catalog)"}
            )
    out = (
        msg
        + "\n\n------- Information about mentioned items ----------\n\n"
        + str(all_products_jsons)
    )
    return out


def output_to_streamlit(
    msg: str, db: Catalog, render_cart: bool = True, render_items: bool = True
) -> None:
    msg = msg.replace("$", "\$").replace("~", "\~")

    predicted_products = parse_for_answer_tags(
        msg, keyword="cart", return_none_if_not_found=True
    )
    mentioned_products = parse_for_answer_tags(
        msg, keyword="item", return_all=True, return_none_if_not_found=True
    )
    if mentioned_products is not None:
        mentioned_products = [
            int(id.strip())
            for mentions in mentioned_products
            for id in mentions.split(",")
            if id.strip().isdigit()
        ]
        mentioned_products = list(dict.fromkeys(mentioned_products))

    if not predicted_products and not mentioned_products:
        st.write(msg)
        return

    # Generate unique ID for this message to avoid conflicts when multiple messages are rendered
    message_hash = str(hash(msg))[:8]
    unique_id = f"mentioned-products-{message_hash}"

    if predicted_products and render_cart:
        start, end = msg.find("<cart>"), msg.find("</cart>") + len("</cart>")
        cart_string = (
            ("\n\n" if start > 0 else "")
            + renderer._shopping_recommendations_to_markdown(
                msg[start:end], db, header_type="cart"
            )
            + ("\n\n" if end < len(msg) else "")
        )
    else:
        cart_string = ""
        start, end = len(msg), len(msg)

    if mentioned_products and render_items:
        mention_string = renderer._shopping_recommendations_to_markdown(
            "<cart>" + ",".join(map(str, mentioned_products)) + "</cart>",
            db,
            header_type="mention",
        )
    else:
        mention_string = ""

    parts_to_render = [
        replace_tags_with_link(msg[:start], "item", f"#{unique_id}"),
        cart_string,
        replace_tags_with_link(msg[end:], "item", f"#{unique_id}"),
    ]
    parts_to_render = [p for p in parts_to_render if p]
    for p in parts_to_render:
        st.markdown(p, unsafe_allow_html=True)
    if mentioned_products:
        with st.expander("Items mentioned in message", expanded=True):
            st.markdown(f'<div id="{unique_id}"></div>', unsafe_allow_html=True)
            st.markdown(mention_string, unsafe_allow_html=True)


def output_to_streamlit_comparison(
    y1: str, y2: str, db: Catalog, validity_fn=None, validity_kwargs=None
) -> None:
    try:
        md = renderer.comparison_to_md(
            y1, y2, db, validity_fn=validity_fn, validity_kwargs=validity_kwargs
        )
    except Exception:
        st.write(y1)
        st.write(y2)
        return
    st.markdown(md, unsafe_allow_html=True)


def user_specification_callback(
    form_results: dict[str, Any], callback_kwargs: dict
) -> dict:
    """
    Process user form and generate hard constraints.
    Budget validation is now handled by the form itself with min/max values.
    """
    # Extract values from initial form
    budget = float(
        form_results.get("How much are you willing to spend? (Budget in dollars)", None)
    )
    section = form_results.get("What section would you like to shop in?", None)

    # Get y0 from mapping
    y0: List[str] = callback_kwargs.get("_y0_mapping", {}).get(section)
    if y0 is None:
        # pick the other carts from the same age group but different gender
        if section == "Mens":
            y0 = callback_kwargs.get("_y0_mapping", {}).get("Womens")
        elif section == "Womens":
            y0 = callback_kwargs.get("_y0_mapping", {}).get("Mens")
        elif section == "Boys":
            y0 = callback_kwargs.get("_y0_mapping", {}).get("Girls")
        elif section == "Girls":
            y0 = callback_kwargs.get("_y0_mapping", {}).get("Boys")
        else:
            y0 = None
    y0 = [f"<cart>{y}</cart>" for y in y0]

    # Create hard constraint for budget
    validity_kwargs = callback_kwargs.get("_validity_kwargs", {})
    hard_constraints = [
        Constraint.create_boolean_penalize_false_constraint(
            description="Shopping cart must match the specified prompt requirements",
            extractor="matches_prompt",
            extractor_kwargs={
                "catalog": callback_kwargs["_validity_kwargs"]["catalog"],
                "prompt": callback_kwargs["_prompt"],
            },
            is_hard=True,
        )
    ]
    if budget is not None:
        hard_constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"Total cost must not exceed \${budget:.2f}",
                extractor="total_cost",
                extractor_kwargs={
                    "catalog": callback_kwargs["_validity_kwargs"]["catalog"],
                    "budget": budget,
                },
                is_hard=True,
            )
        )

    # Update validity_kwargs with hard constraints
    hard_constraints = [
        Constraint.from_dict(
            c, extractor_lookup=callback_kwargs.get("_extractor_lookup", {})
        )
        for c in hard_constraints
    ]
    validity_kwargs["hard_constraints"] = hard_constraints

    # Get new specification from callback_kwargs
    new_specification = callback_kwargs.get("initial_specification") or ""
    if budget is not None:
        new_specification += f" | Budget: \${budget:.2f}"
    if section is not None:
        new_specification += f" | Section: {section}"

    # Return updates for the specification object
    return {
        "validity_kwargs": validity_kwargs,
        "y0": y0,
        "current_specification": new_specification,
        "_render_evaluation_kwargs": {
            "y0": y0,
            "budget": budget,
        },
    }


def validity_fn(
    shopping_cart: str,
    hard_constraints: List[Constraint],
    catalog,
    raise_errors: bool = False,
) -> Tuple[bool, dict]:
    """
    Check if the shopping cart is valid according to constraints.
    """
    try:
        # Check if shopping cart is empty
        if not shopping_cart or shopping_cart.strip() == "":
            if raise_errors:
                raise ValueError("Shopping cart is empty")
            return False, {"violated_constraints": ["Shopping cart is empty"]}

        # Check if all article IDs are valid
        shopping_cart = parse_for_answer_tags(
            shopping_cart, keyword="cart", return_none_if_not_found=True
        )
        if shopping_cart is None:
            if raise_errors:
                raise ValueError("Could not parse the shopping cart")
            return False, {
                "violated_constraints": ["Could not parse the shopping cart"]
            }

        article_ids = [
            int(id.strip()) for id in shopping_cart.split(",") if id.strip().isdigit()
        ]
        if not article_ids:
            if raise_errors:
                raise ValueError("No valid catalog IDs found")
            return False, {"violated_constraints": ["No valid catalog IDs found"]}

        # Check each article ID exists in catalog
        invalid_ids = []
        for article_id in article_ids:
            try:
                catalog.get_row_by_article_id(article_id)
            except ValueError:
                invalid_ids.append(str(article_id))

        if invalid_ids:
            if raise_errors:
                raise ValueError(f"Invalid catalog IDs: {', '.join(invalid_ids)}")
            return False, {
                "violated_constraints": [
                    f"Invalid catalog IDs: {', '.join(invalid_ids)}"
                ]
            }
    except Exception as e:
        if raise_errors:
            raise e
        return False, {"error": str(e)}

    # Check constraints
    (
        is_valid,
        score,
        min_unconstrained_score,
        max_unconstrained_score,
        metadata,
    ) = linear_reward(
        article_ids,
        constraints=hard_constraints,
        weights=None,
        enforce_hard=True,
        raise_errors=raise_errors,
    )
    return is_valid, metadata


# ------------------------
# Local helper to build soft features (descriptions + extractors)
# ------------------------


def _build_shopping_constraints(
    true_products: pd.DataFrame,
    catalog: Catalog,
    features_of_interest: List[str],
    budget: Optional[float] = None,
) -> List[dict]:
    """
    Build all shopping constraints (hard + soft) for a fixed spec instance.

    Hard constraints (type=boolean_penalize_false):
    - Budget: "Total cost must not exceed $B" (extractor: total_cost)
    - Catalog validity: "All items must exist in the catalog" (extractor: all_ids_valid)
    - Item count: "Cart must contain exactly N items" (extractor: num_items_equals)

    Soft constraints (preference features) and their constraint types:
    - Column overlaps (per feature_of_interest): type=multiset_jaccard
      extractor=column_values (pred column values), true_set=true column values (multiset)
    - Exact article overlap: type=multiset_jaccard
      extractor=identity_cart, true_set=true article_ids
    - Neckline / sleeve / dress length / material / fit / closure / leg style / gender / age:
      type=multiset_jaccard over tag multisets (concatenated tags per product)
      extractor=*_tags functions, true_set=true tag list
    - Hood presence / elasticity / sustainability:
      type=multiset_jaccard over flag multisets (0/1 per product)
      extractor=per_item_*_flags, true_set=true flag list
    - Product type count coverage: type=multiset_jaccard
      extractor=product_type_values, true_set=true product_type_name values (multiset)

    Notes:
    - multiset_jaccard computes Jaccard similarity between true and predicted multisets.
    - Hard constraints use penalize semantics and are enforced by validity.
    """
    specs: List[dict] = []

    # Hard constraints first
    if budget is not None:
        specs.append(
            {
                "type": "boolean_penalize_false",
                "description": f"Total cost must not exceed ${budget}",
                "is_hard": True,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "total_cost",
                "extractor_kwargs": {"catalog": catalog, "budget": budget},
                "none_val": 0,
            }
        )
    specs.append(
        {
            "type": "boolean_penalize_false",
            "description": "All items must exist in the catalog",
            "is_hard": True,
            "is_discoverable": True,
            "is_minimal": True,
            "extractor": "all_ids_valid",
            "extractor_kwargs": {"catalog": catalog},
            "none_val": 0,
        }
    )

    # Column-overlap features with target lists
    # Use helper functions from extractors module
    def _values(col: str, unique: bool = True) -> List[str]:
        return extractors._unique_values(true_products, col, unique=unique)

    _format_list = extractors._format_list

    for col in features_of_interest:
        # Build true_set as multiset (all values with duplicates)
        true_set = _values(col, unique=False)
        specs.append(
            {
                "type": "multiset_jaccard",
                "description": f"Desired {col}: {_format_list(_values(col, unique=False))}",
                "is_hard": (col in ["index_name", "product_type_name"]),
                "is_discoverable": True,
                "is_minimal": (col in ["product_type_name"]),
                "extractor": "column_values",
                "extractor_kwargs": {
                    "catalog": catalog,
                    "column": col,
                },
                "true_set": true_set,
            }
        )

    # Use helper function from extractors module
    def _derive_union_tags_df(keyword_map, applicable_fn=None) -> List[str]:
        return extractors._derive_union_tags_df(
            true_products, keyword_map, applicable_fn=applicable_fn
        )

    target_article_ids = (
        true_products["article_id"].astype(str).tolist()
        if "article_id" in true_products
        else []
    )
    # Only derive neckline/sleeve tags for upper or full body garments
    target_neckline = _derive_union_tags_df(
        extractors._NECKLINE_KEYWORDS, applicable_fn=extractors._is_upper_or_full
    )
    target_sleeve = _derive_union_tags_df(
        extractors._SLEEVE_LENGTH_KEYWORDS, applicable_fn=extractors._is_upper_or_full
    )
    # Only derive dress length for dresses/skirts
    target_dress_len = _derive_union_tags_df(
        extractors._DRESS_LENGTH_KEYWORDS, applicable_fn=extractors._is_dress_or_skirt
    )
    target_material = _derive_union_tags_df(extractors._MATERIAL_KEYWORDS)
    target_fit = _derive_union_tags_df(extractors._FIT_KEYWORDS)
    target_closure = _derive_union_tags_df(extractors._CLOSURE_KEYWORDS)
    target_leg = _derive_union_tags_df(
        extractors._LEG_STYLE_KEYWORDS, applicable_fn=extractors._is_lower_or_full
    )
    target_gender = _derive_union_tags_df(extractors._GENDER_KEYWORDS)
    target_sport = _derive_union_tags_df(extractors._SPORT_KEYWORDS)

    # Age tags need custom derivation logic
    def _derive_age_tags_df() -> List[str]:
        """
        Derive age tags per product and return as a list of tag strings.
        Each product's age tag is determined by priority: baby (specific) > kid > adult.
        """
        tags = []
        for _, row in true_products.iterrows():
            text = extractors._row_text(row)
            # Check for explicit baby sizes first (most specific - actual baby products)
            # But exclude "baby/children" which should be treated as kid
            if "baby/children" not in text and extractors._any_in(
                text, extractors._AGE_KEYWORDS["baby"]
            ):
                tags.append("baby")
            # Then check for kid (includes baby/children, children, kids, young boy/girl)
            elif extractors._any_in(text, extractors._AGE_KEYWORDS["kid"]):
                tags.append("kid")
            # Then check for adult (but only if no baby/kid indicators)
            elif extractors._any_in(text, extractors._AGE_KEYWORDS["adult"]):
                tags.append("adult")
            # Default: try to infer from context
            elif "menswear" in text or "ladieswear" in text or "womenswear" in text:
                tags.append("adult")
        return tags

    target_age = _derive_age_tags_df()

    # Use helper function from extractors module
    def _fraction_true_df(flag_fn) -> float:
        return extractors._fraction_true_df(true_products, flag_fn)

    winter_frac = _fraction_true_df(
        lambda r: (
            any(w in extractors._row_text(r) for w in extractors._WARM_KEYWORDS)
            or (
                "outerwear" in str(r.get("section_name", "")).lower()
                or "outer" in str(r.get("section_name", "")).lower()
            )
        )
        and not any(c in extractors._row_text(r) for c in extractors._COLD_KEYWORDS)
    )
    summer_frac = _fraction_true_df(
        lambda r: any(c in extractors._row_text(r) for c in extractors._COLD_KEYWORDS)
        and not any(w in extractors._row_text(r) for w in extractors._WARM_KEYWORDS)
    )
    hood_frac = _fraction_true_df(
        lambda r: (
            "hood" in extractors._row_text(r) or "hooded" in extractors._row_text(r)
        )
    )
    elastic_frac = _fraction_true_df(
        lambda r: any(
            k in extractors._row_text(r)
            for k in ["elastic", "elastication", "stretch", "spandex"]
        )
    )
    sustainable_frac = _fraction_true_df(
        lambda r: (
            "organic" in extractors._row_text(r)
            or "recycled" in extractors._row_text(r)
        )
    )

    # Derived soft features (conditionally include only if applicable)
    def _append_if(
        desc: str,
        extractor_name: str,
        ekw: Dict[str, Any],
        true_set: List[str],
        cond: bool,
    ):
        if not cond:
            return
        specs.append(
            {
                "type": "multiset_jaccard",
                "description": desc,
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": extractor_name,
                "extractor_kwargs": ekw,
                "true_set": true_set,
            }
        )

    specs.append(
        {
            "type": "boolean_penalize_false",
            "description": f"Cart must contain exactly {len(true_products)} items",
            "is_hard": True,
            "is_discoverable": True,
            "is_minimal": True,
            "extractor": "num_items_equals",
            "extractor_kwargs": {"true_products": true_products},
            "none_val": 0,
        }
    )
    if len(target_article_ids) > 0:
        specs.append(
            {
                "type": "multiset_jaccard",
                "description": "Prefer specific items from catalog",
                "is_hard": False,
                "is_discoverable": False,
                "is_minimal": False,
                "extractor": "identity_cart",
                "extractor_kwargs": {},
                "true_set": [int(x) for x in target_article_ids],
            }
        )
    _append_if(
        f"Desired neckline: {_format_list(target_neckline)}",
        "neckline_tags",
        {"catalog": catalog},
        target_neckline,
        len(target_neckline) > 0,
    )
    _append_if(
        f"Desired sleeve length: {_format_list(target_sleeve)}",
        "sleeve_length_tags",
        {"catalog": catalog},
        target_sleeve,
        len(target_sleeve) > 0,
    )
    _append_if(
        f"Desired dress/skirt length: {_format_list(target_dress_len)}",
        "dress_length_tags",
        {"catalog": catalog},
        target_dress_len,
        len(target_dress_len) > 0,
    )
    if winter_frac == 1.0:
        specs.append(
            {
                "type": "penalize_any_not_in_set",
                "description": "Penalize any non-winter-friendly items",
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "per_item_winter_flags",
                "extractor_kwargs": {"catalog": catalog},
                "required_set": [1],
                "none_val": 0.0,
            }
        )
    _append_if(
        f"Desired material: {_format_list(target_material)}",
        "material_tags",
        {"catalog": catalog},
        target_material,
        len(target_material) > 0,
    )
    _append_if(
        f"Desired fit: {_format_list(target_fit)}",
        "fit_tags",
        {"catalog": catalog},
        target_fit,
        len(target_fit) > 0,
    )
    _append_if(
        f"Desired closure: {_format_list(target_closure)}",
        "closure_tags",
        {"catalog": catalog},
        target_closure,
        len(target_closure) > 0,
    )
    # Build true multisets for fraction-based constraints (flags as 0/1)
    true_hood_flags = [
        1
        if ("hood" in extractors._row_text(r) or "hooded" in extractors._row_text(r))
        else 0
        for _, r in true_products.iterrows()
    ]
    true_elastic_flags = [
        1
        if any(
            k in extractors._row_text(r)
            for k in ["elastic", "elastication", "stretch", "spandex"]
        )
        else 0
        for _, r in true_products.iterrows()
    ]
    true_sustainable_flags = [
        1
        if (
            "organic" in extractors._row_text(r)
            or "recycled" in extractors._row_text(r)
        )
        else 0
        for _, r in true_products.iterrows()
    ]

    _append_if(
        f"Hooded items: {int(hood_frac * len(true_products))} / {len(true_products)}",
        "per_item_hood_flags",
        {"catalog": catalog},
        true_hood_flags,
        hood_frac > 0.0,
    )
    _append_if(
        f"Elastic/stretch items: {int(elastic_frac * len(true_products))} / {len(true_products)}",
        "per_item_elasticity_flags",
        {"catalog": catalog},
        true_elastic_flags,
        elastic_frac > 0.0,
    )
    _append_if(
        f"Sustainable (organic/recycled) items: {int(sustainable_frac * len(true_products))} / {len(true_products)}",
        "per_item_sustainability_flags",
        {"catalog": catalog},
        true_sustainable_flags,
        sustainable_frac > 0.0,
    )
    if summer_frac == 1.0:
        specs.append(
            {
                "type": "penalize_any_not_in_set",
                "description": "Penalize any non-summer-friendly items",
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "per_item_summer_flags",
                "extractor_kwargs": {"catalog": catalog},
                "required_set": [1],
                "none_val": 0.0,
            }
        )
    _append_if(
        f"Desired leg style: {_format_list(target_leg)}",
        "leg_style_tags",
        {"catalog": catalog},
        target_leg,
        len(target_leg) > 0,
    )
    _append_if(
        f"Desired gender: {_format_list(target_gender)}",
        "gender_tags",
        {"catalog": catalog},
        target_gender,
        len(target_gender) > 0,
    )
    _append_if(
        f"Desired age: {_format_list(target_age)}",
        "age_tags",
        {"catalog": catalog},
        target_age,
        len(target_age) > 0,
    )
    _append_if(
        f"Desired sport: {_format_list(target_sport)}",
        "sport_tags",
        {"catalog": catalog},
        target_sport,
        len(target_sport) > 0,
    )

    return specs


def _compute_shopping_feature_importances(
    true_products: pd.DataFrame, catalog: Catalog, features: List[Constraint]
) -> List[float]:
    """
    Compute per-feature importances based on consistency in the true cart.
    Higher when the true cart is more unanimous/consistent for that attribute.

    Rules:
    - Column value rewards (multiset_jaccard/column_values): 1 / #unique values in true column
    - Article-id reward (multiset_jaccard/identity_cart): 1 / #true items
    - Product type coverage (multiset_jaccard/product_type_values): 1 / #unique product_type_name
    - Tag overlaps (multiset_jaccard/*_tags): 1 / #distinct true tags (after concatenation)
    - Presence-like flags (multiset_jaccard/per_item_*_flags): |2·p - 1| where p is true fraction
    - Unanimity-based penalize_any_not_in_set (winter/summer/sport): 1.0
    - Hard constraints: 1.0 (ignored later by hard penalty)
    """

    def _unique_count(col: str) -> int:
        if col in true_products:
            return int(true_products[col].dropna().astype(str).nunique())
        return 0

    def _tag_consistency(keyword_map) -> float:
        # Use concatenated tags to match multiset_jaccard behavior
        tags = set()
        for _, row in true_products.iterrows():
            tag_str = extractors._derive_tags_concatenated(
                extractors._row_text(row), keyword_map
            )
            if tag_str:
                tags.add(tag_str)
        k = max(1, len(tags))
        return 1.0 / float(k)

    def _fraction_flag(flag_fn) -> float:
        vals = []
        for _, r in true_products.iterrows():
            vals.append(1 if flag_fn(r) else 0)
        if not vals:
            return 0.0
        p = sum(vals) / len(vals)
        return abs(2 * p - 1)

    imps: List[float] = []
    for c in features:
        if c.is_hard:
            imps.append(1.0)
            continue
        ename = getattr(c.extractor, "__name__", "")
        ctype = getattr(c, "type", "")
        imp = 0.5
        if ctype == "multiset_jaccard" and ename == "column_values":
            col = (getattr(c, "extractor_kwargs", {}) or {}).get("column")
            k = _unique_count(col) if col is not None else 0
            imp = 1.0 / float(max(1, k))
        elif ctype == "multiset_jaccard" and ename == "identity_cart":
            imp = 1.0 / float(max(1, len(true_products)))
        elif ctype == "multiset_jaccard" and ename == "product_type_values":
            k = _unique_count("product_type_name")
            imp = 1.0 / float(max(1, k))
        elif ctype == "multiset_jaccard" and ename == "neckline_tags":
            imp = _tag_consistency(extractors._NECKLINE_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "sleeve_length_tags":
            imp = _tag_consistency(extractors._SLEEVE_LENGTH_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "dress_length_tags":
            imp = _tag_consistency(extractors._DRESS_LENGTH_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "material_tags":
            imp = _tag_consistency(extractors._MATERIAL_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "fit_tags":
            imp = _tag_consistency(extractors._FIT_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "closure_tags":
            imp = _tag_consistency(extractors._CLOSURE_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "leg_style_tags":
            imp = _tag_consistency(extractors._LEG_STYLE_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "gender_tags":
            imp = _tag_consistency(extractors._GENDER_KEYWORDS)
        elif ctype == "multiset_jaccard" and ename == "age_tags":
            # Age tags are single values per product, so use unique count
            true_age_set = set()
            for _, row in true_products.iterrows():
                text = extractors._row_text(row)
                if "baby/children" not in text and extractors._any_in(
                    text, extractors._AGE_KEYWORDS["baby"]
                ):
                    true_age_set.add("baby")
                elif extractors._any_in(text, extractors._AGE_KEYWORDS["kid"]):
                    true_age_set.add("kid")
                elif extractors._any_in(text, extractors._AGE_KEYWORDS["adult"]):
                    true_age_set.add("adult")
                elif "menswear" in text or "ladieswear" in text or "womenswear" in text:
                    true_age_set.add("adult")
            k = max(1, len(true_age_set))
            imp = 1.0 / float(k)
        elif ctype == "multiset_jaccard" and ename == "per_item_hood_flags":
            imp = _fraction_flag(
                lambda r: (
                    "hood" in extractors._row_text(r)
                    or "hooded" in extractors._row_text(r)
                )
            )
        elif ctype == "multiset_jaccard" and ename == "per_item_elasticity_flags":
            imp = _fraction_flag(
                lambda r: any(
                    k in extractors._row_text(r)
                    for k in ["elastic", "elastication", "stretch", "spandex"]
                )
            )
        elif ctype == "multiset_jaccard" and ename == "per_item_sustainability_flags":
            imp = _fraction_flag(
                lambda r: (
                    "organic" in extractors._row_text(r)
                    or "recycled" in extractors._row_text(r)
                )
            )
        elif ctype == "penalize_any_not_in_set":
            imp = 1.0
        imps.append(float(max(1e-6, imp)))
    return imps


def _parse_shopping_solutions(msg: str, **kwargs) -> List[str]:
    """
    Extract complete shopping cart solutions from a message (does not include individual item mentions).
    Treat <cart>...</cart> as a solution.
    """
    ys: List[str] = []
    cart = parse_for_answer_tags(msg, keyword="cart", return_none_if_not_found=True)
    if cart:
        ys.append(f"<cart>{cart}</cart>")
    # dedup while preserving order
    return list(dict.fromkeys(ys))


def _parse_shopping_solutions_and_options(msg: str, **kwargs) -> List[str]:
    """
    Extract both complete shopping cart solutions and individual item mentions from a message.
    Treat <cart>...</cart> as a solution and each <item>...</item> group as a solution, but replace the tags with <cart>.
    """
    ys: List[str] = []
    cart = parse_for_answer_tags(msg, keyword="cart", return_none_if_not_found=True)
    if cart:
        ys.append(f"<cart>{cart}</cart>")
    items = parse_for_answer_tags(
        msg, keyword="item", return_all=True, return_none_if_not_found=True
    )
    if items:
        for group in items:
            ys.append(f"<cart>{group}</cart>")
    # dedup while preserving order
    return list(dict.fromkeys(ys))


def _parse_y_fn(yhat: str, catalog: Catalog, raise_errors: bool = False) -> Any:
    """
    Parse the solution attempt.
    """
    try:
        # Check if all article IDs are valid
        shopping_cart = parse_for_answer_tags(
            yhat, keyword="cart", return_none_if_not_found=True
        )
        if shopping_cart is None:
            if raise_errors:
                raise ValueError("Could not parse the shopping cart")
            return None

        article_ids = [
            int(id.strip()) for id in shopping_cart.split(",") if id.strip().isdigit()
        ]
        if not article_ids:
            if raise_errors:
                raise ValueError("No valid catalog IDs found")
            return None

        # Check each article ID exists in catalog
        invalid_ids = []
        for article_id in article_ids:
            try:
                catalog.get_row_by_article_id(article_id)
            except ValueError:
                invalid_ids.append(str(article_id))

        if invalid_ids:
            if raise_errors:
                raise ValueError(f"Invalid catalog IDs: {', '.join(invalid_ids)}")
            return None
    except Exception as e:
        if raise_errors:
            raise e
        return None

    return article_ids
