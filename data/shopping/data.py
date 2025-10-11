import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from langchain_core.tools import tool
import pandas as pd
import json
from PIL import Image
from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
)
from data.database import Database
from data.actions import Action, get_jupyter_actions
from collections import Counter
import inflect
from data.reward import linear_reward
from utils.misc import (
    hash,
    add_section,
    parse_for_answer_tags,
    replace_tags_with_link,
)
from utils.streamlit_types import FormElement, form_element_to_streamlit
from data.reward import Constraint
from data.shopping.reward_utils.helpers import soft_jaccard, clip_score
import data.shopping.streamlit_render as renderer
import streamlit as st
from typing import Callable

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

COMMONSENSE_DESCRIPTION = "You are actually putting items in the shopping cart, so you should make sure the cart fits within budget. Additionally, ONLY real products from the H&M catalog are valid."

PREDICTION_FMT_INSTRUCTIONS = "Return the article_ids of the products to recommend to the customer, separated by commas and wrapped in <cart></cart>, e.g.: '<cart>123456,123457,123458</cart>'."

MSG_FMT_INSTRUCTIONS = "Communicate with the user in language. To render a widget description of a single product, including a picture of the product, you can mention its article_id and wrap it in <item></item>, e.g.: '<item>123456</item>'. This will append a widget describing the product at the end of your message."


def render_fixed_task_explanation():
    """Render the fixed task explanation for shopping."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


class Catalog(Database):
    def __init__(self):
        # Load the catalog data
        df = pd.read_csv(f"{DATASET_ROOT}/assets/catalog.csv")

        # Load column descriptions
        column_descriptions = json.load(
            open(f"{DATASET_ROOT}/assets/column_descriptions.json")
        )

        # Initialize the Database parent class
        super().__init__(
            {
                "catalog": (
                    "Catalog of products from H&M",
                    df,
                    column_descriptions,
                )
            }
        )

        # Store the dataframe for backward compatibility
        self.df = df

    def get_row_by_article_id(self, article_id: int) -> pd.Series:
        article_id = int(article_id)
        matches = self.df[self.df["article_id"] == article_id]
        if len(matches) == 0:
            raise ValueError(f"Catalog ID {article_id} not found in catalog")
        return matches.iloc[0]

    def get_image_by_article_id(self, article_id: int) -> Image.Image:
        article_id = str(article_id)
        try:
            img_path = (
                f"{DATASET_ROOT}/assets/images/0{article_id[:2]}/0{article_id}.jpg"
            )
            return Image.open(img_path)
        except FileNotFoundError:
            return None


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

    def _create_user_expertise_form(self) -> List[FormElement]:
        """
        Create user expertise form elements for fashion and shopping knowledge.
        """
        return [
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

    def _create_user_specification_form_initial(
        self, intent_data: Dict = None
    ) -> List[FormElement]:
        """
        Create initial form elements for budget and demographics.
        """
        return [
            FormElement(
                input_type="text",
                label="Think of a person you'd like to shop for (yourself, a family member, a friend, etc.). Imagine you are buying a gift for this person.",
            ),
            FormElement(
                input_type="slider",
                label="How much are you willing to spend? (Budget in dollars)",
                required=True,
                help=f"Your maximum budget for this shopping trip (minimum: \${30:.2f})",
                min_value=30,
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
            ystar = ",".join(items_df["article_id"].astype(str).tolist())
            budget = round(items_df["price"].sum(), -1)

            signature = f"The task is to help the customer who is shopping identify the best products to buy. The customer's intent is: ``{intent}``. Their budget is {budget}."
            product_type_counts = items_df["product_type_name"].value_counts()
            signature += (
                " They are looking to buy the following products: "
                + ", ".join([f"{k} ({v})" for k, v in product_type_counts.items()])
            )

            theta = add_section(
                "Desired attributes",
                format_items(
                    items_df,
                    FEATURES_OF_INTEREST,
                    self._catalog.column_descriptions["catalog"],
                ).replace("\n\n", "\n<chunk>\n"),
            )
            theta += "\n<chunk>\n" + add_section(
                "Customer information", format_customer_info(customer_info)
            )

            initial_constraints = [
                Constraint.create_boolean_penalize_false_constraint(
                    description=f"Total cost must not exceed \${budget}",
                    extractor="total_cost",
                    extractor_kwargs={
                        "catalog": self._catalog,
                        "budget": budget,
                    },
                    is_hard=True,
                )
            ]
            initial_constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in initial_constraints
            ]

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

            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": initial_constraints,
                    "catalog": self._catalog,
                },
                validity_fn_tool_name="check_shopping_cart_validity",
                validity_fn_tool_description="Check if the shopping cart is valid and within budget",
                reward_fn=reward_fn,
                reward_kwargs={
                    "true_products": items_df,
                    "catalog": self._catalog,
                    "features_of_interest": FEATURES_OF_INTEREST,
                    "budget": budget,
                },
                reward_fn_tool_name="score_shopping_cart",
                reward_fn_tool_description="Score the shopping cart based on product matching",
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
                render_msg_kwargs=["db"],
                db=self._catalog,
                name=f"shopping_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
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

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification=f"Buy {prompt_as_str} from H&M tailored for the person you have in mind. You can assume that all products are available in all sizes.",
                user_specification_form_initial=self._create_user_specification_form_initial(
                    custom_intent
                ),
                user_specification_form_final=[],
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
                render_msg_fn=lambda msg, db: output_to_streamlit(
                    msg, db, render_cart=False
                ),
                render_msg_kwargs=["db"],
                db=self._catalog,
                render_comparison_fn=output_to_streamlit_comparison,
                name=f"custom_shopping_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
                _y0_mapping=custom_intent["y0"],
                _prompt=prompt,
                _extractor_lookup=self._extractor_lookup,
                render_evaluation_fn=lambda **kwargs: renderer.render_eval(
                    **kwargs,
                    db=self._catalog,
                ),
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


def reward_fn(
    predicted_products: str,
    true_products: pd.DataFrame,
    features_of_interest: List[str],
    catalog: Catalog,
    budget: int,
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    """
    Reward function for the shopping dataset.
    Returns:
        - score: float in [0, 100]
        - info: dict
            - optimal_matching: list of tuples (predicted_article_id, true_article_id)
    """

    predicted_products = parse_for_answer_tags(predicted_products, keyword="cart")
    if predicted_products is None:
        if raise_errors:
            raise ValueError(
                "Could not parse the predicted products. Make sure to wrap the article_ids in <cart></cart> tags."
            )
        return float("-inf"), {"error": "Could not parse the predicted products"}

    predicted_products = set(predicted_products.split(","))
    if len(predicted_products) == 0:
        if raise_errors:
            raise ValueError("No predicted products")
        return float("-inf"), {"error": "No predicted products"}

    if raise_errors:
        for p in predicted_products:
            try:
                int(p)
            except ValueError:
                raise ValueError(
                    f"Invalid predicted product: '{p}'. Make sure to return the article_ids of the products to recommend to the customer, separated by commas, e.g.: '123456,123457,123458'."
                )

    if predicted_products == set(true_products["article_id"].astype(str).tolist()):
        return 100.0, {"parsed_yhat": predicted_products}

    try:
        predicted_series = [
            (
                catalog.get_row_by_article_id(p),
                catalog.get_image_by_article_id(p),
            )
            for p in predicted_products
        ]
    except ValueError as e:
        if raise_errors:
            raise e
        return float("-inf"), {"error": str(e)}

    true_series = [
        (
            true_products.iloc[i],
            catalog.get_image_by_article_id(true_products.iloc[i]["article_id"]),
        )
        for i in range(len(true_products))
    ]

    def sim_fn(x, y):
        return product_match(x, y, features_of_interest)

    score, optimal_matching, sim_matrix = soft_jaccard(
        predicted_series, true_series, sim_fn
    )
    optimal_matching = [
        (predicted_series[i][0].article_id, true_series[j][0].article_id)
        for i, j in optimal_matching
    ]

    return (
        score * 100,  # scale to 0-100
        {
            "optimal_matching": optimal_matching,
            "parsed_yhat": predicted_products,
            "sim_matrix": sim_matrix,
        },
    )


def product_match(
    predicted_product: Tuple[pd.Series, Image.Image],
    true_product: Tuple[pd.Series, Image.Image],
    features_of_interest: List[str],
) -> float:
    """
    Compute the % of features of interest that match between the predicted and true products.
    """
    if predicted_product[0] is None or true_product[0] is None:
        return 0
    if "article_id" not in features_of_interest:
        features_of_interest.append("article_id")
    if "prod_name" not in features_of_interest:
        features_of_interest.append("prod_name")

    predicted_series, predicted_image = predicted_product
    true_series, true_image = true_product

    feature_score = np.mean(
        [
            predicted_series[feature] == true_series[feature]
            for feature in features_of_interest
        ]
    )

    if (
        VISUAL_SCORE_WEIGHT > 0
        and predicted_image is not None
        and true_image is not None
    ):
        # only compute if we're going to need it
        visual_score = clip_score(
            predicted_image,
            true_image,
        )
    else:
        visual_score = 0

    return (
        feature_score * (1 - VISUAL_SCORE_WEIGHT) + visual_score * VISUAL_SCORE_WEIGHT
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


def output_to_streamlit(
    msg: str, db: Catalog, render_cart: bool = True, render_items: bool = True
) -> None:
    msg = msg.replace("$", "\$")

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
        mentioned_products = list(set(mentioned_products))

    if not predicted_products and not mentioned_products:
        st.write(msg)
        return

    # Generate unique ID for this message to avoid conflicts when multiple messages are rendered
    message_hash = hash(msg)[:8]
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
