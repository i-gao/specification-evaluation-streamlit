# from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import asdict
import os
import streamlit as st
import json

from data.reward import Constraint, linear_reward
from data.dataset import (
    SpecificationCollection,
    LinearFixedSpecification,
    CustomSpecification,
)
from utils.misc import (
    parse_json,
    subset_data,
    add_section,
    parse_for_answer_tags,
    replace_tags_with_link,
)
from utils.streamlit_types import FormElement
from data.meal_planning.db import (
    RecipeDB,
    DAYS_OF_THE_WEEK,
    MEALS_OF_THE_DAY,
    DIETS,
    INTOLERANCES,
)
from data.meal_planning.parser import parse_meal_plan
from data.meal_planning.nutrition_utils import (
    convert_height_to_cm,
    convert_weight_to_kg,
    get_target_calories,
    get_healthy_carb_range,
    get_healthy_protein_range,
    get_healthy_fat_range,
)
from data.actions import get_jupyter_actions
import data.meal_planning.streamlit_render as renderer


"""
Ideas for future work:
- Harder version: cooking for more than 1 person
"""

DEV_FRAC = 0.8
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
PREDICTION_FMT_INSTRUCTIONS = (
    """The meal plan is returned as a JSON string wrapped in <meal_plan></meal_plan> tags with the following structure:
<meal_plan>
{
    "sunday": {
        "breakfast": [
            {
                "action": "cook" | "eat",
                "recipe_title": str,
            },
            ...
        ],
        ...
    },
    ...
}
</meal_plan>
""".strip()
    + f"\nThe outer keys are the days of the week (selected from {DAYS_OF_THE_WEEK}), and the inner keys are the meals of the day (selected from {MEALS_OF_THE_DAY}; note that 'snack' occurs before 'dinner')."
    + """Each meal slot contains an (ordered) list of actions:
    1. One kind of action is to cook a recipe: the `action` field is set to "cook" and `recipe_title` is set to the title of a recipe from the database. The database specifies how many servings each cook action makes. It is assumed that servings are refrigerated, and that leftovers do not expire. Note that cooking does not automatically include eating: a separate "eat" action must be added to the meal plan.
    2. Another kind of action is to eat a recipe: the `action` field is set to "eat" and `recipe_title` is set to the title of a recipe from the database. Recipes can only be eaten if they have been cooked. The number of servings eaten must not exceed the number of servings made by the recipe.
Note that recipes do not need to be cooked and eaten at the same time: a recipe may be cooked at one meal, and then eaten at another meal. 

If a slot in the meal plan is left empty or if the slot is omitted entirely, it means that no recipe is cooked or eaten for that meal. The two are exchangeable.
""".strip()
)

MSG_FMT_INSTRUCTIONS = """Always wrap recipe titles in <recipe></recipe>, e.g.: '<recipe>Chicken Parmesan</recipe>'. This will append a widget describing the recipe at the end of your message so the user can view the recipe. You should always do this by default."""


def get_commonsense_description():
    """Get the commonsense description, dynamically using the number of days."""
    days_str = ", ".join([d.capitalize() for d in DAYS_OF_THE_WEEK])
    last_day = DAYS_OF_THE_WEEK[-1].capitalize()
    return f"""A meal plan goes through every meal slot of the day and specifies what (if anything) is cooked, as well as what (and how many servings) is eaten. Recipes may be cooked and then eaten in the same meal, or cooked and the remaining servings eaten in a later meal (to be reheated later). Meal plans must carefully track how many servings each recipe makes and ensure that no more than that number of servings are consumed throughout the {len(DAYS_OF_THE_WEEK)}-day period. Recipes cannot be halved or doubled: they make exactly the listed number of servings. Servings can only be consumed in integer amounts; 1.5 servings of a recipe cannot be consumed. Recipes can be cooked more than once. 
Multiple recipes can be cooked at each mealtime (e.g. a drink, a main course, and a dessert); the total cooking time will be the sum of the total times for each recipe. 

If a slot in the meal plan is left empty or if the slot is omitted entirely, it means that no recipe is cooked or eaten for that meal. The two are exchangeable.

The meal plan covers {days_str}. All servings leftover at the end of {last_day} will be wasted.

The environment provides a Jupyter notebook and a CSV of recipes from AllRecipes.com. Meal plans must use recipes from the provided AllRecipes catalog. Using other recipes is not allowed. """


COMMONSENSE_DESCRIPTION = get_commonsense_description()


def get_custom_instructions():
    """Get custom instructions, dynamically using the number of days."""
    days_str = ", ".join([d.capitalize() for d in DAYS_OF_THE_WEEK])
    return f"""
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write you a perfect meal prep plan that you can actually follow for the next {len(DAYS_OF_THE_WEEK)} days ({days_str}).** A meal plan is a {len(DAYS_OF_THE_WEEK)}-day calendar that specifies what to eat for every meal of the day. The plan also specifies when to cook each recipe.

The plan must work with your schedule, dietary restrictions, and preferences.
"""


CUSTOM_INSTRUCTIONS = get_custom_instructions()


def render_fixed_task_explanation():
    """Render the fixed task explanation for meal planning."""
    st.markdown(get_commonsense_description())


def render_custom_task_explanation():
    """Render the custom task explanation for meal planning."""
    st.markdown(CUSTOM_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


class MealPlanningDataset(SpecificationCollection):
    """
    The MealPlanning benchmark evaluates how well LMs can generate personalized
    meal plans which obey some constraints.

    Original recipes: scraped from AllRecipes

    Profiles: programmatically generated
    Each profile is treated as a separate "task" to solve. A profile consists of information
    about a person, including their dietary restrictions, intolerances, and preferences.
    - The dataset consists of a single (x), representing the set of available recipes.
    """

    @property
    def dataset_name(self) -> str:
        return "meal_planning"

    @property
    def dataset_pretty_name(self) -> str:
        return "Meal Planning"

    @property
    def dataset_description(self) -> str:
        days_str = ", ".join([d.capitalize() for d in DAYS_OF_THE_WEEK])
        return f"Work with the assistant to **write a personal meal plan for the next {len(DAYS_OF_THE_WEEK)} days ({days_str})** using real recipes from AllRecipes.com."

    @property
    def assets_file_id(self) -> str:
        return "14BMzVj-YMdjAF5UIhbNaxHlQXIdobCqn"

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
                "image_name": "meal_planning",
                "dockerfile_path": "data/meal_planning/reward_utils/Dockerfile",
                "build_context": "data/meal_planning",
                "description": "Docker image for Meal Planning code evaluation",
            },
        ]

    def _create_user_expertise_form(self) -> List[FormElement]:
        """
        Create user expertise form elements for meal planning domain knowledge.
        """
        return [
            FormElement(
                input_type="radio",
                label="How many of your meals do you cook?",
                options=[
                    "I have never cooked before",
                    "I do not regularly cook",
                    "A few days a week",
                    "Most days of the week",
                ],
                required=True,
                help="This helps us understand your experience level with cooking",
            ),
            FormElement(
                input_type="radio",
                label="How familiar are you with your nutritional needs?",
                options=[
                    "I have never thought about what calorie, macronutrients, or vitamins are",
                    "I have basic knowledge of what calories / macronutrients / vitamins are",
                    "I have some knowledge of how many calories / macronutrients / vitamins I personally need",
                    "I actively try to eat the right amount of calories / macronutrients / vitamins",
                ],
                required=True,
                help="This helps us understand your experience level with nutrition",
            ),
        ]

    def __init__(
        self,
        dev: bool = False,
        docker_image: str = None,
        fixed_indexes: Optional[List[int]] = None,
        custom_indexes: Optional[List[int]] = None,
        persist_docker_container: bool = True,
        auto_patch_eat_before_cook: bool = True,
        eval_num_items_per_comparison: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)
        self.eval_num_items_per_comparison = eval_num_items_per_comparison

        self._docker_image = docker_image
        self._persist_docker_container = persist_docker_container
        self._auto_patch_eat_before_cook = auto_patch_eat_before_cook

        # Load all the fixed spec profiles
        with open(f"{DATASET_ROOT}/assets/profiles.json", "r") as f:
            profiles = json.load(f)
        # Convert all_weights lists back to numpy arrays
        for profile in profiles:
            profile["all_weights"] = np.array(profile["all_weights"])
        profiles = subset_data(profiles, DEV_FRAC, 1.0, dev)
        self._profiles = profiles
        self.fixed_length = len(self._profiles)

        # Only 1 custom spec
        self.custom_length = 1

        # Import extractors and build lookup
        import data.meal_planning.extractors as extractors_mod

        self._extractor_lookup = {
            name: func
            for name, func in extractors_mod.__dict__.items()
            if callable(func)
        }

        # No ystars
        self._ystars = {}
        y0_mapping_raw = {
            k: json.load(open(f"{DATASET_ROOT}/assets/{k}.json"))
            for k in ["vegetarian", "gluten-free", "normal", "vegan"]
        }
        # Filter y0s to only include days in DAYS_OF_THE_WEEK
        self._y0_mapping = self._filter_y0_mapping(y0_mapping_raw)

        # Load the recipes database to get column information
        self._recipe_db = RecipeDB()
        self._desc_json = {
            "filename": "recipes.csv",
            "description": "Database of recipes from AllRecipes.com",
            "columns": self._recipe_db._list_columns("recipes"),
        }

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)
        if custom_indexes is not None:
            self.load_custom_specs(indexes=custom_indexes)

    def _filter_y0_mapping(self, y0_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter y0 meal plans to only include days in DAYS_OF_THE_WEEK.
        This ensures y0s match the current day configuration.
        """
        filtered_mapping = {}
        for key, y0_value in y0_mapping.items():
            if y0_value is None:
                filtered_mapping[key] = None
                continue

            # Parse the y0 if it's a string
            if isinstance(y0_value, str):
                try:
                    y0_dict = json.loads(y0_value)
                except (json.JSONDecodeError, TypeError):
                    filtered_mapping[key] = y0_value
                    continue
            elif isinstance(y0_value, dict):
                y0_dict = y0_value
            else:
                filtered_mapping[key] = y0_value
                continue

            # Filter to only include days in DAYS_OF_THE_WEEK
            filtered_y0 = {
                day.lower(): y0_dict.get(day.lower(), y0_dict.get(day, {}))
                for day in DAYS_OF_THE_WEEK
            }

            # Convert back to string if original was string
            if isinstance(y0_value, str):
                filtered_mapping[key] = json.dumps(filtered_y0)
            else:
                filtered_mapping[key] = filtered_y0

        return filtered_mapping

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, LinearFixedSpecification]:
        if indexes is None:
            return {}

        # Load requested specs
        specs = {}
        for ix in indexes:
            profile = self._profiles[ix]

            dem = profile["demographic_information"]
            days_str = ", ".join([d.capitalize() for d in DAYS_OF_THE_WEEK])
            signature = f"The task is to write out a {len(DAYS_OF_THE_WEEK)}-day meal plan ({days_str}). The client only eats {' and '.join(dem['meals_considered'])} each day."
            signature += "\n\nDemographic information: " + (
                f"Sex: {dem['sex']}\n"
                + f"Weight: {dem['weight']} kg\n"
                + f"Height: {dem['height']} cm\n"
                + f"Age: {dem['age']}\n"
                + f"Activity level: {dem['activity_level']}\n"
                + f"Goal: {dem['goal']} weight"
            )

            # constraints & theta
            constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in profile["constraints"]
            ]

            # actions
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

            # Use all_weights from profile (hard constraints get 99999, soft get random weights)
            # All weights should be positive; linear_reward will handle sign correction for penalty constraints
            if "all_weights" in profile and len(profile["all_weights"]) == len(
                constraints
            ):
                weights = [
                    abs(float(w)) for w in profile["all_weights"]
                ]  # Use absolute value
            else:
                raise ValueError(
                    f"Profile must contain 'all_weights' with length matching constraints. "
                    f"Found {len(profile.get('all_weights', []))} weights for {len(constraints)} constraints."
                )

            # Create parse_y_fn wrapper that captures recipe_db and auto_patch_eat_before_cook
            def parse_meal_plan_wrapper(
                yhat: str, raise_errors: bool = False
            ) -> Optional[Any]:
                """Parse meal plan from string."""
                return parse_meal_plan(
                    yhat,
                    self._recipe_db,
                    raise_errors=raise_errors,
                    auto_patch_eat_before_cook=self._auto_patch_eat_before_cook,
                )

            # specification
            spec = LinearFixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                initial_specification=signature,
                commonsense_description=get_commonsense_description(),
                features=constraints,
                weights=weights,
                parse_y_fn=parse_meal_plan_wrapper,
                parse_solutions_fn=parse_meal_plan_solutions,
                parse_solutions_and_options_fn=parse_meal_plan_solutions_and_options,
                validity_fn_tool_name="check_meal_plan_validity",
                validity_fn_tool_description="Check if the meal plan satisfies all hard constraints",
                reward_fn_tool_name="score_meal_plan",
                reward_fn_tool_description="Score the meal plan based on preference constraints",
                ystar=self._ystars.get(str(ix), None),
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=actions,
                msg_fmt_instructions=MSG_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_fn_txt=output_to_txt,
                render_msg_kwargs=["db", "auto_patch_eat_before_cook"],
                name=f"meal_planning_{ix}",
                db=self._recipe_db,
                auto_patch_eat_before_cook=self._auto_patch_eat_before_cook,
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
            )
            specs[ix] = spec
        return specs

    def _load_custom_specs(self, indexes: Optional[List[int]] = None):
        """
        Create a skeleton meal planning specification without specific task details.

        Args:
            docker_image (str): The Docker image to use for the environment

        Returns:
            Specification: A skeleton specification for meal planning
        """
        if indexes is None:
            return {}

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

            # Always start with the two fixed constraints
            constraints = [
                Constraint.create_boolean_penalize_false_constraint(
                    description="All recipes must be cooked before they are consumed",
                    extractor="check_recipes_eaten_after_cooked",
                    is_hard=True,
                ),
                Constraint.create_boolean_penalize_false_constraint(
                    description=f"The total number of servings consumed of a recipe across the {len(DAYS_OF_THE_WEEK)}-day period must be <= the total number of servings cooked of the recipe",
                    extractor="check_servings_consumed_lt_cooked_total",
                    is_hard=True,
                ),
            ]
            constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in constraints
            ]
            initial_specification = f"Generate a meal plan for yourself for the next {len(DAYS_OF_THE_WEEK)} days ({', '.join([d.capitalize() for d in DAYS_OF_THE_WEEK])}). Only plan for 1 person (yourself)."

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification=initial_specification,
                current_specification=initial_specification,
                commonsense_description=get_commonsense_description(),
                user_specification_form_final=self._create_user_specification_form_final(),
                user_specification_callback=user_specification_callback,
                user_specification_callback_kwargs=[
                    "_validity_kwargs",
                    "_y0_mapping",
                    "_extractor_lookup",
                    "initial_specification",
                ],
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": constraints,
                    "recipe_db": self._recipe_db,
                    "auto_patch_eat_before_cook": self._auto_patch_eat_before_cook,
                },
                validity_fn_tool_name="check_meal_plan_validity",
                validity_fn_tool_description="Check if the meal plan satisfies all hard constraints",
                y0=None,  # Not provided
                render_task_explanation=self._render_custom_task_explanation,
                actions=actions,
                msg_fmt_instructions=MSG_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_fn_txt=output_to_txt,
                render_msg_kwargs=["db", "auto_patch_eat_before_cook"],
                db=self._recipe_db,
                auto_patch_eat_before_cook=self._auto_patch_eat_before_cook,
                render_comparison_fn=output_to_streamlit_comparison,
                name=f"custom_meal_planning_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
                _y0_mapping=self._y0_mapping,
                _extractor_lookup=self._extractor_lookup,
                render_evaluation_fn=lambda **kwargs: renderer.render_eval(
                    **kwargs,
                    db=self._recipe_db,
                ),
                render_evaluation_kwargs={
                    "num_items_per_comparison": getattr(self, "eval_num_items_per_comparison", 5),
                },
            )
            specs[ix] = spec
        return specs

    def _create_user_evaluation_form(self) -> List[FormElement]:
        """Create the user evaluation form for meal planning."""
        return [
            FormElement(
                input_type="radio",
                label="Compare the **recipe choices** of meal plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                default="0",
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **cooking times** of the recipes in meal plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **variety of cooking new meals vs. eating leftovers** in meal plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how well meal plans A and B **fit into your upcoming schedule,** accounting for your existing plans / time constraints. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **calorie totals** of meal plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **nutritional benefits** of meal plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label=f"Which meal plan are you more likely to follow for the next {len(DAYS_OF_THE_WEEK)} days: A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
        ]

    def _create_user_specification_form_final(self) -> List[FormElement]:
        """
        Create final form elements for detailed meal planning requirements.
        Includes all demographic information and constraints.
        """
        return [
            FormElement(
                input_type="radio",
                label="Sex",
                options=["Male", "Female"],
                default="Male",
                required=True,
                help="Your biological sex for nutritional calculations",
            ),
            FormElement(
                input_type="radio",
                label="Height Unit",
                options=["cm", "in"],
                default="cm",
                required=True,
                help="Select your preferred height unit",
            ),
            FormElement(
                input_type="text_input",
                label="Approximate height",
                required=True,
                help="Your height (in the selected unit)",
            ),
            FormElement(
                input_type="radio",
                label="Weight Unit",
                options=["kg", "lbs"],
                default="kg",
                required=True,
                help="Select your preferred weight unit",
            ),
            FormElement(
                input_type="number_input",
                label="Approximate weight",
                value=None,
                required=True,
                step=10,
                min_value=0,
                max_value=300,
                help="Your current weight (enter as '70' for kg or '154' for lbs)",
            ),
            FormElement(
                input_type="number_input",
                label="Approximate age (years)",
                value=None,
                required=True,
                step=1,
                min_value=10,
                max_value=100,
                help="Your age in years",
            ),
            FormElement(
                input_type="radio",
                label="Activity Level",
                options=[
                    "Sedentary",
                    "Lightly Active",
                    "Moderately Active",
                    "Very Active",
                    "Extremely Active",
                ],
                default="Moderately Active",
                required=True,
                help="Your daily activity level for nutritional calculations",
            ),
            FormElement(
                input_type="radio",
                label="Weight Goal",
                options=["Lose Weight", "Maintain Weight", "Gain Weight"],
                default="Maintain Weight",
                required=True,
                help="Your weight management goal",
            ),
            FormElement(
                input_type="selectbox",
                label="Dietary restrictions",
                options=[("diet", diet) for diet in DIETS]
                + [("intolerance", intolerance) for intolerance in INTOLERANCES],
                format_func=lambda x: x[1] + " " + x[0],
                required=False,
                help="Select any dietary restrictions or preferences",
            ),
        ]

    def _render_custom_task_explanation(self):
        """Render the custom task explanation for meal planning."""

        st.markdown("### What you need to prompt the assistant to do")
        st.markdown(
            f"In this task, **your goal is to get the assistant to write you a perfect meal prep plan that you can actually follow for the next {len(DAYS_OF_THE_WEEK)} days ({', '.join([d.capitalize() for d in DAYS_OF_THE_WEEK])}).** A meal plan is a {len(DAYS_OF_THE_WEEK)}-day calendar that specifies what to eat for every meal of the day. The plan also specifies when to cook each recipe."
        )

        with st.container(border=True):
            example_plan = self._y0_mapping["normal"]
            st.info(f"*Example:* A {len(DAYS_OF_THE_WEEK)}-day meal plan")
            # Parse the example plan first
            parsed_plan = parse_meal_plan(
                json.dumps(example_plan), self._recipe_db, leave_invalid=True
            )
            st.markdown(
                renderer._render_calendar_table(parsed_plan),
                unsafe_allow_html=True,
            )
            with st.expander("🍳 Which days will I have to cook?", expanded=False):
                st.markdown(
                    renderer._render_cooking_calendar(parsed_plan),
                    unsafe_allow_html=True,
                )

        st.markdown(
            "The plan must work with your schedule, dietary restrictions, and preferences. You should make the meal plan assuming you only need to cook for yourself."
        )

        st.markdown(
            "Think about your dietary restrictions, cooking schedule, and food preferences. The assistant should personalize the meal plan to your needs, picking recipes that match your taste and dietary requirements."
        )
        st.markdown("### Making sure your meal plan is valid")
        st.markdown(
            "To successfully complete this task, your meal plan must *be valid.*"
        )

        st.markdown(
            "* A valid plan must ONLY use recipes from AllRecipes.com. Using other recipes is not allowed."
        )

        with st.container(border=True):
            # Example with invalid recipe
            plan = json.dumps(self._y0_mapping["normal"]).replace(
                "Hawaiian Pizza", "Made-up Recipe"
            )
            st.error(
                ":red[:material/close: *Example:* This is an invalid plan because it includes a made-up recipe, designated by the :material/error: icon]"
            )
            # Parse the invalid plan first
            parsed_invalid_plan = parse_meal_plan(
                plan, self._recipe_db, leave_invalid=True
            )
            st.markdown(
                renderer._render_calendar_table(parsed_invalid_plan),
                unsafe_allow_html=True,
            )

        st.markdown("* You can cook recipes and eat them later as leftovers.")

        # with st.container(border=True):
        #     dummy_plan = {
        #         "sunday": {
        #             "lunch": [
        #                 {
        #                     "action": "cook",
        #                     "recipe_title": "Vietnamese Pork And Five Spice",
        #                 },
        #                 {
        #                     "action": "eat",
        #                     "recipe_title": "Vietnamese Pork And Five Spice",
        #                 },
        #                 {
        #                     "action": "eat",
        #                     "recipe_title": "Vietnamese Pork And Five Spice",
        #                 },
        #             ],
        #         },
        #     }
        #     parsed_dummy_plan = parse_meal_plan(
        #         json.dumps(dummy_plan), self._recipe_db, leave_invalid=True
        #     )
        #     st.error(
        #         "*Example:* This meal plan is invalid because it consumes 2 servings of the Banh Mi sandwich, but the recipe only makes 1 serving."
        #     )
        #     st.markdown("### Sunday")
        #     renderer._render_day_details("sunday", parsed_dummy_plan)

        st.markdown(
            "* The assistant needs to respect your dietary restrictions and/or allergies. You will be able to see details about recipes in the recipe details section; these contain information about the allergens they contain and the diets they are compatible with."
        )

        with st.container(border=True):
            dummy_plan = {
                "sunday": {
                    "lunch": [
                        {
                            "action": "cook",
                            "recipe_title": "Vietnamese Pork And Five Spice",
                        },
                        {
                            "action": "eat",
                            "recipe_title": "Vietnamese Pork And Five Spice",
                        },
                        {
                            "action": "eat",
                            "recipe_title": "Vietnamese Pork And Five Spice",
                        },
                    ],
                },
            }
            parsed_dummy_plan = parse_meal_plan(
                json.dumps(dummy_plan), self._recipe_db, leave_invalid=True
            )
            renderer._render_recipe_details_streamlit(parsed_dummy_plan, "dummy_plan")


def user_specification_callback(
    form_results: dict[str, Any], callback_kwargs: dict
) -> dict:
    """
    Process form results and return updates for the specification.
    This callback handles both initial and final form results.
    """
    validity_kwargs = callback_kwargs.get("validity_kwargs", {})
    constraints = [
        Constraint.create_boolean_penalize_false_constraint(
            description="All recipes must be cooked before they are consumed",
            extractor="check_recipes_eaten_after_cooked",
            is_hard=True,
        ),
        Constraint.create_boolean_penalize_false_constraint(
            description="The total number of servings consumed of a recipe across the week must be <= the total number of servings cooked of the recipe",
            extractor="check_servings_consumed_lt_cooked_total",
            is_hard=True,
        ),
    ]
    # Convert height and weight to standard units (cm and kg)
    try:
        height_value = float(form_results.get("Approximate height", "170"))
        height_unit = form_results.get("Height Unit", "cm")
        height_cm = convert_height_to_cm(height_value, height_unit)

        weight_value = float(form_results.get("Approximate weight", "70"))
        weight_unit = form_results.get("Weight Unit", "kg")
        weight_kg = convert_weight_to_kg(weight_value, weight_unit)

        # Store converted values for potential future use
        form_results["height_cm"] = height_cm
        form_results["weight_kg"] = weight_kg
    except (ValueError, TypeError):
        # If conversion fails, use defaults
        form_results["height_cm"] = 170.0
        form_results["weight_kg"] = 70.0

    # Calculate nutritional targets based on demographic information
    try:
        age = int(form_results.get("Approximate age (years)", "30"))
        sex = form_results.get("Sex", "Male").lower()
        activity_level = (
            form_results.get("Activity Level", "Moderately Active")
            .lower()
            .replace(" ", "_")
        )
        goal = (
            form_results.get("Weight Goal", "Maintain Weight").lower().replace(" ", "")
        )

        # Calculate target calories
        target_calories = get_target_calories(
            form_results["weight_kg"],
            form_results["height_cm"],
            age,
            sex,
            activity_level,
            goal,
        )

        # Calculate healthy macronutrient ranges
        carb_range = get_healthy_carb_range(target_calories)
        protein_range = get_healthy_protein_range(target_calories)
        fat_range = get_healthy_fat_range(target_calories)

        # Store nutritional targets
        form_results["target_calories"] = target_calories
        form_results["carb_range"] = carb_range
        form_results["protein_range"] = protein_range
        form_results["fat_range"] = fat_range

    except (ValueError, TypeError):
        # If calculation fails, use defaults
        form_results["target_calories"] = 2000.0
        form_results["carb_range"] = (225.0, 325.0)  # 45-65% of 2000 cal
        form_results["protein_range"] = (50.0, 175.0)  # 10-35% of 2000 cal
        form_results["fat_range"] = (44.4, 77.8)  # 20-35% of 2000 cal

    # Add constraints from final form (if present)
    dietary_restrictions = form_results.get("Dietary restrictions", None)
    if dietary_restrictions:
        restriction_type, restriction_value = dietary_restrictions
        if restriction_type == "diet":
            constraints.append(
                Constraint.create_boolean_penalize_false_constraint(
                    description=f"Must follow {restriction_value} diet for all meals",
                    extractor="recipes_follow_diet",
                    extractor_kwargs={"diet": restriction_value},
                    is_hard=True,
                )
            )
        elif restriction_type == "intolerance":
            constraints.append(
                Constraint.create_boolean_penalize_false_constraint(
                    description=f"Must avoid {restriction_value} for all meals",
                    extractor="recipes_avoid_intolerance",
                    extractor_kwargs={"intolerance": restriction_value},
                    is_hard=True,
                )
            )

    # Add nutritional target constraints - using patterns from generate_profiles.py
    target_calories = form_results.get("target_calories")
    if target_calories is not None:
        # Calorie target constraints
        for day in DAYS_OF_THE_WEEK:
            constraints.append(
                Constraint.create_radial_band_constraint(
                    description=f"Aim for daily calorie target: {target_calories:.0f} calories on {day.capitalize()}",
                    extractor="daily_calories",
                    extractor_kwargs={"day": day},
                    lower=target_calories - 100,
                    upper=target_calories + 100,
                    sigma=100.0,
                    is_hard=False,
                )
            )

        # Macronutrient range constraints
        carb_range = form_results.get("carb_range")
        protein_range = form_results.get("protein_range")
        fat_range = form_results.get("fat_range")

        for day in DAYS_OF_THE_WEEK:
            # Carbohydrate constraints
            if carb_range:
                constraints.append(
                    Constraint.create_radial_band_constraint(
                        description=f"Healthy macronutrient range: aim for {carb_range[0]:.1f} to {carb_range[1]:.1f} grams of carbs on {day.capitalize()}",
                        lower=carb_range[0],
                        upper=carb_range[1],
                        sigma=10.0,
                        extractor="daily_carbohydrate",
                        extractor_kwargs={"day": day},
                        is_hard=False,
                    )
                )

            # Protein constraints
            if protein_range:
                constraints.append(
                    Constraint.create_radial_band_constraint(
                        description=f"Healthy macronutrient range: aim for {protein_range[0]:.1f} to {protein_range[1]:.1f} grams of protein on {day.capitalize()}",
                        lower=protein_range[0],
                        upper=protein_range[1],
                        sigma=10.0,
                        extractor="daily_protein",
                        extractor_kwargs={"day": day},
                        is_hard=False,
                    )
                )

            # Fat constraints
            if fat_range:
                constraints.append(
                    Constraint.create_radial_band_constraint(
                        description=f"Healthy macronutrient range: aim for {fat_range[0]:.1f} to {fat_range[1]:.1f} grams of fat on {day.capitalize()}",
                        lower=fat_range[0],
                        upper=fat_range[1],
                        sigma=10.0,
                        extractor="daily_total_fat",
                        extractor_kwargs={"day": day},
                        is_hard=False,
                    )
                )

    # Update validity_kwargs
    constraints = [
        Constraint.from_dict(
            c, extractor_lookup=callback_kwargs.get("_extractor_lookup", {})
        )
        for c in constraints
    ]
    validity_kwargs["hard_constraints"] = constraints

    # Get new specification from callback_kwargs
    new_specification = callback_kwargs.get("initial_specification") or ""
    if "Weight Goal" in form_results:
        new_specification += f" | Goal: {form_results['Weight Goal']}"
    if dietary_restrictions:
        new_specification += f" | Dietary restrictions: {dietary_restrictions[1]}"

    # y0
    y0_mapping = callback_kwargs.get("_y0_mapping", {})

    DIETS_TO_Y0 = {
        "Gluten free": "gluten-free",
        "Ketogenic": "gluten-free",
        "Vegetarian": "vegetarian",
        "Lacto-Vegetarian": "vegan",
        "Ovo-Vegetarian": "vegan",
        "Vegan": "vegan",
        "Pescetarian": "vegetarian",
        "Paleo": "gluten-free",
        "Primal": "gluten-free",
        "Whole30": "gluten-free",
        "Clean eating": "gluten-free",
        "Mediterranean": "gluten-free",
    }

    if dietary_restrictions and dietary_restrictions[0] == "diet":
        y0 = y0_mapping.get(DIETS_TO_Y0[dietary_restrictions[1]])
    else:
        y0 = y0_mapping.get("normal")

    # Wrap y0 in tags if it exists
    if y0 is not None:
        y0_str = json.dumps(y0) if isinstance(y0, dict) else y0
        y0 = f"<meal_plan>{y0_str}</meal_plan>"

    # Return updates for the specification object
    return {
        "validity_kwargs": validity_kwargs,
        "current_specification": new_specification,
        "y0": y0,
        "_render_evaluation_kwargs": {
            "y0": y0,
        },
    }


def validity_fn(
    yhat: str,
    hard_constraints: List[Constraint],
    recipe_db: RecipeDB,
    auto_patch_eat_before_cook: bool = False,
    raise_errors: bool = False,
) -> Tuple[bool, dict]:
    """
    Evaluate a single meal plan against its constraints and return detailed violation information.
    """
    meal_plan = parse_meal_plan(
        yhat,
        recipe_db,
        raise_errors=raise_errors,
        auto_patch_eat_before_cook=auto_patch_eat_before_cook,
    )
    if meal_plan is None:
        if raise_errors:
            raise Exception("Could not parse a meal plan from the message.")
        return False, {"parsed_plan": None}

    is_valid, score, min_unconstrained_score, max_unconstrained_score, metadata = (
        linear_reward(
            meal_plan,
            constraints=hard_constraints,
            weights=None,
            enforce_hard=True,
            raise_errors=raise_errors,
        )
    )
    return is_valid, metadata


def reward_fn(
    yhat: str,
    soft_constraints: List[Constraint],
    weights: np.ndarray,
    recipe_db: RecipeDB,
    auto_patch_eat_before_cook: bool = False,
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    """
    Evaluate a single meal plan's preference score.

    Args:
        yhat: The predicted meal plan
        soft_constraints: The soft constraints for preference scoring
        weights: The weights of the soft constraints
        recipe_db: The database of recipes
        raise_errors: Whether to raise errors on invalid input
    """
    # convert yhat to a meal plan
    meal_plan = parse_meal_plan(
        yhat,
        recipe_db,
        raise_errors=raise_errors,
        auto_patch_eat_before_cook=auto_patch_eat_before_cook,
    )
    if meal_plan is None:
        if raise_errors:
            raise Exception("Could not parse a meal plan from the message.")
        return float("-inf"), {"error": "Could not parse meal plan"}

    try:
        is_valid, score, min_unconstrained_score, max_unconstrained_score, metadata = (
            linear_reward(
                meal_plan,
                constraints=soft_constraints,
                weights=weights,
                enforce_hard=False,
                raise_errors=raise_errors,
            )
        )
    except Exception as e:
        if raise_errors:
            raise Exception(str(e))
        return float("-inf"), {"error": str(e)}

    # rescale from real numbers to [0, 1]
    score = (score - min_unconstrained_score) / (
        max_unconstrained_score - min_unconstrained_score
    )

    return (
        score * 100,  # rescale from [0, 1] to [0, 100]
        metadata,
    )


def parse_meal_plan_solutions(msg: str) -> List[str]:
    """Parse complete meal plan solutions from string (does not include individual recipe mentions)."""
    to_return = []
    # First try to parse from <meal_plan> tags
    meal_plan_content = parse_for_answer_tags(
        msg, keyword="meal_plan", return_none_if_not_found=True
    )
    if meal_plan_content:
        js = parse_json(meal_plan_content)
        if js is not None and len(js) > 0:
            to_return.append(json.dumps(js))
    else:
        # Fall back to parsing JSON directly (for backward compatibility)
        js = parse_json(msg)
        if js is not None and len(js) > 0:
            to_return.append(json.dumps(js))
    return to_return


def parse_meal_plan_solutions_and_options(msg: str) -> List[str]:
    """Parse both complete meal plan solutions and individual recipe mentions from string."""
    to_return = []
    # First try to parse from <meal_plan> tags
    meal_plan_content = parse_for_answer_tags(
        msg, keyword="meal_plan", return_none_if_not_found=True
    )
    if meal_plan_content:
        js = parse_json(meal_plan_content)
        if js is not None and len(js) > 0:
            to_return.append(json.dumps(js))

    # Parse recipe mentions
    mentioned_recipes = parse_for_answer_tags(
        msg, keyword="recipe", return_all=True, return_none_if_not_found=True
    )
    if mentioned_recipes is not None:
        mentioned_recipes = [
            recipe.strip()
            for mentions in mentioned_recipes
            for recipe in mentions.split(",")
            if recipe.strip()
        ]
        mentioned_recipes = list(dict.fromkeys(mentioned_recipes))
        # in order to compare recipes, create an empty meal plan except for one meal with the recipe
        for recipe in mentioned_recipes:
            to_return.append(
                json.dumps(
                    {
                        day: {
                            "breakfast": [
                                {"action": "cook", "recipe_title": recipe},
                                {"action": "eat", "recipe_title": recipe},
                            ]
                        }
                        for day in DAYS_OF_THE_WEEK
                    }
                )
            )
    return to_return


def output_to_streamlit(
    msg: str, db: RecipeDB, auto_patch_eat_before_cook: bool = False
) -> None:
    msg = msg.replace("$", "\$").replace("~", "\~")
    # Parse meal plan JSON
    js, start_end = parse_json(msg, return_start_end=True)

    # Parse recipe mentions
    mentioned_recipes = parse_for_answer_tags(
        msg, keyword="recipe", return_all=True, return_none_if_not_found=True
    )
    if mentioned_recipes is not None:
        mentioned_recipes = [
            recipe.strip()
            for mentions in mentioned_recipes
            for recipe in mentions.split(",")
            if recipe.strip()
        ]
        mentioned_recipes = list(dict.fromkeys(mentioned_recipes))

    if not js and not mentioned_recipes:
        st.write(msg)
        return

    # Generate unique ID for this message to avoid conflicts when multiple messages are rendered
    message_hash = str(hash(msg))[:8]
    unique_id = f"mentioned-recipes-{message_hash}"

    if js is None or start_end is None:
        # No meal plan, just render the message with recipe mentions
        st.markdown(
            replace_tags_with_link(msg, "recipe", f"#{unique_id}"),
            unsafe_allow_html=True,
        )
        if mentioned_recipes:
            with st.expander("Recipes mentioned in message", expanded=True):
                st.markdown(f'<div id="{unique_id}"></div>', unsafe_allow_html=True)
                renderer.render_recipe_mentions(mentioned_recipes, db)
        return

    if start_end[0] > 0:
        st.markdown(
            replace_tags_with_link(msg[: start_end[0]], "recipe", f"#{unique_id}"),
            unsafe_allow_html=True,
        )

    parsed = parse_meal_plan(
        msg[start_end[0] : start_end[1]],
        db,
        leave_invalid=True,
        auto_patch_eat_before_cook=auto_patch_eat_before_cook,
    )

    renderer.render_meal_plan_streamlit(parsed)

    if start_end[1] < len(msg):
        st.markdown(
            replace_tags_with_link(msg[start_end[1] :], "recipe", f"#{unique_id}"),
            unsafe_allow_html=True,
        )

    # Render recipe mentions if any
    if mentioned_recipes:
        st.markdown("---")
        with st.expander("Recipes mentioned in message", expanded=True):
            renderer.render_recipe_mentions(mentioned_recipes, db)


def output_to_txt(
    msg: str,
    db: RecipeDB,
    auto_patch_eat_before_cook: bool = False,
) -> str:
    """
    Returns the rendered message in a text format.
    All recipes in <recipe> tags are rendered as JSONs from the database.
    """
    mentioned_recipes = parse_for_answer_tags(
        msg, keyword="recipe", return_all=True, return_none_if_not_found=True
    )
    if mentioned_recipes is not None:
        mentioned_recipes = [
            recipe.strip()
            for mentions in mentioned_recipes
            for recipe in mentions.split(",")
            if recipe.strip()
        ]
        mentioned_recipes = list(dict.fromkeys(mentioned_recipes))
    else:
        return msg

    all_recipes_jsons = []
    for recipe_name in mentioned_recipes:
        try:
            recipe = db.get_recipe_by_name(recipe_name)
            if recipe is not None:
                # Convert Recipe dataclass to dict
                recipe_dict = asdict(recipe)
                recipe_dict = {
                    "title": recipe_dict["title"],
                    "ingredients": recipe_dict["ingredients"],
                    "cuisine": recipe_dict["cuisine"],
                    "total_time": recipe_dict["total_time"],
                    "num_servings": recipe_dict["num_servings"],
                    "rating": recipe_dict["rating"],
                    "num_reviews": recipe_dict["num_reviews"],
                    "calories": recipe_dict["calories"],
                    "protein": recipe_dict["protein"],
                    "total_fat": recipe_dict["total_fat"],
                    "total_carbohydrate": recipe_dict["total_carbohydrate"],
                    "diet": recipe_dict["diet"],
                    "intolerances": recipe_dict["intolerances"],
                    "equipment": recipe_dict["equipment"],
                    "food_type": recipe_dict["food_type"],
                }
                all_recipes_jsons.append(recipe_dict)
            else:
                all_recipes_jsons.append(
                    {"title": recipe_name, "name": "Invalid recipe (not in database)"}
                )
        except Exception as e:
            all_recipes_jsons.append(
                {"title": recipe_name, "error": f"Error retrieving recipe: {str(e)}"}
            )

    out = msg
    if all_recipes_jsons:
        out += "\n\n------- Information about mentioned recipes ----------\n\n"
        out += str(all_recipes_jsons)
    return out


def output_to_streamlit_comparison(
    y1: str,
    y2: str,
    db: RecipeDB,
    validity_fn=None,
    validity_kwargs=None,
    auto_patch_eat_before_cook: bool = False,
) -> None:
    parsed1 = parse_meal_plan(
        y1,
        db,
        leave_invalid=True,
        auto_patch_eat_before_cook=auto_patch_eat_before_cook,
    )
    parsed2 = parse_meal_plan(
        y2,
        db,
        leave_invalid=True,
        auto_patch_eat_before_cook=auto_patch_eat_before_cook,
    )

    a_valid = a_metadata = b_valid = b_metadata = None
    if validity_fn is not None and validity_kwargs is not None:
        a_valid, a_metadata = validity_fn(
            y1, **(validity_kwargs or {}), raise_errors=False
        )
        b_valid, b_metadata = validity_fn(
            y2, **(validity_kwargs or {}), raise_errors=False
        )

    renderer.output_to_streamlit_comparison(
        parsed1,
        parsed2,
        db,
        a_valid,
        b_valid,
        a_metadata,
        b_metadata,
    )
