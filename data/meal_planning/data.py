# from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import dill
import os
import streamlit as st


# import sys
import json

from data.reward import Constraint, linear_reward
from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
)
from utils.misc import (
    parse_json,
    subset_data,
    add_section,
    parse_for_answer_tags,
    replace_tags_with_link,
)
from utils.streamlit_types import FormElement, form_element_to_streamlit
from data.meal_planning.db import (
    RecipeDB,
    DAYS_OF_THE_WEEK,
    MEALS,
    Recipe,
    DIETS,
    INTOLERANCES,
)
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

from collections import defaultdict

"""
Ideas for future work:
- Harder version: cooking for more than 1 person
"""

DEV_FRAC = 0.1
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
FMT_INSTRUCTIONS = (
    """Return the meal plan as a JSON string with the following structure:
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
""".strip()
    + f"\nThe outer keys are the days of the week (choose from {DAYS_OF_THE_WEEK}), and the inner keys are the meals of the day (choose from {MEALS}; note that 'snack' occurs before 'dinner')."
    + """Each meal slot contains an (ordered) list of actions:
    1. One kind of action is to cook a recipe: the dict should set `action` to "cook" and `recipe_title` to the title of a recipe from the database. The database specifies how many servings each cook action makes. Assume the client will refrigerate all servings, and that leftovers do not expire. Note that if you cook, you do not automatically eat: you must add a separate "eat" action to the meal plan.
    2. Another kind of action is to eat a recipe: the dict should set `action` to "eat" and `recipe_title` to the title of a recipe from the database. Recipes can only be eaten if they have been cooked. Make sure not to eat more servings than the recipe makes.
Note that you do not always have to cook and eat each recipe at the same time: you may cook the recipe at one meal, and then eat it at another meal. 

You can cook multiple recipes at each mealtime (e.g. a drink, a main course, and a dessert); the total cooking time will be the max of the total time for each recipe.

To render a description of a single recipe (instead of a full meal plan) to the user, you can mention its recipe_title and wrap it in <recipe></recipe>, e.g.: '<recipe>Chicken Parmesan</recipe>'. Do not put <recipe></recipe> tags in the JSON of a full meal plan.
""".strip()
)

COMMONSENSE_DESCRIPTION = """In our task, a valid meal plan:
* ONLY uses recipes from AllRecipes.com. Using other recipes is not allowed.
* Tells us when to cook. Each recipe makes a certain number of servings; we can then parcel out those servings to different meals.
* Tells us when and how many servings to eat. You cannot eat more servings of a recipe than you have cooked. If a recipe makes only 1 serving, you cannot eat 2 servings of it.
* Does not violate any personal constraints.

AllRecipes.com recipes have ingredients, instructions, nutrition facts, and take time to cook.

Clarifications:
* You CAN cook the same recipe multiple times to make more servings of it.
* You CAN eat leftovers from a previously cooked meal.
"""


CUSTOM_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write you a perfect meal prep plan that you can actually follow this coming week.** A meal plan is a week-long calendar that specifies what to eat for every meal of the day. The plan also specifies when to cook each recipe.

The plan must work with your schedule, dietary restrictions, and preferences.
"""


def render_fixed_task_explanation():
    """Render the fixed task explanation for meal planning."""
    st.markdown(COMMONSENSE_DESCRIPTION)


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
        return "Work with the assistant to **write a personal meal plan for the next week** using real recipes from AllRecipes.com."

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
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        self._docker_image = docker_image
        self._persist_docker_container = persist_docker_container
        self._auto_patch_eat_before_cook = auto_patch_eat_before_cook

        # Load all the fixed spec profiles
        profiles = dill.load(open(f"{DATASET_ROOT}/assets/profiles.pkl", "rb"))
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

        # Load some ystars
        try:
            self._ystars = json.load(open(f"{DATASET_ROOT}/assets/manual_ystars.json"))
        except Exception:
            self._ystars = {}

        self._y0_mapping = json.load(open(f"{DATASET_ROOT}/assets/y0_mapping.json"))

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

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}

        # Load requested specs
        specs = {}
        for ix in indexes:
            profile = self._profiles[ix]

            dem = profile["demographic_information"]
            signature = f"The task is to write out a one-week meal plan. The week starts on Sunday and ends on Saturday. The client only eats {' and '.join(dem['meals_considered'])} each day. The meal plan should designate what to eat and when to cook. Format the plan as a JSON.\n\nTo help you, the environment provides a Jupyter notebook and a CSV of recipes. All foods in the meal plan must correspond to a recipe from the given CSV."
            signature += "\n\n" + add_section(
                "Constraints (must be satisfied)", profile["hard_description"]
            )
            signature += "\n\n" + add_section(
                "Client information",
                f"Sex: {dem['sex']}\n"
                + f"Weight: {dem['weight']} kg\n"
                + f"Height: {dem['height']} cm\n"
                + f"Age: {dem['age']}\n"
                + f"Activity level: {dem['activity_level']}\n"
                + f"Goal: {dem['goal']} weight",
            )

            # constraints & theta
            constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in profile["constraints"]
            ]
            hard_constraints = [c for c in constraints if c.is_hard]
            soft_constraints = [c for c in constraints if not c.is_hard]

            theta = add_section(
                "Preferences (most important to least important)",
                profile["soft_description"].replace("\n", "\n<chunk>\n"),
            )

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

            # specification
            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": hard_constraints,
                    "recipe_db": self._recipe_db,
                    "auto_patch_eat_before_cook": self._auto_patch_eat_before_cook,
                },
                # validity_fn_tool_name=None,  # Not provided
                # validity_fn_tool_description=None,  # Not provided
                reward_fn=reward_fn,
                reward_kwargs={
                    "soft_constraints": soft_constraints,
                    "weights": profile["weights"],
                    "recipe_db": self._recipe_db,
                    "auto_patch_eat_before_cook": self._auto_patch_eat_before_cook,
                },
                # reward_fn_tool_name=None,  # Not provided
                # reward_fn_tool_description=None,  # Not provided
                ystar=self._ystars.get(str(ix), None),
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=actions,
                fmt_instructions=FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
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
                    description="The total number of servings consumed of a recipe across the week must be <= the total number of servings cooked of the recipe",
                    extractor="check_servings_consumed_lt_cooked_total",
                    is_hard=True,
                ),
            ]
            constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in constraints
            ]

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification="Generate a meal plan for yourself for the next week. Only plan for 1 person (yourself).",
                user_specification_form_initial=[],
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
                fmt_instructions=FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
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
                render_evaluation_fn=lambda **kwargs: renderer.render_eval_meal(
                    **kwargs,
                    db=self._recipe_db,
                    auto_patch_eat_before_cook=self._auto_patch_eat_before_cook,
                ),
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
                label="Which meal plan are you more likely to follow next week: A or B?",
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
                input_type="multiselect",
                label="Allergies/Intolerances",
                options=INTOLERANCES,
                default=[],
                required=False,
                help="Select any foods you are allergic to or intolerant of",
            ),
            FormElement(
                input_type="multiselect",
                label="Dietary Preferences",
                options=DIETS,
                default=[],
                required=False,
                help="Select any dietary preferences or restrictions",
            ),
        ]

    def _render_custom_task_explanation(self):
        """Render the custom task explanation for meal planning."""

        st.markdown("### What you need to prompt the assistant to do")
        st.markdown(
            "In this task, **your goal is to get the assistant to write you a perfect meal prep plan that you can actually follow this coming week.** A meal plan is a week-long calendar that specifies what to eat for every meal of the day. The plan also specifies when to cook each recipe."
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
        #                     "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
        #                 },
        #                 {
        #                     "action": "eat",
        #                     "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
        #                 },
        #                 {
        #                     "action": "eat",
        #                     "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
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
                            "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
                        },
                        {
                            "action": "eat",
                            "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
                        },
                        {
                            "action": "eat",
                            "recipe_title": "Roasted Pork Bánh Mì (Vietnamese Sandwich)",
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
    # Allergy/intolerance constraints - using the pattern from generate_profiles.py
    allergies = form_results.get("Allergies/Intolerances", [])
    if allergies:
        for intolerance in allergies:
            if intolerance is None:
                continue
            constraints.append(
                Constraint.create_boolean_penalize_false_constraint(
                    description=f"Must avoid {intolerance.lower()} for all meals",
                    extractor="recipes_avoid_intolerance",
                    extractor_kwargs={"intolerance": intolerance},
                    is_hard=True,
                )
            )

    # Dietary preference constraints - using patterns from generate_profiles.py
    dietary_prefs = form_results.get("Dietary Preferences", [])
    for diet in dietary_prefs:
        if diet is None:
            continue
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"Must follow {diet} diet for all meals",
                extractor="recipes_follow_diet",
                extractor_kwargs={"diet": diet},
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
    if "Allergies/Intolerances" in form_results:
        new_specification += f" | Allergies/Intolerances: {' and '.join(form_results['Allergies/Intolerances'])}"
    if "Dietary Preferences" in form_results:
        new_specification += f" | Dietary Restrictions: {' and '.join(form_results['Dietary Preferences'])}"

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

    diet_key = (
        DIETS_TO_Y0[dietary_prefs[0]]
        if dietary_prefs and dietary_prefs[0]
        else "normal"
    )
    y0 = y0_mapping.get(diet_key)

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


def parse_meal_plan(
    yhat: str,
    recipe_db: RecipeDB,
    raise_errors: bool = False,
    leave_invalid: bool = False,
    auto_patch_eat_before_cook: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Parse a meal plan from a JSON string.
    Assumes a dictionary with the following structure:
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
    and returns a dictionary with the following structure:
    {
        "sunday": {
            "breakfast": [
                {"recipe": Recipe, "cook": True, "servings_consumed": 2},
                ...
            ],
            ...
        }
    }
    If auto_patch_eat_before_cook is True, then the meal plan will be patched to ensure that each recipe is cooked before it is eaten. The cook action will be inserted at the same time as the eat action.
    """
    meal_plan = parse_json(yhat)
    if meal_plan is None:
        return None

    # do some automatic correction for case, missing fields, etc.
    try:
        # lower case all the keys
        meal_plan = {
            k.lower(): (
                {ki.lower(): vi for ki, vi in v.items()} if isinstance(v, dict) else {}
            )
            for k, v in meal_plan.items()
        }
    except Exception:
        print(f"Error parsing meal plan: {meal_plan}")
        return None

    def find_recipe_dict_by_title(lst, title):
        return next((d for d in lst if d["recipe"].title == title), None)

    # build the meal plan with actual Recipe objects
    MISSING_MEAL = None
    corrected_meal_plan = defaultdict(lambda: defaultdict(list))
    servings_available = defaultdict(int)
    for day in DAYS_OF_THE_WEEK:
        for meal_type in MEALS:
            # if this (day, meal_type) pair is missing, set it to MISSING_MEAL
            if (
                day not in meal_plan
                or meal_type not in meal_plan[day]
                or meal_plan[day][meal_type] is None
                or len(meal_plan[day][meal_type]) == 0
            ):
                corrected_meal_plan[day][meal_type] = MISSING_MEAL
                continue

            # read the list of actions from the original meal plan
            actions = meal_plan[day][meal_type]
            # sort the actions to put cook actions first
            actions = sorted(
                actions,
                key=lambda x: x["action"].strip().lower() == "cook",
                reverse=True,
            )
            for d in actions:
                # remove malformed empty actions
                if len(d) == 0 or d["recipe_title"] == "":
                    continue

                # add the recipe to the meal plan
                recipe = recipe_db.get_recipe_by_name(d["recipe_title"])
                if recipe is None:
                    if raise_errors:
                        raise ValueError(
                            f"Recipe was not found on AllRecipes: {d['recipe_title']}"
                        )
                    if not leave_invalid:
                        continue
                    elif leave_invalid:
                        recipe = Recipe(
                            title=d["recipe_title"],
                            ingredients=None,
                            instructions=None,
                            cuisine=None,
                        )

                # if cook, add to the meal plan and update the servings_available
                if d["action"].strip().lower() == "cook":
                    corrected_meal_plan[day][meal_type].append(
                        {
                            "recipe": recipe,
                            "cook": True,
                            "servings_consumed": 0,
                        }
                    )
                    servings_available[recipe.title] += recipe.num_servings

                # if eat and there are servings available, add to the meal plan
                elif (d["action"].strip().lower() == "eat") and (
                    servings_available[recipe.title] >= 1
                ):
                    _dict = find_recipe_dict_by_title(
                        corrected_meal_plan[day][meal_type], recipe.title
                    )
                    if _dict is None:
                        # we haven't encountered this recipe in this meal slot yet
                        corrected_meal_plan[day][meal_type].append(
                            {
                                "recipe": recipe,
                                "cook": False,
                                "servings_consumed": 1,
                            }
                        )
                    else:
                        _dict["servings_consumed"] += 1
                    servings_available[recipe.title] -= 1

                # if eat and there are no servings available, insert a cook action
                elif (d["action"].strip().lower() == "eat") and (
                    servings_available[recipe.title] < 1
                ):
                    corrected_meal_plan[day][meal_type].append(
                        {
                            "recipe": recipe,
                            "cook": True,
                            "servings_consumed": 1,
                        }
                    )
                    servings_available[recipe.title] += recipe.num_servings - 1

            # if after cycling through all the actions, the meal plan is still empty, set it to MISSING_MEAL
            if (
                corrected_meal_plan[day][meal_type] is not MISSING_MEAL
                and len(corrected_meal_plan[day][meal_type]) == 0
            ):
                corrected_meal_plan[day][meal_type] = MISSING_MEAL

    return corrected_meal_plan


def output_to_streamlit(
    msg: str, db: RecipeDB, auto_patch_eat_before_cook: bool = False
) -> None:
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
        mentioned_recipes = list(set(mentioned_recipes))

    if js is None:
        # No meal plan, just render the message with recipe mentions
        st.markdown(
            replace_tags_with_link(msg, "recipe", "#mentioned-recipes"),
            unsafe_allow_html=True,
        )
        if mentioned_recipes:
            with st.expander("Recipes mentioned in message", expanded=False):
                renderer.render_recipe_mentions(mentioned_recipes, db)
        return

    if start_end[0] > 0:
        st.markdown(
            replace_tags_with_link(msg[: start_end[0]], "recipe", "#mentioned-recipes"),
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
            replace_tags_with_link(msg[start_end[1] :], "recipe", "#mentioned-recipes"),
            unsafe_allow_html=True,
        )

    # Render recipe mentions if any
    if mentioned_recipes:
        st.markdown("---")
        with st.expander("Recipes mentioned in message", expanded=False):
            renderer.render_recipe_mentions(mentioned_recipes, db)


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
