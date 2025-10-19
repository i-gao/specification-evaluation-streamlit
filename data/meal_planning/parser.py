from collections import defaultdict
from typing import Dict, Any

from utils.misc import parse_json
from data.meal_planning.db import (
    RecipeDB,
    DAYS_OF_THE_WEEK,
    MEALS,
    Recipe,
)


def parse_meal_plan(
    yhat: str,
    recipe_db: RecipeDB,
    raise_errors: bool = False,
    leave_invalid: bool = False,
    auto_patch_eat_before_cook: bool = True,
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
                invalid = False
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
                        invalid = True

                # if cook, add to the meal plan and update the servings_available
                if d["action"].strip().lower() == "cook":
                    corrected_meal_plan[day][meal_type].append(
                        {
                            "recipe": recipe,
                            "cook": True,
                            "servings_consumed": 0,
                            "invalid": invalid,
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
                                "invalid": invalid,
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
                            "invalid": invalid,
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
