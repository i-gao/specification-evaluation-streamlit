# Extractor functions for meal planning constraints
# Each function takes a meal_plan as input and returns a tuple (value, detailed_message)
# where value is the value needed for the constraint and detailed_message describes the result

from data.meal_planning.db import DAYS_OF_THE_WEEK, MEALS_OF_THE_DAY

"""
Assume a meal plan is a dictionary with the following structure:
{
    "sunday": {
        "breakfast": [
            {'recipe': dict, 'cook': bool, 'servings_consumed': int}
            ...
        ],
        ...
    },
    ...
}
"""


def check_recipes_eaten_after_cooked(meal_plan):
    """
    Check that all recipes are eaten after they are cooked.
    Returns (bool, str): (success, detailed_message)
    """
    try:
        meals_cooked = set()  # recipe titles
        violations = []
        for day in DAYS_OF_THE_WEEK:
            for meal_name in MEALS_OF_THE_DAY:
                if meal_plan[day][meal_name] is None:
                    continue
                for recipe_dict in meal_plan[day][meal_name]:
                    recipe_title = recipe_dict["recipe"].title
                    if recipe_dict["cook"]:
                        meals_cooked.add(recipe_title)
                    if recipe_title not in meals_cooked:
                        violations.append(
                            f"Recipe '{recipe_title}' was consumed at {day.capitalize()} {meal_name} but was not cooked beforehand"
                        )

        if violations:
            return False, f"Leftover constraint violations: {'; '.join(violations)}"
        return True, "All recipes were properly cooked before being consumed"
    except (KeyError, AttributeError, TypeError) as e:
        return False, f"Error checking leftover constraints: {str(e)}"


def _get_matching_dicts(list_of_dicts, recipe_title):
    return [d for d in list_of_dicts if d["recipe"].title == recipe_title]


def check_servings_consumed_lt_cooked_total(meal_plan):
    """
    Check that the total number of servings consumed per recipe is
    <= the number of servings cooked of the recipe.
    Returns (bool, str): (success, detailed_message)
    """
    try:
        servings_remaining = {}  # recipe_title -> remaining_servings
        violations = []
        for day in DAYS_OF_THE_WEEK:
            for meal_name in MEALS_OF_THE_DAY:
                if meal_plan[day][meal_name] is None:
                    continue

                seen_recipes = set()
                for recipe_dict in meal_plan[day][meal_name]:
                    recipe_title = recipe_dict["recipe"].title
                    if recipe_title in seen_recipes:
                        continue
                    seen_recipes.add(recipe_title)

                    # Look for future dicts
                    servings_consumed = 0
                    for d in _get_matching_dicts(
                        meal_plan[day][meal_name], recipe_title
                    ):
                        servings_consumed += d["servings_consumed"]
                        if d["cook"]:
                            original_servings = d["recipe"].num_servings
                            if recipe_title in servings_remaining:
                                servings_remaining[recipe_title] += original_servings
                            else:
                                servings_remaining[recipe_title] = original_servings

                    if servings_remaining.get(recipe_title, 0) < servings_consumed:
                        violations.append(
                            f"In {day.capitalize()} {meal_name}, {servings_consumed} servings of recipe '{recipe_title}' are consumed but only {servings_remaining.get(recipe_title, 0)} servings are available"
                        )
                    servings_remaining[recipe_title] -= servings_consumed

        if violations:
            return False, f"Serving consumption violations: {'; '.join(violations)}"
        return True, "All recipes were consumed within their cooked serving limits"
    except (KeyError, AttributeError, TypeError) as e:
        return False, f"Error checking serving consumption: {str(e)}"


def something_is_eaten(meal_plan, day, meal):
    """Return (bool, str) if meal_plan[day][meal] has some servings consumed."""
    if meal_plan[day][meal] is None:
        return False, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] > 0:
            return (
                True,
                f"Meal consumed at {day.capitalize()} {meal}: {recipe_dict['recipe'].title}",
            )
    return False, f"No servings consumed at {day.capitalize()} {meal}"


def all_non_blocked_meals_filled(meal_plan, meals_considered, blocked_meals):
    """
    Return (List[bool], str) a list of booleans, one per non-blocked considered meal,
    indicating if that meal has something eaten.
    True means the meal is filled (has something eaten).
    """
    meal_booleans = []
    meal_descriptions = []
    for day in DAYS_OF_THE_WEEK:
        for meal in meals_considered:
            meal_key = f"{day}:{meal}"
            # Skip blocked meals
            if meal_key in blocked_meals:
                continue
            # Check if meal is filled
            is_filled = False
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["servings_consumed"] > 0:
                        is_filled = True
                        break
            meal_booleans.append(is_filled)
            meal_descriptions.append(f"{meal_key}: {is_filled}")

    filled_count = sum(meal_booleans)
    total_meals = len(meal_booleans)
    return (
        meal_booleans,
        f"Non-blocked meals filled: {filled_count}/{total_meals} ({', '.join(meal_descriptions)})",
    )


def nothing_is_eaten(meal_plan, day, meal):
    """Return (bool, str) if meal_plan[day][meal] has no servings consumed."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal} (as expected)"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] > 0:
            return (
                False,
                f"Unexpected meal consumed at {day.capitalize()} {meal}: {recipe_dict['recipe'].title}",
            )
    return True, f"No servings consumed at {day.capitalize()} {meal} (as expected)"


def meal_should_be_empty(meal_plan, meal_to_skip):
    """
    Return (List[bool], str) a list of booleans, one per day, indicating if the meal has something eaten.
    True means violation (meal should be empty but has something).
    """
    violations = []
    for day in DAYS_OF_THE_WEEK:
        has_something = False
        if meal_plan[day][meal_to_skip] is not None:
            for recipe_dict in meal_plan[day][meal_to_skip]:
                if recipe_dict["servings_consumed"] > 0:
                    has_something = True
                    break
        violations.append(has_something)

    violation_count = sum(violations)
    return (
        violations,
        f"Meal '{meal_to_skip}' should be empty but has something eaten on {violation_count}/{len(violations)} days",
    )


def cuisine_appears(meal_plan, cuisine):
    """
    Return (List[bool], str) a list of booleans, one per meal, indicating if the cuisine appears in that meal.
    Best case is all True (cuisine appears in all meals).
    """
    meal_booleans = []
    meal_descriptions = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            meal_key = f"{day}:{meal}"
            appears = False
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["servings_consumed"] > 0:
                        if recipe_dict["recipe"].cuisine == cuisine:
                            appears = True
                            break
            meal_booleans.append(appears)
            meal_descriptions.append(f"{meal_key}: {appears}")

    true_count = sum(meal_booleans)
    total_meals = len(meal_booleans)
    return (
        meal_booleans,
        f"Cuisine '{cuisine}' appears in {true_count}/{total_meals} meals: {', '.join(meal_descriptions)}",
    )


def food_type_appears(meal_plan, food_type):
    """
    Return (List[bool], str) a list of booleans, one per meal, indicating if the food type appears in that meal.
    Best case is all True (food type appears in all meals).
    """
    meal_booleans = []
    meal_descriptions = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            meal_key = f"{day}:{meal}"
            appears = False
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["servings_consumed"] > 0:
                        if recipe_dict["recipe"].food_type == food_type:
                            appears = True
                            break
            meal_booleans.append(appears)
            meal_descriptions.append(f"{meal_key}: {appears}")

    true_count = sum(meal_booleans)
    total_meals = len(meal_booleans)
    return (
        meal_booleans,
        f"Food type '{food_type}' appears in {true_count}/{total_meals} meals: {', '.join(meal_descriptions)}",
    )


def food_keyword_appears(meal_plan, keyword):
    """
    Return (List[bool], str) a list of booleans, one per meal, indicating if the food keyword appears in that meal.
    Best case is all True (keyword appears in all meals).
    """
    meal_booleans = []
    meal_descriptions = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            meal_key = f"{day}:{meal}"
            appears = False
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["servings_consumed"] > 0:
                        if keyword.lower() in recipe_dict["recipe"].title.lower():
                            appears = True
                            break
            meal_booleans.append(appears)
            meal_descriptions.append(f"{meal_key}: {appears}")

    true_count = sum(meal_booleans)
    total_meals = len(meal_booleans)
    return (
        meal_booleans,
        f"Food keyword '{keyword}' appears in {true_count}/{total_meals} meals: {', '.join(meal_descriptions)}",
    )


def min_star_rating(meal_plan):
    """
    Return (float, str) the minimum star rating across all meals in the meal plan period.
    Returns None if no ratings are present.
    """
    all_ratings = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    recipe_rating = recipe_dict["recipe"].rating
                    if recipe_rating is not None:
                        all_ratings.append(recipe_rating)

    if not all_ratings:
        return (
            None,
            f"No star ratings found across the {len(DAYS_OF_THE_WEEK)}-day period",
        )

    min_rating = min(all_ratings)
    return (
        min_rating,
        f"Minimum star rating across the {len(DAYS_OF_THE_WEEK)}-day period: {min_rating:.1f}",
    )


###### Concerns cooked recipes ######


def total_cooks(meal_plan):
    """Return (int, str) the total number of times cooking was required in the meal plan period."""
    cook_count = 0
    cooked_recipes = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["cook"]:
                        cook_count += 1
                        cooked_recipes.append(
                            f"{day.capitalize()} {meal}: {recipe_dict['recipe'].title}"
                        )

    message = f"Total cooking sessions: {cook_count}"
    if cooked_recipes:
        message += f" ({'; '.join(cooked_recipes)})"
    return cook_count, message


def cooking_time_under(meal_plan, day, meal, time_limit):
    """Return (bool, str) if the max of total_time for all cooked recipes at (day, meal) <= time_limit."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    total_time = 0
    cooked_recipes = []
    for recipe_dict in meal_plan[day][meal]:
        if not recipe_dict["cook"]:  # only check cooked recipes
            continue
        recipe_time = recipe_dict["recipe"].total_time
        total_time += recipe_time
        cooked_recipes.append(f"{recipe_dict['recipe'].title} ({recipe_time} min)")

    if total_time > time_limit:
        return (
            False,
            f"Cooking time exceeded at {day.capitalize()} {meal}: {total_time} minutes total ({'; '.join(cooked_recipes)}) exceeds limit of {time_limit} minutes",
        )
    return (
        True,
        f"Cooking time at {day.capitalize()} {meal}: {total_time} minutes ({'; '.join(cooked_recipes)}) within limit of {time_limit} minutes",
    )


def all_equipment(meal_plan):
    """
    Return (List[str], str) all equipment from cooked recipes across the meal plan period (multiset).
    """
    equipment_list = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if not recipe_dict["cook"]:  # only check cooked recipes
                    continue
                if recipe_dict["servings_consumed"] > 0:
                    # Add each equipment once per serving consumed (multiset)
                    for _ in range(recipe_dict["servings_consumed"]):
                        equipment_list.extend(recipe_dict["recipe"].equipment)
    return (
        equipment_list,
        f"All equipment across the {len(DAYS_OF_THE_WEEK)}-day period: {len(set(equipment_list))} unique equipment",
    )


###### Concerns consumed recipes ######


def recipes_follow_diet(meal_plan, day, meal, diet):
    """Return (bool, str) if diet is NOT in the recipe diet for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if diet not in recipe_dict["recipe"].diet:
            return (
                False,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} violates diet restriction: {diet}",
            )
    return (
        True,
        f"All consumed recipes at {day.capitalize()} {meal} follow diet restriction: {diet}",
    )


def recipes_avoid_intolerance(meal_plan, day, meal, intolerance):
    """Return (bool, str) if intolerance is NOT in the recipe intolerances for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if intolerance in recipe_dict["recipe"].intolerances:
            return (
                False,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} contains intolerance trigger: {intolerance}",
            )
    return (
        True,
        f"All consumed recipes at {day.capitalize()} {meal} avoid intolerance: {intolerance}",
    )


def meals_violating_diet(meal_plan, diet):
    """
    Return (List[str], str) all (day, meal) pairs that violate the diet requirement.
    Each violation is represented as "day:meal" string.
    """
    violations = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] == 0:
                    continue
                if diet not in recipe_dict["recipe"].diet:
                    violations.append(f"{day}:{meal}")
                    break  # Only count each meal once
    return (
        violations,
        f"Meals that violate {diet} diet: {', '.join(violations) if violations else 'none'}",
    )


def all_intolerances(meal_plan):
    """
    Return (List[str], str) all intolerances from consumed recipes across the meal plan period (multiset).
    """
    intolerances = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    # Add each intolerance once per serving consumed (multiset)
                    for _ in range(recipe_dict["servings_consumed"]):
                        intolerances.extend(recipe_dict["recipe"].intolerances)
    return (
        intolerances,
        f"All intolerances across the {len(DAYS_OF_THE_WEEK)}-day period: {len(set(intolerances))} unique intolerances",
    )


def num_repeated_recipes(meal_plan):
    """Return (int, str) the number of repeatedly consumed recipes in the meal plan period."""
    seen_recipes = set()
    num_repeats = 0
    repeated_recipes = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] == 0:
                    continue
                title = recipe_dict["recipe"].title
                if title in seen_recipes:
                    num_repeats += 1
                    repeated_recipes.append(f"{day.capitalize()} {meal}: {title}")
                else:
                    seen_recipes.add(title)

    message = f"Number of recipe repeats: {num_repeats}"
    if repeated_recipes:
        message += f" ({'; '.join(repeated_recipes)})"
    return num_repeats, message


def taco_tuesday(meal_plan):
    """Return (bool, str) if any consumed Tuesday recipe is Latin American."""
    for meal in MEALS_OF_THE_DAY:
        if meal_plan["tuesday"][meal] is None:
            continue
        for recipe_dict in meal_plan["tuesday"][meal]:
            if recipe_dict["servings_consumed"] == 0:
                continue
            if recipe_dict["recipe"].cuisine in [
                "Tex-Mex",
                "Colombian",
                "Puerto Rican",
                "Chilean",
                "Brazilian",
                "Cuban",
                "Argentinian",
                "Peruvian",
            ]:
                return (
                    True,
                    f"Taco Tuesday satisfied: '{recipe_dict['recipe'].title}' consumed at Tuesday {meal}",
                )
    return False, "Taco Tuesday not satisfied: no Mexican cuisine consumed on Tuesday"


def meatless_monday(meal_plan):
    """Return (bool, str) if ALL consumed Monday recipes are vegetarian."""
    for meal in MEALS_OF_THE_DAY:
        if meal_plan["monday"][meal] is None:
            continue
        for recipe_dict in meal_plan["monday"][meal]:
            if recipe_dict["servings_consumed"] == 0:
                continue
            if "Vegetarian" not in recipe_dict["recipe"].diet:
                return (
                    False,
                    f"Meatless Monday violated: '{recipe_dict['recipe'].title}' consumed at Monday {meal} is not vegetarian",
                )
    return (
        True,
        "Meatless Monday satisfied: all consumed recipes on Monday are vegetarian",
    )


def pizza_friday(meal_plan):
    """Return (bool, str) if any consumed Friday recipe contains pizza."""
    for meal in MEALS_OF_THE_DAY:
        if meal_plan["friday"][meal] is None:
            continue
        for recipe_dict in meal_plan["friday"][meal]:
            if recipe_dict["servings_consumed"] == 0:
                continue
            if "pizza" in recipe_dict["recipe"].title.lower():
                return (
                    True,
                    f"Pizza Friday satisfied: '{recipe_dict['recipe'].title}' consumed at Friday {meal}",
                )
    return False, "Pizza Friday not satisfied: no pizza consumed on Friday"


def daily_protein(meal_plan, day):
    """Return (float, str) the sum of protein for all consumed recipes at (day, meal)."""
    total_protein = 0
    protein_sources = []
    for meal in MEALS_OF_THE_DAY:
        if meal_plan[day][meal] is not None:
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    protein = (
                        recipe_dict["recipe"].protein * recipe_dict["servings_consumed"]
                    )
                    total_protein += protein
                    protein_sources.append(
                        f"{recipe_dict['recipe'].title} ({protein:.1f}g)"
                    )

    message = f"Total protein on {day.capitalize()}: {total_protein:.1f}g"
    if protein_sources:
        message += f" from {', '.join(protein_sources)}"
    return total_protein, message


def daily_total_fat(meal_plan, day):
    """Return (float, str) the sum of total_fat for all consumed recipes at (day, meal)."""
    total_fat = 0
    fat_sources = []
    for meal in MEALS_OF_THE_DAY:
        if meal_plan[day][meal] is not None:
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    fat = (
                        recipe_dict["recipe"].total_fat
                        * recipe_dict["servings_consumed"]
                    )
                    total_fat += fat
                    fat_sources.append(f"{recipe_dict['recipe'].title} ({fat:.1f}g)")

    message = f"Total fat on {day.capitalize()}: {total_fat:.1f}g"
    if fat_sources:
        message += f" from {', '.join(fat_sources)}"
    return total_fat, message


def daily_carbohydrate(meal_plan, day):
    """Return (float, str) the sum of total_carbohydrate for all consumed recipes at (day, meal)."""
    total_carbs = 0
    carb_sources = []
    for meal in MEALS_OF_THE_DAY:
        if meal_plan[day][meal] is not None:
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    carbs = (
                        recipe_dict["recipe"].total_carbohydrate
                        * recipe_dict["servings_consumed"]
                    )
                    total_carbs += carbs
                    carb_sources.append(f"{recipe_dict['recipe'].title} ({carbs:.1f}g)")

    message = f"Total carbohydrates on {day.capitalize()}: {total_carbs:.1f}g"
    if carb_sources:
        message += f" from {', '.join(carb_sources)}"
    return total_carbs, message


def daily_fiber(meal_plan, day):
    """Return (float, str) the sum of dietary_fiber for all consumed recipes at (day, meal)."""
    total_fiber = 0
    fiber_sources = []
    for meal in MEALS_OF_THE_DAY:
        if meal_plan[day][meal] is not None:
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    fiber = (
                        recipe_dict["recipe"].dietary_fiber
                        * recipe_dict["servings_consumed"]
                    )
                    total_fiber += fiber
                    fiber_sources.append(
                        f"{recipe_dict['recipe'].title} ({fiber:.1f}g)"
                    )

    message = f"Total fiber on {day.capitalize()}: {total_fiber:.1f}g"
    if fiber_sources:
        message += f" from {', '.join(fiber_sources)}"
    return total_fiber, message


def daily_calories(meal_plan, day):
    """Return (float, str) the sum of calories for all recipes consumed on a given day."""
    total_calories = 0
    calorie_sources = []
    for meal in MEALS_OF_THE_DAY:
        if meal_plan[day][meal] is not None:
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    calories = (
                        recipe_dict["recipe"].calories
                        * recipe_dict["servings_consumed"]
                    )
                    total_calories += calories
                    calorie_sources.append(
                        f"{recipe_dict['recipe'].title} ({calories:.0f} cal)"
                    )

    message = f"Total calories on {day.capitalize()}: {total_calories:.0f} cal"
    if calorie_sources:
        message += f" from {', '.join(calorie_sources)}"
    return total_calories, message


def all_ingredients(meal_plan):
    """Return (List[str], str) all ingredients from consumed recipes across the meal plan period."""
    ingredients = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            if meal_plan[day][meal] is None:
                continue
            for recipe_dict in meal_plan[day][meal]:
                if recipe_dict["servings_consumed"] > 0:
                    # Add each ingredient once per serving consumed (multiset)
                    for _ in range(recipe_dict["servings_consumed"]):
                        ingredients.extend(recipe_dict["recipe"].ingredients)
    return (
        ingredients,
        f"All ingredients across the {len(DAYS_OF_THE_WEEK)}-day period: {len(set(ingredients))} unique ingredients",
    )


def ingredient_appears(meal_plan, ingredient):
    """
    Return (List[bool], str) a list of booleans, one per meal, indicating if the ingredient appears in that meal.
    Best case is all True (ingredient appears in all meals).
    """
    meal_booleans = []
    meal_descriptions = []
    for day in DAYS_OF_THE_WEEK:
        for meal in MEALS_OF_THE_DAY:
            meal_key = f"{day}:{meal}"
            appears = False
            if meal_plan[day][meal] is not None:
                for recipe_dict in meal_plan[day][meal]:
                    if recipe_dict["servings_consumed"] > 0:
                        if any(
                            [
                                ingredient.lower() in i.lower()
                                for i in recipe_dict["recipe"].ingredients
                            ]
                        ):
                            appears = True
                            break
            meal_booleans.append(appears)
            meal_descriptions.append(f"{meal_key}: {appears}")

    true_count = sum(meal_booleans)
    total_meals = len(meal_booleans)
    return (
        meal_booleans,
        f"Ingredient '{ingredient}' appears in {true_count}/{total_meals} meals: {', '.join(meal_descriptions)}",
    )
