# Extractor functions for meal planning constraints
# Each function takes a meal_plan as input and returns a tuple (value, detailed_message)
# where value is the value needed for the constraint and detailed_message describes the result

DAYS_OF_THE_WEEK = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]
MEALS_OF_THE_DAY = [
    "breakfast",
    "lunch",
    "snack",
    "dinner",
]

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


def meal_servings_at_least(meal_plan, day, meal, min_servings):
    """Return (bool, str) if servings_consumed >= min_servings for any recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] < min_servings:
            return (
                False,
                f"Insufficient servings at {day.capitalize()} {meal}: {recipe_dict['recipe'].title} has {recipe_dict['servings_consumed']} servings (need at least {min_servings})",
            )
    return (
        True,
        f"All recipes at {day.capitalize()} {meal} meet minimum serving requirement of {min_servings}",
    )


###### Concerns cooked recipes ######


def total_cooks(meal_plan):
    """Return (int, str) the total number of times cooking was required in the week."""
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


def recipes_avoid_equipment(meal_plan, day, meal, equipment):
    """Return (bool, str) if equipment is NOT in the recipe equipment for any cooked recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if not recipe_dict["cook"]:  # only check cooked recipes
            continue
        if equipment in recipe_dict["recipe"].equipment:
            return (
                False,
                f"Recipe '{recipe_dict['recipe'].title}' at {day.capitalize()} {meal} requires unavailable equipment: {equipment}",
            )
    return (
        True,
        f"All cooked recipes at {day.capitalize()} {meal} avoid unavailable equipment: {equipment}",
    )


def recipe_star_rating(meal_plan, day, meal):
    """Return (float, str) the minimum rating across cooked recipes at (day, meal), or (None, str) if not present."""
    if meal_plan[day][meal] is None:
        return None, f"No meal scheduled for {day.capitalize()} {meal}"
    ratings = []
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["cook"]:
            ratings.append((recipe_dict["recipe"].rating, recipe_dict["recipe"].title))

    if not ratings:
        return None, f"No cooked recipes at {day.capitalize()} {meal}"

    min_rating, min_recipe = min(ratings, key=lambda x: x[0])
    return (
        min_rating,
        f"Lowest rated cooked recipe at {day.capitalize()} {meal}: '{min_recipe}' with {min_rating} stars",
    )


###### Concerns consumed recipes ######


def recipes_avoid_ingredient(meal_plan, day, meal, ingredient):
    """
    Return (bool, str) if ingredient is NOT in a consumed recipe at (day, meal).
    """
    if meal_plan[day][meal] is None:
        return True, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        for recipe_ingredient in recipe_dict["recipe"].ingredients:
            if ingredient in recipe_ingredient:
                return (
                    False,
                    f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} contains forbidden ingredient: {ingredient}",
                )
    return (
        True,
        f"All consumed recipes at {day.capitalize()} {meal} avoid ingredient: {ingredient}",
    )


def recipes_include_ingredient(meal_plan, day, meal, ingredient):
    """Return (bool, str) if ingredient is in the recipe ingredients for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return False, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if ingredient in recipe_dict["recipe"].ingredients:
            return (
                True,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} contains preferred ingredient: {ingredient}",
            )
    return (
        False,
        f"No consumed recipes at {day.capitalize()} {meal} contain preferred ingredient: {ingredient}",
    )


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


def recipe_cuisine_equals_cuisine(meal_plan, day, meal, cuisine):
    """Return (bool, str) if the recipe cuisine matches for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return False, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if recipe_dict["recipe"].cuisine == cuisine:
            return (
                True,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} matches preferred cuisine: {cuisine}",
            )
    return (
        False,
        f"No consumed recipes at {day.capitalize()} {meal} match preferred cuisine: {cuisine}",
    )


def recipe_food_type_equals_food_type(meal_plan, day, meal, food_type):
    """Return (bool, str) if the recipe food type matches for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return False, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if recipe_dict["recipe"].food_type == food_type:
            return (
                True,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} matches preferred food type: {food_type}",
            )
    return (
        False,
        f"No consumed recipes at {day.capitalize()} {meal} match preferred food type: {food_type}",
    )


def num_repeated_recipes(meal_plan):
    """Return (int, str) the number of repeatedly consumed recipes in the week."""
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


def keyword_in_recipe_title(meal_plan, day, meal, keyword):
    """Return (bool, str) if the keyword is in the recipe title for any consumed recipe at (day, meal)."""
    if meal_plan[day][meal] is None:
        return False, f"No meal scheduled for {day.capitalize()} {meal}"
    for recipe_dict in meal_plan[day][meal]:
        if recipe_dict["servings_consumed"] == 0:
            continue
        if keyword.lower() in recipe_dict["recipe"].title.lower():
            return (
                True,
                f"Recipe '{recipe_dict['recipe'].title}' consumed at {day.capitalize()} {meal} is labeled as {keyword}",
            )
    return (
        False,
        f"No consumed recipes at {day.capitalize()} {meal} are labeled as {keyword}",
    )
