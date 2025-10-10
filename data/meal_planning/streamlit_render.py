from typing import List, Dict, Any, Optional
import streamlit as st
import uuid
from data.meal_planning.db import RecipeDB, DAYS_OF_THE_WEEK, MEALS, Recipe


def render_eval_meal(
    *, final_prediction: str, y0: Optional[str], db, auto_patch_eat_before_cook: bool
):
    """
    Render evaluation UI for Meal Planning custom specs and return (completed, feedback).
    """
    from utils.streamlit_types import FormElement, form_element_to_streamlit
    from data.meal_planning.data import (
        output_to_streamlit,
        output_to_streamlit_comparison,
    )

    st.markdown("## Evaluate the assistant's meal plan")
    with st.container(key="meal_eval_display", width="stretch"):
        try:
            if y0 is not None:
                output_to_streamlit_comparison(
                    y0,
                    final_prediction,
                    db,
                    auto_patch_eat_before_cook=auto_patch_eat_before_cook,
                )
            else:
                output_to_streamlit(
                    final_prediction,
                    db,
                    auto_patch_eat_before_cook=auto_patch_eat_before_cook,
                )
        except Exception as e:
            st.write("Error rendering plans:", str(e))

    form_elements: List[FormElement] = [
        FormElement(
            input_type="text_area",
            label="Describe the pros and cons of the plan in a few sentences.",
            height=120,
        ),
        FormElement(
            input_type="radio",
            label="Do you think more exploration could have led to a better plan?",
            options=["Yes", "Maybe", "No"],
        ),
    ]

    with st.form(key="meal_custom_eval_form"):
        feedback: Dict[str, Any] = {}
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
                    if not feedback.get(label):
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None


def render_meal_plan_streamlit(meal_plan: Dict[str, Dict[str, Any]]) -> None:
    unique_id = str(uuid.uuid4())[:8]
    _render_meal_plan_streamlit(meal_plan, unique_id)


def output_to_streamlit_comparison(
    parsed1: Dict[str, Dict[str, Any]],
    parsed2: Dict[str, Dict[str, Any]],
    db: RecipeDB,
    valid1: bool,
    valid2: bool,
    metadata1: Dict[str, Any],
    metadata2: Dict[str, Any],
) -> None:
    unique_id = str(uuid.uuid4())[:8]

    a_valid = valid1
    a_metadata = metadata1
    b_valid = valid2
    b_metadata = metadata2

    tab1, tab2 = st.tabs(["Plan A", "Plan B"])

    with tab1:
        if a_valid is not None:
            if a_valid:
                st.markdown(":small[:green[:material/check: Plan A is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan A invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (a_metadata or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        if parsed1:
            _render_meal_plan_streamlit(parsed1, f"{unique_id}_a")
        else:
            st.markdown("*Invalid meal plan*")

    with tab2:
        if b_valid is not None:
            if b_valid:
                st.markdown(":small[:green[:material/check: Plan B is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan B invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (b_metadata or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        if parsed2:
            _render_meal_plan_streamlit(parsed2, f"{unique_id}_b")
        else:
            st.markdown("*Invalid meal plan*")


# -------------------- Helpers used by Streamlit renders --------------------


def _render_meal_plan_streamlit(
    meal_plan: Dict[str, Dict[str, Any]], unique_id: str
) -> None:
    if not meal_plan:
        st.markdown("*No meal plan data available*")
        return

    st.markdown(_render_calendar_table(meal_plan), unsafe_allow_html=True)

    with st.container(horizontal=False, gap="small"):
        with st.expander("🍽️ Day-by-day breakdown", expanded=False):
            _render_detailed_sections_streamlit(meal_plan, unique_id)

        with st.expander("🍳 Which days will I have to cook?", expanded=False):
            st.markdown(_render_cooking_calendar(meal_plan), unsafe_allow_html=True)

        with st.expander("🧾 How much food will I waste?", expanded=False):
            st.markdown(
                _render_weekly_recipe_summary(meal_plan), unsafe_allow_html=True
            )

    _render_recipe_details_streamlit(meal_plan, unique_id)


MEAL_EMOJIS = {"breakfast": "🌅", "lunch": "🌞", "snack": "🍎", "dinner": "🌙"}


def _render_detailed_sections_streamlit(
    meal_plan: Dict[str, Any], unique_id: str
) -> None:
    cols = st.columns(len(DAYS_OF_THE_WEEK))
    for i, day in enumerate(DAYS_OF_THE_WEEK):
        with cols[i]:
            if day in meal_plan:

                @st.dialog(f"{day.capitalize()} details", width="large")
                def _show_day_dialog(day: str, meal_plan: Dict[str, Any]) -> None:
                    _render_day_details(day, meal_plan)

                st.button(
                    day.capitalize(),
                    on_click=_show_day_dialog,
                    args=(day, meal_plan),
                    width="stretch",
                    key=f"day_button_{day}_{unique_id}",
                )


def _render_day_details(
    day: str,
    meal_plan: Dict[str, Any],
) -> None:
    cook_index = _build_cook_index(meal_plan)

    table_data: List[List[str]] = []
    table_data.append(["Meal", "To Cook", "To Eat"])

    for m_idx, meal in enumerate(MEALS):
        items = meal_plan[day].get(meal)
        emoji = MEAL_EMOJIS.get(meal, "🍽️")
        meal_cell = f"{emoji} {meal.capitalize()}"

        if not items:
            table_data.append([meal_cell, "*Skip meal*", "*Skip meal*"])
            continue

        to_cook: List[str] = []
        to_eat: List[str] = []
        for i, item in enumerate(items):
            recipe = item.get("recipe")
            if recipe is None:
                continue

            title = recipe.title
            if item.get("cook", False):
                servings_now = int(item.get("servings_consumed", 0) or 0)
                details = f"Makes {int(recipe.num_servings)} servings"
                if servings_now > 0:
                    details += f", consume {servings_now} servings now"
                if servings_now < int(recipe.num_servings):
                    details += f", refrigerate {int(recipe.num_servings) - servings_now} servings for later"
                err = " :material/error:" if _is_invalid_recipe(recipe) else ""
                to_cook.append(
                    f":{COLORS[i % len(COLORS)]}-background[{title}{err}] (cooking time: {recipe.total_time} min)<br>{details}"
                )
                if servings_now > 0:
                    to_eat.append(
                        f":{COLORS[i % len(COLORS)]}-background[{title}{err}] (eat {servings_now} servings)"
                    )
            else:
                servings_now = int(item.get("servings_consumed", 0) or 0)
                cooked_where = _find_cooked_when(cook_index, day, meal, title)
                note = f"\tCooked on {cooked_where}" if cooked_where else ""
                err = " :material/error:" if _is_invalid_recipe(recipe) else ""
                if servings_now > 0:
                    to_eat.append(
                        f":{COLORS[i % len(COLORS)]}-background[{title}{err}] (eat {servings_now} servings)<br>{note}"
                    )

        cook_cell = "<br>".join(to_cook) if to_cook else "*No cooking*"
        eat_cell = "<br>".join(to_eat) if to_eat else "*No eating*"
        table_data.append([meal_cell, cook_cell, eat_cell])

    if len(table_data) > 1:
        table_lines: List[str] = []
        table_lines.append("| " + " | ".join(table_data[0]) + " |")
        table_lines.append("|" + "|".join(["---"] * len(table_data[0])) + "|")
        for row in table_data[1:]:
            table_lines.append("| " + " | ".join(row) + " |")
        table_markdown = "\n".join(table_lines)
        st.markdown(table_markdown, unsafe_allow_html=True)


def _render_recipe_details_streamlit(meal_plan: Dict[str, Any], unique_id: str) -> None:
    all_recipes: Dict[str, Any] = {}
    for day in DAYS_OF_THE_WEEK:
        if day not in meal_plan:
            continue
        for meal in MEALS:
            items = meal_plan[day].get(meal)
            if not items:
                continue
            for item in items:
                recipe = item.get("recipe")
                if recipe is None:
                    continue
                all_recipes[recipe.title] = recipe

    if all_recipes:
        st.markdown("### Recipe Details")
        st.markdown(
            "Click on a recipe to view its details, including ingredients, instructions, allergens, and compatible diets."
        )

        sorted_recipes = sorted(all_recipes.values(), key=lambda r: r.title)
        cols = st.columns(3)
        for i, recipe in enumerate(sorted_recipes):
            with cols[i % 3]:
                if _is_invalid_recipe(recipe):
                    st.button(
                        f":material/error: {recipe.title} (not found)",
                        disabled=True,
                        key=f"recipe_invalid_{recipe.title}_{unique_id}",
                        use_container_width=True,
                    )
                    continue

                @st.dialog(f"{recipe.title}", width="large")
                def _show_recipe_dialog(recipe: Recipe) -> None:
                    st.markdown(
                        _recipe_details(recipe),
                        unsafe_allow_html=True,
                    )

                st.button(
                    recipe.title,
                    on_click=_show_recipe_dialog,
                    args=(recipe,),
                    key=f"recipe_{recipe.title}_{unique_id}",
                    use_container_width=True,
                )


def _render_calendar_table(meal_plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("### 🗓️ Meal Plan at a Glance")
    lines.append("This is a summary of the meals you will be eating this week.")
    lines.append("* :material/nest_eco_leaf: recipe cooked fresh at that time")
    lines.append("* :material/microwave: recipe reheated from leftovers")
    lines.append("* :material/error: invalid recipe not found in database")
    lines.append("")

    header = (
        "| | " + " | ".join([f"**{d[:3].upper()}**" for d in DAYS_OF_THE_WEEK]) + " |"
    )
    sep = "|" + "|".join(["------"] * (len(DAYS_OF_THE_WEEK) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for meal in MEALS:
        row_cells = [f"**{meal.capitalize()}**"]
        for day in DAYS_OF_THE_WEEK:
            titles = _get_eaten_titles_for_cell(meal_plan, day, meal)
            cell = "<br>".join(titles) if titles else "*Skip meal*"
            row_cells.append(cell)
        lines.append("| " + " | ".join(row_cells) + " |")

    total_row_cells = ["**Daily nutrition totals**"]
    for day in DAYS_OF_THE_WEEK:
        totals = _calculate_daily_totals(meal_plan.get(day, {}))
        total_row_cells.append(
            (
                f"{totals['calories']:.0f} kcal <br> Protein {totals['protein']:.0f}g <br> Carbs {totals['carbs']:.0f}g <br> Fat {totals['fat']:.0f}g <br> Fiber {totals['fiber']:.0f}g"
            )
        )
    lines.append("| " + " | ".join(total_row_cells) + " |")

    return "\n".join(lines)


COLORS = ["gray", "primary", "orange", "yellow", "green", "blue", "violet"]


def _render_cooking_calendar(meal_plan: Dict[str, Any]) -> str:
    def _slot_total_time(recipes: List[Any]) -> int:
        times: List[int] = []
        for r in recipes:
            if r is None:
                continue
            t = int(r.total_time) if getattr(r, "total_time", 0) else 0
            if t <= 0:
                prep = int(getattr(r, "prep_time", 0) or 0)
                cook = int(getattr(r, "cook_time", 0) or 0)
                t = prep + cook
            times.append(t)
        return sum(times) if times else 0

    lines: List[str] = []
    lines.append(
        "This calendar marks where cooking is required and the total time per cooking slot."
    )
    lines.append("")

    header = (
        "| | " + " | ".join([f"**{d[:3].upper()}**" for d in DAYS_OF_THE_WEEK]) + " |"
    )
    sep = "|" + "|".join(["------"] * (len(DAYS_OF_THE_WEEK) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for meal in MEALS:
        row_cells = [f"**{meal.capitalize()}**"]
        for day in DAYS_OF_THE_WEEK:
            items = meal_plan.get(day, {}).get(meal)
            if not items:
                row_cells.append("")
                continue
            cook_recipes = [it.get("recipe") for it in items if it.get("cook", False)]
            n = len(cook_recipes)
            if n == 0:
                row_cells.append("")
            else:
                tmin = _slot_total_time(cook_recipes)
                row_cells.append(f"{n} recipe{'s' if n != 1 else ''}, total {tmin} min")
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def _render_weekly_recipe_summary(meal_plan: Dict[str, Any]) -> str:
    stats = _aggregate_weekly_recipe_stats(meal_plan)
    if not stats:
        return ""

    entries = sorted(
        stats.items(), key=lambda kv: (kv[1]["wasted"], kv[0]), reverse=True
    )

    lines: List[str] = []
    lines.append("This summary shows how much food was wasted across the week.")
    lines.append("")
    lines.append(
        "| Recipe | Cooked (servings) | Consumed (servings) | Wasted (servings) |"
    )
    lines.append("|---|---:|---:|---:|")
    for title, d in entries:
        err = " :material/error:" if d.get("invalid", False) else ""
        lines.append(
            f"| {title}{err} | {int(d['cooked'])} | {int(d['consumed'])} | {int(d['wasted'])} |"
        )
    return "\n".join(lines)


def _aggregate_weekly_recipe_stats(
    meal_plan: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for day in DAYS_OF_THE_WEEK:
        day_plan = meal_plan.get(day, {})
        for meal in MEALS:
            items = day_plan.get(meal)
            if not items:
                continue
            for it in items:
                recipe = it.get("recipe")
                if recipe is None or not getattr(recipe, "title", None):
                    continue
                title = recipe.title
                if title not in stats:
                    stats[title] = {
                        "cooked": 0.0,
                        "consumed": 0.0,
                        "wasted": 0.0,
                        "invalid": False,
                    }
                if it.get("cook", False):
                    stats[title]["cooked"] += float(
                        int(getattr(recipe, "num_servings", 0) or 0)
                    )
                stats[title]["consumed"] += float(
                    int(it.get("servings_consumed", 0) or 0)
                )
                if _is_invalid_recipe(recipe):
                    stats[title]["invalid"] = True

    for title, d in stats.items():
        cooked = d["cooked"]
        consumed = d["consumed"]
        d["wasted"] = cooked - consumed if cooked > consumed else 0.0

    return stats


def _is_invalid_recipe(recipe: Any) -> bool:
    try:
        if recipe is None:
            return True
        if (
            getattr(recipe, "ingredients", None) is None
            and getattr(recipe, "instructions", None) is None
            and getattr(recipe, "cuisine", None) is None
        ):
            return True
        if (
            isinstance(getattr(recipe, "title", None), str)
            and "Invalid recipe:" in recipe.title
        ):
            return True
        return False
    except Exception:
        return True


def _get_eaten_titles_for_cell(
    meal_plan: Dict[str, Any], day: str, meal: str
) -> List[str]:
    if day not in meal_plan or meal not in meal_plan[day] or not meal_plan[day][meal]:
        return []
    items = meal_plan[day][meal]
    eaten_titles: List[str] = []
    seen = set()
    for i, item in enumerate(items):
        recipe = item.get("recipe")
        if recipe is None:
            continue
        title = recipe.title

        # Check if recipe is invalid (has None values for required fields)
        is_invalid = _is_invalid_recipe(recipe)

        if item.get("cook", False):
            if item.get("servings_consumed", 0) > 0 and title not in seen:
                if is_invalid:
                    eaten_titles.append(f":red-background[ :material/error: {title}]")
                else:
                    eaten_titles.append(
                        f":primary-background[ :material/nest_eco_leaf: {title}]"
                    )
                seen.add(title)
        else:
            if item.get("servings_consumed", 0) > 0 and title not in seen:
                if is_invalid:
                    eaten_titles.append(f":red-background[ :material/error: {title}]")
                else:
                    eaten_titles.append(
                        f":gray-background[ :material/microwave: {title}]"
                    )
                seen.add(title)
    return eaten_titles


def _calculate_daily_totals(day_plan: Dict[str, Any]) -> Dict[str, float]:
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
    if not day_plan:
        return totals
    for meal, items in day_plan.items():
        if not items:
            continue
        for item in items:
            recipe = item.get("recipe")
            if recipe is None:
                continue
            servings_consumed = int(item.get("servings_consumed", 0) or 0)
            if servings_consumed > 0:
                totals["calories"] += (recipe.calories or 0.0) * servings_consumed
                totals["protein"] += (recipe.protein or 0.0) * servings_consumed
                totals["carbs"] += (
                    recipe.total_carbohydrate or 0.0
                ) * servings_consumed
                totals["fat"] += (recipe.total_fat or 0.0) * servings_consumed
                fiber_val = getattr(recipe, "dietary_fiber", None) or 0.0
                totals["fiber"] += fiber_val * servings_consumed
    return totals


def _build_cook_index(meal_plan: Dict[str, Any]) -> Dict[str, List[tuple]]:
    index: Dict[str, List[tuple]] = {}
    for d_idx, day in enumerate(DAYS_OF_THE_WEEK):
        day_plan = meal_plan.get(day, {})
        for m_idx, meal in enumerate(MEALS):
            items = day_plan.get(meal)
            if not items:
                continue
            for item in items:
                if item.get("cook", False) and item.get("recipe") is not None:
                    title = item["recipe"].title
                    index.setdefault(title, []).append((d_idx, m_idx))
    return index


def _find_cooked_when(
    cook_index: Dict[str, List[tuple]],
    day: str,
    meal: str,
    title: str,
) -> str:
    if title not in cook_index:
        return ""
    d_idx = DAYS_OF_THE_WEEK.index(day)
    m_idx = MEALS.index(meal)
    candidates = [
        (di, mi)
        for (di, mi) in cook_index[title]
        if (di < d_idx) or (di == d_idx and mi <= m_idx)
    ]
    if not candidates:
        return ""
    di, mi = sorted(candidates)[-1]
    day_label = DAYS_OF_THE_WEEK[di][:3].capitalize()
    meal_label = MEALS[mi].capitalize()
    return f"{day_label} {meal_label}"


def _recipe_details(recipe: Recipe) -> str:
    lines: List[str] = []

    basic_info: List[str] = []
    if getattr(recipe, "cuisine", None):
        basic_info.append(f"**Cuisine:** {recipe.cuisine}")
    if getattr(recipe, "rating", 0) > 0:
        basic_info.append(f"**Rating:** {recipe.rating:.1f}⭐")
    if getattr(recipe, "num_reviews", 0) > 0:
        basic_info.append(f"**Reviews:** {recipe.num_reviews}")

    dietary_info: List[str] = []
    if getattr(recipe, "diet", None):
        dietary_info.append(f"**Compatible diets:** {', '.join(recipe.diet)}")
    if getattr(recipe, "intolerances", None):
        dietary_info.append(f"**Allergens:** {', '.join(recipe.intolerances)}")

    nutrition_info: List[str] = []
    if getattr(recipe, "calories", 0) > 0:
        nutrition_info.append(f"**Calories (per serving):** {recipe.calories:.0f}")
    if getattr(recipe, "protein", 0) > 0:
        nutrition_info.append(f"**Protein:** {recipe.protein:.1f}g")
    if getattr(recipe, "total_fat", 0) > 0:
        nutrition_info.append(f"**Fat:** {recipe.total_fat:.1f}g")
    if getattr(recipe, "total_carbohydrate", 0) > 0:
        nutrition_info.append(
            f"**Total Carbohydrates:** {recipe.total_carbohydrate:.1f}g"
        )

    timing_info: List[str] = []
    if getattr(recipe, "prep_time", 0) > 0:
        timing_info.append(f"**Prep:** {recipe.prep_time} min")
    if getattr(recipe, "cook_time", 0) > 0:
        timing_info.append(f"**Cook:** {recipe.cook_time} min")
    if getattr(recipe, "total_time", 0) > 0:
        timing_info.append(f"**Total:** {recipe.total_time} min")

    equipment_info: List[str] = []
    if getattr(recipe, "equipment", None):
        equipment_info.append(f"**Equipment:** {', '.join(recipe.equipment)}")

    if _is_invalid_recipe(recipe):
        lines.append(
            ":material/error: *The assistant recommended an invalid recipe that is not in the database.*"
        )
        lines.append("</details>")
        lines.append("  ")
        return "\n".join(lines)

    if basic_info:
        lines.append("  " + " | ".join(basic_info))
        lines.append("  ")

    if dietary_info:
        lines.append("  " + " | ".join(dietary_info))
        lines.append("  ")

    if timing_info:
        lines.append("  " + " | ".join(timing_info))
        lines.append("  ")

    if nutrition_info:
        lines.append("  " + " | ".join(nutrition_info))
        lines.append("  ")

    if equipment_info:
        lines.append("  " + " | ".join(equipment_info))
        lines.append("  ")

    if getattr(recipe, "ingredients", None):
        lines.append("  **Ingredients:**")
        for ingredient in recipe.ingredients:
            lines.append(f"  - {ingredient}")
        lines.append("  ")

    if getattr(recipe, "instructions", None):
        lines.append("  **Instructions:**")
        lines.append(f"{recipe.instructions}")

    return "\n".join(lines)


def render_recipe_mentions(recipe_names: List[str], db: RecipeDB) -> None:
    """
    Render a section showing mentioned recipes with their details.
    Uses a grid of buttons that launch dialogs, similar to workout planning exercises.
    """
    if not recipe_names:
        return

    st.markdown('<div id="mentioned-recipes"></div>', unsafe_allow_html=True)
    st.markdown(
        "Click on a recipe to view its details, including ingredients, instructions, nutrition facts, cooking time, and dietary information."
    )

    # Get unique recipes (first occurrence only as requested)
    seen_recipes = set()
    unique_recipes = []
    for recipe_name in recipe_names:
        if recipe_name not in seen_recipes:
            seen_recipes.add(recipe_name)
            unique_recipes.append(recipe_name)

    if not unique_recipes:
        return

    # Create a grid of buttons (3 columns)
    cols = st.columns(3)
    for i, recipe_name in enumerate(unique_recipes):
        with cols[i % 3]:
            try:
                # Try to find the recipe in the database
                recipe = db.get_recipe_by_name(recipe_name)
                if recipe is not None:
                    # Create dialog for valid recipe
                    @st.dialog(f"{recipe_name}", width="large")
                    def _show_recipe_dialog(recipe: Recipe) -> None:
                        st.markdown(
                            _recipe_details(recipe),
                            unsafe_allow_html=True,
                        )

                    st.button(
                        f":material/skillet: {recipe_name}",
                        on_click=_show_recipe_dialog,
                        args=(recipe,),
                        key=f"recipe_{recipe_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
                else:
                    # Recipe not found in database - show disabled button
                    st.button(
                        f":material/error: {recipe_name} (not found)",
                        disabled=True,
                        key=f"recipe_invalid_{recipe_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
            except Exception:
                # Error finding recipe - show disabled button
                st.button(
                    f":material/error: {recipe_name} (error)",
                    disabled=True,
                    key=f"recipe_error_{recipe_name}_{uuid.uuid4().hex[:8]}",
                    use_container_width=True,
                )
