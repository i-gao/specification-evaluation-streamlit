from typing import List, Dict, Any, Callable
import streamlit as st
import uuid
from data.meal_planning.db import RecipeDB, DAYS_OF_THE_WEEK, MEALS, Recipe
from data.meal_planning.parser import parse_meal_plan
import random
from evaluation.qualitative_eval import COMPARISON_LIKERT
from data.reward import likert_to_win_rate, pairwise_win_rate


def render_eval(*, final_prediction: str, y0: str, db: RecipeDB):
    ranking_done = rank_recipes(final_prediction=final_prediction, y0=y0, db=db)
    if not ranking_done:
        return False, None

    comparison_done = render_comparison(final_prediction=final_prediction, y0=y0, db=db)
    if not comparison_done:
        return False, None

    # Compute the final score
    prediction_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["recipe"][
                "predicted_ranks"
            ].values()
        )
    ]
    y0_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["recipe"][
                "y0_ranks"
            ].values()
        )
    ]
    p_recipe_wins = pairwise_win_rate(prediction_rankings, y0_rankings)
    other_wins, total = likert_to_win_rate(
        [
            st.session_state.form_results["final_evaluation"][
                "cooking_schedule_preference"
            ],
            st.session_state.form_results["final_evaluation"]["freshness_preference"],
            st.session_state.form_results["final_evaluation"]["variety_preference"],
            st.session_state.form_results["final_evaluation"][
                "schedule_fit_preference"
            ],
            st.session_state.form_results["final_evaluation"]["nutritional_preference"],
        ],
        return_total=True,
    )
    p_wins = (p_recipe_wins + other_wins) / (total + 1)
    st.session_state.form_results["final_evaluation"]["score"] = p_wins

    return True, None


def rank_recipes(*, final_prediction: str, y0: str, db: RecipeDB):
    predicted = parse_meal_plan(final_prediction, db)
    y0 = parse_meal_plan(y0, db)

    done = "recipe" in st.session_state.form_results["final_evaluation"]
    if not done:
        predicted_recipes = [
            recipe
            for day in DAYS_OF_THE_WEEK
            for meal in MEALS
            for recipe in (
                predicted[day][meal] if predicted[day][meal] is not None else []
            )
        ]
        y0_recipes = [
            recipe
            for day in DAYS_OF_THE_WEEK
            for meal in MEALS
            for recipe in (y0[day][meal] if y0[day][meal] is not None else [])
        ]
        _render_carousel(
            predicted_recipes,
            y0_recipes,
            show_k=None,
        )
        return

    return True


@st.fragment
def _render_carousel(
    predicted: List[Dict[str, Any]],
    y0: List[Dict[str, Any]],
    name: str = "recipe",
    md_fn: callable = lambda d: _recipe_details(d["recipe"]),
    filter_fn: Callable[[Dict[str, Any]], bool] = None,
    show_k: int = None,
):
    """
    Args:
        predicted: list of predicted options
        y0: list of y0 options
        name: name of the thing being ranked. Used for saving to session state.
        md_fn: function to render the option
        filter_fn: function to filter the options
        show_k: number of options to show

    Adds to session state:
        - ranking: a dict mapping a rank (0-indexed) to a name
        - y0_ranks: a dict mapping name to a rank
        - predicted_ranks: a list of ranks for the predicted options
    """
    predicted = [
        p
        for p in predicted
        if p is not None and (filter_fn is None or filter_fn(p))
    ]
    predicted = list({d["recipe"].title: d for d in predicted}.values())

    y0 = [
        p for p in y0 if p is not None and (filter_fn is None or filter_fn(p))
    ]
    y0 = list({d["recipe"].title: d for d in y0}.values())

    if len(predicted) == 0:
        # set difference is the entire y0, and y0 auto-wins
        dummy_rank = {i: y["recipe"].title for i, y in enumerate(y0)}
        st.session_state.form_results["final_evaluation"][name] = {
            "ranking": dummy_rank,
            "y0_ranks": {v: k for k, v in dummy_rank.items()},
            "predicted_ranks": {},
        }
        st.rerun()

    # find the set difference
    predicted_names = set([p["recipe"].title for p in predicted])
    y0_names = set([p["recipe"].title for p in y0])
    diff_names = (predicted_names - y0_names).union(y0_names - predicted_names)
    if not diff_names:
        # don't render anything
        return

    predicted_options = [p for p in predicted if p["recipe"].title in diff_names]
    y0_options = [p for p in y0 if p["recipe"].title in diff_names]
    if show_k is not None and len(diff_names) > show_k:
        # try to get a roughly balanced set of options
        if len(predicted_options) < show_k / 2:
            options = predicted_options + y0_options[: show_k - len(predicted_options)]
        elif len(y0_options) < show_k / 2:
            options = predicted_options[: show_k - len(y0_options)] + y0_options
        else:
            options = (
                predicted_options[: show_k // 2 + show_k % 2]
                + y0_options[: show_k // 2]
            )
    else:
        options = predicted_options + y0_options

    # Stabilize the options order across reruns within this fragment
    state_key = f"options_order_{name}"
    if state_key not in st.session_state:
        # Store stable order as indices into the current options list
        order = list(range(len(options)))
        random.shuffle(order)
        st.session_state[state_key] = order
    order = st.session_state[state_key]
    # Reorder options according to stored order, truncating/expanding safely
    if len(order) != len(options):
        order = list(range(len(options)))
        st.session_state[state_key] = order
    options = [options[i] for i in order]

    # display the carousel
    from evaluation.app.components import carousel

    def display_fn(i):
        st.markdown(f"### {options[i]['recipe'].title}")
        st.markdown(
            md_fn(
                options[i],
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Review the assistant's recommendations")
    st.markdown(f"The assistant has recommended {len(options)} {name}s for you.")
    carousel([lambda i=i: display_fn(i) for i in range(len(options))], height=300)

    with st.form(key=f"ranking_form_{name}"):
        rank = st.multiselect(
            f"Rank the {name} above from MOST to LEAST preferred.",
            [i for i in range(len(options))],
            default=[],
            format_func=lambda x: f"Option {x + 1}: {options[x]['recipe'].title}",
            key=f"ranking_multiselect_{name}",
        )
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            print(rank)
            print(len(rank), len(options))
            if len(rank) != len(options):
                st.error("Please rank all options")
                return
            ranking = {i: options[i]["recipe"].title for i in rank}
            st.session_state.form_results["final_evaluation"][name] = {
                "ranking": ranking,
                "y0_ranks": {v: k for k, v in ranking.items() if v in y0_names},
                "predicted_ranks": {
                    v: k for k, v in ranking.items() if v in predicted_names
                },
            }
            st.rerun()


def render_comparison(*, final_prediction: str, y0: str, db: RecipeDB):
    predicted = parse_meal_plan(final_prediction, db)
    y0 = parse_meal_plan(y0, db)

    with st.container(border=True):
        st.markdown("## Compare these meal plans")

        output_to_streamlit_comparison(
            predicted, y0, db, valid1=None, valid2=None, metadata1=None, metadata2=None
        )

    st.divider()

    # Render form
    with st.form(key="meal_planning_comparison_form"):
        cooking_schedule_preference = st.radio(
            "Compare the **cooking schedule** of the recipes in meal plans A and B. Do you have a preference?",
            options=["-"] + COMPARISON_LIKERT,
        )
        freshness_preference = st.radio(
            "Compare the **balance of eating new meals vs. eating leftovers** in meal plans A and B. Do you have a preference?",
            options=["-"] + COMPARISON_LIKERT,
        )
        variety_preference = st.radio(
            "Compare the **variety of meals** in meal plans A and B. Do you have a preference?",
            options=["-"] + COMPARISON_LIKERT,
        )
        schedule_fit_preference = st.radio(
            "Compare how well meal plans A and B **fit into your upcoming schedule,** accounting for your existing plans / time constraints. Which one do you prefer?",
            options=["-"] + COMPARISON_LIKERT,
        )
        nutritional_preference = st.radio(
            "Compare the **nutritional profiles** of meal plans A and B. Do you have a preference?",
            options=["-"] + COMPARISON_LIKERT,
        )
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            if any(
                v is None or v == "-"
                for v in [
                    cooking_schedule_preference,
                    freshness_preference,
                    variety_preference,
                    schedule_fit_preference,
                    nutritional_preference,
                ]
            ):
                st.error("Please fill out all fields")
                return
            st.session_state.form_results["final_evaluation"].update(
                {
                    "cooking_schedule_preference": cooking_schedule_preference,
                    "freshness_preference": freshness_preference,
                    "variety_preference": variety_preference,
                    "schedule_fit_preference": schedule_fit_preference,
                    "nutritional_preference": nutritional_preference,
                }
            )
    return True


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
