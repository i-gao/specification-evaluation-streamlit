# Extractor functions for shopping constraints
# Each function takes a shopping_cart (comma-separated article IDs) and catalog as input
# and returns a tuple (value, detailed_message) where value is the value needed for the constraint
# and detailed_message describes the result

"""
Assume a shopping cart is a string of comma-separated article IDs, e.g., "123456,123457,123458"
The catalog is a pandas DataFrame with product information including:
- article_id: unique product identifier
- price: product price
- product_group_name: main category (e.g., "Ladies", "Men", "Kids")
- product_type_name: specific type (e.g., "Hoodie", "Dress", "Jeans")
- colour_group_name: color group (e.g., "Black", "Blue", "Red")
- department_name: department (e.g., "Ladieswear", "Menswear")
- index_group_name: brand/collection (e.g., "H&M Basics", "H&M Trend")
- section_name: section (e.g., "Casual", "Formal")
- garment_group_name: garment type (e.g., "Tops", "Bottoms", "Dresses")
"""

from typing import List


def total_cost(shopping_cart: List[int], catalog, budget):
    """
    Calculate the total cost of items in the shopping cart.
    If budget is provided, returns boolean indicating if cost is within budget.
    Returns (float/bool, str): (total_cost or within_budget, message)
    """
    try:
        if not shopping_cart:
            return True, "Shopping cart is empty (within budget)"

        total = 0.0
        valid_items = []
        invalid_items = []

        for article_id in shopping_cart:
            try:
                product = catalog.get_row_by_article_id(article_id)
                price = float(product["price"])
                total += price
                valid_items.append(f"{article_id} (${price:.2f})")
            except ValueError:
                invalid_items.append(str(article_id))

        within_budget = total <= budget
        message_parts = [f"Total cost: ${total:.2f}"]
        return within_budget, "; ".join(message_parts)

    except Exception as e:
        if budget is not None:
            return False, f"Error calculating total cost: {str(e)}"
        return 0.0, f"Error calculating total cost: {str(e)}"


ITEM_TO_KEYWORDS = {
    "hoodie": ["hoodie", "hood"],
    "dress": ["dress"],
    "shoes": ["sneakers", "shoes", "boots", "booties", "sandals"],
    "pants": ["pants", "trousers"],
    "shirt": ["shirt", "t-shirt", "polo", "blouse", "top"],
    "shorts": ["shorts"],
    "jacket": ["jacket", "coat"],
    "sweater": ["sweater", "knitwear", "cardigan"],
    "blouse": ["blouse", "top", "shirt", "vest top"],
    "sleeveless top": ["sleeveless top", "top", "vest top"],
    "hat": ["hat", "cap", "beanie", "bucket"],
}


def matches_prompt(shopping_cart: List[int], catalog, prompt):
    """
    Check if shopping cart matches the prompt requirements.
    Prompt is a list of garment types where frequency indicates quantity needed.
    Returns (bool, str): (matches_prompt, detailed_message)
    """
    try:
        if not shopping_cart:
            return False, "Shopping cart is empty"

        if not prompt or not isinstance(prompt, list):
            return True, "No prompt requirements specified"

        # Count required garment types from prompt
        required_counts = {}
        for garment_type in prompt:
            garment_lower = garment_type.lower()
            required_counts[garment_lower] = required_counts.get(garment_lower, 0) + 1

        # Count actual garment types in shopping cart
        actual_counts = {}
        for article_id in shopping_cart:
            try:
                product = catalog.get_row_by_article_id(article_id)
                # Check multiple fields for garment type
                product_text = f"{product['product_type_name']} {product['garment_group_name']} {product['prod_name']}".lower()

                # Try to match against required garment types
                for required_type in required_counts.keys():
                    if any(
                        keyword in product_text
                        for keyword in ITEM_TO_KEYWORDS[required_type]
                    ):
                        actual_counts[required_type] = (
                            actual_counts.get(required_type, 0) + 1
                        )
                        break
            except ValueError:
                continue

        # Check if all requirements are met
        missing_items = []
        for garment_type, required_count in required_counts.items():
            actual_count = actual_counts.get(garment_type, 0)
            if actual_count < required_count:
                missing_items.append(
                    f"{garment_type} (needed: {required_count}, found: {actual_count})"
                )

        matches = len(missing_items) == 0
        if matches:
            message = "The cart matches the prompt requirements."
        else:
            message = "The cart does not match the prompt requirements."
        if missing_items:
            message += f" Missing: {', '.join(missing_items)}"

        return matches, message

    except Exception as e:
        return False, f"Error checking prompt match: {str(e)}"


def all_ids_valid(shopping_cart: List[int], catalog):
    """
    Check that all article IDs exist in the catalog.
    Returns (bool, str)
    """
    try:
        invalid_ids = []
        for article_id in shopping_cart or []:
            try:
                catalog.get_row_by_article_id(int(article_id))
            except ValueError:
                invalid_ids.append(str(article_id))
        if invalid_ids:
            return False, f"Invalid catalog IDs: {', '.join(invalid_ids)}"
        return True, "All article IDs valid"
    except Exception as e:
        return False, f"Error validating IDs: {str(e)}"


def num_items_equals(shopping_cart: List[int], true_products):
    """
    Return a boolean indicating exact match of the number of items
    between predicted cart and true_products.
    """
    try:
        n_pred = len(shopping_cart or [])
        n_true = len(true_products) if hasattr(true_products, "__len__") else 0
        ok = n_pred == n_true
        return ok, f"Items count: pred={n_pred}, true={n_true}"
    except Exception as e:
        return False, f"Error comparing item counts: {str(e)}"


def identity_cart(shopping_cart: List[int]):
    """
    Return the predicted article_id list unchanged for set-based constraints.
    """
    try:
        return list(map(int, shopping_cart or [])), "cart ids"
    except Exception as e:
        return [], f"Error returning cart ids: {str(e)}"


def column_values(shopping_cart: List[int], catalog, column: str):
    """
    Return the list of predicted values for a catalog column.
    """
    try:
        vals = []
        for article_id in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(article_id))
                vals.append(str(row.get(column)))
            except ValueError:
                continue
        vals = [v for v in vals if v is not None]
        return vals, f"Column {column} values: {', '.join(map(str, vals))}"
    except Exception as e:
        return [], f"Error collecting column values: {str(e)}"


# ----- Derived attribute parsing from text -----

_NECKLINE_KEYWORDS = {
    "v-neck": ["v-neck", "v neck"],
    "crew": ["crew neck", "round neck", "crewneck"],
    "turtleneck": ["turtleneck", "roll-neck", "roll neck"],
    "polo": ["polo collar", "polo"],
    "collared": ["collar", "notch lapels"],
    "hooded": ["hood", "hooded"],
}

_SLEEVE_LENGTH_KEYWORDS = {
    "long": ["long-sleeved", "long sleeve", "long sleeves"],
    "short": ["short-sleeved", "short sleeve", "short sleeves"],
    "sleeveless": ["sleeveless", "no sleeves"],
}

_DRESS_LENGTH_KEYWORDS = {
    "mini": ["mini"],
    "midi": ["midi"],
    "maxi": ["maxi"],
    "knee-length": ["knee-length", "knee length"],
    "short": ["short"],
    "long": ["long"],
}

_WARM_KEYWORDS = [
    "padded",
    "wool",
    "knit",
    "knitted",
    "fleece",
    "quilted",
    "down",
    "insulated",
    "thermal",
    "jacket",
    "coat",
    "sweater",
    "hood",
]
_COLD_KEYWORDS = ["sleeveless", "shorts", "crop", "short-sleeved", "sandals"]

_PATTERN_KEYWORDS = {
    "solid": ["solid"],
    "stripe": ["stripe", "striped"],
    "check": ["check", "checked", "checkered"],
    "all over pattern": ["all over pattern", "pattern"],
    "melange": ["melange"],
}

_MATERIAL_KEYWORDS = {
    "cotton": ["cotton"],
    "polyester": ["polyester"],
    "wool": ["wool"],
    "denim": ["denim"],
    "leather": ["leather"],
    "viscose": ["viscose"],
    "linen": ["linen"],
    "silk": ["silk"],
}

_FIT_KEYWORDS = {
    "regular": ["regular fit", "regular"],
    "slim": ["slim fit", "slim"],
    "skinny": ["skinny"],
    "loose": ["loose fit", "loose", "relaxed"],
    "oversized": ["oversized"],
}

_CLOSURE_KEYWORDS = {
    "zip": ["zip", "zipper", "zipped"],
    "button": ["button", "buttoned", "buttons"],
    "drawstring": ["drawstring"],
    "lace": ["lace-up", "lacing"],
}

_LEG_STYLE_KEYWORDS = {
    "tapered": ["tapered"],
    "straight": ["straight"],
    "wide": ["wide"],
    "skinny": ["skinny"],
    "shorts": ["short shorts", "shorts"],
}

_SPORT_KEYWORDS = {
    "sport": [
        "sport",
        "running",
        "training",
        "gym",
        "athletic",
        "workout",
        "tennis",
        "soccer",
        "football",
        "basketball",
        "athleisure",
    ]
}

_GENDER_KEYWORDS = {
    "man_boy": [
        "menswear",
        "men",
        "boy",
        "boys",
        "young boy",
        "kids boy",
        "men's",
        "men ",
        " man ",
        " boy ",
    ],
    "woman_girl": [
        "ladieswear",
        "womenswear",
        "womens",
        "women",
        "woman",
        "girl",
        "girls",
        "young girl",
        "kids girl",
        "women's",
        "women ",
        " woman ",
        " girl ",
    ],
}

_AGE_KEYWORDS = {
    "baby": [
        "baby",
        "baby sizes",
        "baby sizes 50-98",
    ],
    "kid": [
        "children",
        "kids",
        "young boy",
        "young girl",
        "children sizes",
        "baby/children",
    ],
    "adult": [
        "menswear",
        "ladieswear",
        "womenswear",
    ],
}


def _row_text(row) -> str:
    parts = []
    for col in [
        "search_area",
        "detail_desc",
        "prod_name",
        "product_type_name",
        "section_name",
        "garment_group_name",
        "index_name",
        "department_name",
    ]:
        try:
            v = str(row.get(col, "")).lower()
        except Exception:
            v = ""
        parts.append(v)
    return " ".join(parts)


def _any_in(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _derive_tags(text: str, keyword_map) -> set:
    tags = set()
    for tag, kws in keyword_map.items():
        if _any_in(text, kws):
            tags.add(tag)
    return tags


def _derive_tags_concatenated(text: str, keyword_map) -> str:
    """
    Derive tags from text and concatenate them with '+' as a single tag string.
    If no tags found, returns empty string.
    Tags are sorted for consistent concatenation.
    """
    tags = _derive_tags(text, keyword_map)
    if not tags:
        return ""
    return "+".join(sorted(tags))


def _is_upper_or_full(row) -> bool:
    g = str(row.get("product_group_name", "")).lower()
    return ("upper body" in g) or ("full body" in g)


def _is_lower_or_full(row) -> bool:
    g = str(row.get("product_group_name", "")).lower()
    return ("lower body" in g) or ("full body" in g)


def _is_dress_or_skirt(row) -> bool:
    t = str(row.get("product_type_name", "")).lower()
    return ("dress" in t) or ("skirt" in t)


def per_item_winter_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=winter-friendly, 0 otherwise) for the predicted cart.
    """
    try:

        def is_winter(row) -> int:
            text = _row_text(row)
            warm = any(w in text for w in _WARM_KEYWORDS)
            cold = any(c in text for c in _COLD_KEYWORDS)
            g = str(row.get("section_name", "")).lower()
            outer = "outerwear" in g or "outer" in g
            return 1 if (warm or outer) and not cold else 0

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            vals.append(is_winter(row))
        return vals, f"Winter flags: {vals}"
    except Exception as e:
        return [], f"Error computing winter flags: {str(e)}"


def per_item_summer_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=summer-friendly, 0 otherwise) for the predicted cart.
    """
    try:

        def is_summer(row) -> int:
            text = _row_text(row)
            cold = any(c in text for c in _COLD_KEYWORDS)
            warm = any(w in text for w in _WARM_KEYWORDS)
            return 1 if cold and not warm else 0

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            vals.append(is_summer(row))
        return vals, f"Summer flags: {vals}"
    except Exception as e:
        return [], f"Error computing summer flags: {str(e)}"


def neckline_tags(shopping_cart: List[int], catalog):
    """
    Return the list of neckline tags (concatenated per product) for predicted cart.
    Only includes tags from applicable items (upper or full body).
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            if _is_upper_or_full(row):
                tag_str = _derive_tags_concatenated(_row_text(row), _NECKLINE_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
        return tags, f"Neckline tags: {tags}"
    except Exception as e:
        return [], f"Error collecting neckline tags: {str(e)}"


def sleeve_length_tags(shopping_cart: List[int], catalog):
    """
    Return the list of sleeve length tags (concatenated per product) for predicted cart.
    Only includes tags from applicable items (upper or full body).
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            if _is_upper_or_full(row):
                tag_str = _derive_tags_concatenated(
                    _row_text(row), _SLEEVE_LENGTH_KEYWORDS
                )
                if tag_str:
                    tags.append(tag_str)
        return tags, f"Sleeve length tags: {tags}"
    except Exception as e:
        return [], f"Error collecting sleeve length tags: {str(e)}"


def dress_length_tags(shopping_cart: List[int], catalog):
    """
    Return the list of dress/skirt length tags (concatenated per product) for predicted cart.
    Only includes tags from applicable items (dresses or skirts).
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            if _is_dress_or_skirt(row):
                tag_str = _derive_tags_concatenated(
                    _row_text(row), _DRESS_LENGTH_KEYWORDS
                )
                if tag_str:
                    tags.append(tag_str)
        return tags, f"Dress length tags: {tags}"
    except Exception as e:
        return [], f"Error collecting dress length tags: {str(e)}"


def material_tags(shopping_cart: List[int], catalog):
    """
    Return the list of material tags (concatenated per product) for predicted cart.
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = _derive_tags_concatenated(_row_text(row), _MATERIAL_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Material tags: {tags}"
    except Exception as e:
        return [], f"Error collecting material tags: {str(e)}"


def fit_tags(shopping_cart: List[int], catalog):
    """
    Return the list of fit tags (concatenated per product) for predicted cart.
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = _derive_tags_concatenated(_row_text(row), _FIT_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Fit tags: {tags}"
    except Exception as e:
        return [], f"Error collecting fit tags: {str(e)}"


def closure_tags(shopping_cart: List[int], catalog):
    """
    Return the list of closure tags (concatenated per product) for predicted cart.
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = _derive_tags_concatenated(_row_text(row), _CLOSURE_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Closure tags: {tags}"
    except Exception as e:
        return [], f"Error collecting closure tags: {str(e)}"


def leg_style_tags(shopping_cart: List[int], catalog):
    """
    Return the list of leg style tags (concatenated per product) for predicted cart.
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = _derive_tags_concatenated(_row_text(row), _LEG_STYLE_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Leg style tags: {tags}"
    except Exception as e:
        return [], f"Error collecting leg style tags: {str(e)}"


def gender_tags(shopping_cart: List[int], catalog):
    """
    Return the list of gender tags (concatenated per product) for predicted cart.
    Returns (list, str): (list of tag strings, message)
    """
    try:
        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = _derive_tags_concatenated(_row_text(row), _GENDER_KEYWORDS)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Gender tags: {tags}"
    except Exception as e:
        return [], f"Error collecting gender tags: {str(e)}"


def age_tags(shopping_cart: List[int], catalog):
    """
    Return the list of age tags for predicted cart.
    Returns (list, str): (list of age category strings, message)
    """
    try:

        def derive_age_tag(row) -> str:
            text = _row_text(row)
            if "baby/children" not in text and _any_in(text, _AGE_KEYWORDS["baby"]):
                return "baby"
            if _any_in(text, _AGE_KEYWORDS["kid"]):
                return "kid"
            if _any_in(text, _AGE_KEYWORDS["adult"]):
                return "adult"
            if "menswear" in text or "ladieswear" in text or "womenswear" in text:
                return "adult"
            return ""

        tags = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                tag_str = derive_age_tag(row)
                if tag_str:
                    tags.append(tag_str)
            except ValueError:
                continue
        return tags, f"Age tags: {tags}"
    except Exception as e:
        return [], f"Error collecting age tags: {str(e)}"


def product_type_values(shopping_cart: List[int], catalog):
    """
    Return the list of product_type_name values for predicted cart (multiset).
    Returns (list, str): (list of product type strings, message)
    """
    try:
        vals = []
        for article_id in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(article_id))
                t = str(row.get("product_type_name"))
                if t and t != "nan":
                    vals.append(t)
            except ValueError:
                continue
        return vals, f"Product type values: {vals}"
    except Exception as e:
        return [], f"Error collecting product type values: {str(e)}"


def per_item_hood_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=has hood, 0 otherwise) for the predicted cart.
    Returns (list, str): (list of flags, message)
    """
    try:

        def has_hood(row) -> int:
            t = _row_text(row)
            return 1 if ("hood" in t or "hooded" in t) else 0

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                vals.append(has_hood(row))
            except ValueError:
                continue
        return vals, f"Hood flags: {vals}"
    except Exception as e:
        return [], f"Error computing hood flags: {str(e)}"


def per_item_elasticity_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=elastic, 0 otherwise) for the predicted cart.
    Returns (list, str): (list of flags, message)
    """
    try:

        def is_elastic(row) -> int:
            t = _row_text(row)
            return (
                1
                if (
                    "elastic" in t
                    or "elastication" in t
                    or "stretch" in t
                    or "spandex" in t
                )
                else 0
            )

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                vals.append(is_elastic(row))
            except ValueError:
                continue
        return vals, f"Elasticity flags: {vals}"
    except Exception as e:
        return [], f"Error computing elasticity flags: {str(e)}"


def per_item_sustainability_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=sustainable, 0 otherwise) for the predicted cart.
    Returns (list, str): (list of flags, message)
    """
    try:

        def is_sustainable(row) -> int:
            t = _row_text(row)
            return 1 if ("organic" in t or "recycled" in t) else 0

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
                vals.append(is_sustainable(row))
            except ValueError:
                continue
        return vals, f"Sustainability flags: {vals}"
    except Exception as e:
        return [], f"Error computing sustainability flags: {str(e)}"


def per_item_sport_flags(shopping_cart: List[int], catalog):
    """
    Return per-item flags (1=sporty, 0 otherwise) for the predicted cart.
    """
    try:

        def is_sport(row) -> int:
            text = _row_text(row)
            if any(k in text for k in _SPORT_KEYWORDS):
                return 1
            dept = str(row.get("department_name", "")).lower()
            idx = str(row.get("index_name", "")).lower()
            idxg = str(row.get("index_group_name", "")).lower()
            sec = str(row.get("section_name", "")).lower()
            return (
                1
                if (
                    "sport" in dept
                    or "sport" in idx
                    or "sport" in idxg
                    or "sport" in sec
                )
                else 0
            )

        vals = []
        for aid in shopping_cart or []:
            try:
                row = catalog.get_row_by_article_id(int(aid))
            except ValueError:
                continue
            vals.append(is_sport(row))
        return vals, f"Sport flags: {vals}"
    except Exception as e:
        return [], f"Error computing sport flags: {str(e)}"


# ---------- Description/spec builders for dataset wiring ----------


def _unique_values(df, col: str, unique: bool = True) -> List[str]:
    """
    Get values from a column in a dataframe.

    Args:
        df: DataFrame to extract from
        col: Column name
        unique: If True, return unique sorted values. If False, return all values as list.
    """
    if col not in df:
        return []
    if unique:
        return sorted(list(dict.fromkeys(df[col].dropna().astype(str).tolist())))
    else:
        return df[col].dropna().astype(str).tolist()


def _format_list(vals: List[str], max_items: int = 6) -> str:
    vals = [v for v in vals if v]
    if len(vals) > max_items:
        return ", ".join(vals[:max_items]) + ", ..."
    return ", ".join(vals)


def _derive_union_tags_df(df, keyword_map, applicable_fn=None) -> List[str]:
    """
    Derive concatenated tags per product and return as a list of tag strings.
    Each product's tags are concatenated with '+' as a single tag string.

    Args:
        df: DataFrame to extract from
        keyword_map: Keyword map for tag derivation
        applicable_fn: Optional function(row) -> bool to filter which rows to process
    """
    tags = []
    for _, row in df.iterrows():
        if applicable_fn is not None and not applicable_fn(row):
            continue
        tag_str = _derive_tags_concatenated(_row_text(row), keyword_map)
        if tag_str:
            tags.append(tag_str)
    return sorted(tags)


def _fraction_true_df(df, flag_fn) -> float:
    vals = []
    for _, row in df.iterrows():
        vals.append(1 if flag_fn(row) else 0)
    return (sum(vals) / len(vals)) if vals else 0.0
