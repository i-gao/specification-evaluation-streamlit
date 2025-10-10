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
                valid_items.append(f"{article_id} (\${price:.2f})")
            except ValueError:
                invalid_items.append(str(article_id))
        
        within_budget = total <= budget
        message_parts = [f"Total cost: \${total:.2f}"]
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
                    if any(keyword in product_text for keyword in ITEM_TO_KEYWORDS[required_type]):
                        actual_counts[required_type] = actual_counts.get(required_type, 0) + 1
                        break
            except ValueError:
                continue
        
        # Check if all requirements are met
        missing_items = []
        for garment_type, required_count in required_counts.items():
            actual_count = actual_counts.get(garment_type, 0)
            if actual_count < required_count:
                missing_items.append(f"{garment_type} (needed: {required_count}, found: {actual_count})")
        
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