from typing import Tuple


def convert_height_to_cm(height_value: float, unit: str) -> float:
    """
    Convert height to centimeters.

    Args:
        height_value: Height value in the specified unit
        unit: "cm" or "in"

    Returns:
        Height in centimeters
    """
    if unit == "cm":
        return height_value
    elif unit == "in":
        return height_value * 2.54
    elif unit == "ft-in":
        # Parse feet-inches format (e.g., "5-8" means 5 feet 8 inches)
        if isinstance(height_value, str) and "-" in str(height_value):
            feet, inches = map(int, str(height_value).split("-"))
        else:
            # If it's a single number, treat as feet only
            feet = int(height_value)
            inches = 0
        return (feet * 12 + inches) * 2.54
    else:
        raise ValueError(f"Unsupported height unit: {unit}")


def convert_weight_to_kg(weight_value: float, unit: str) -> float:
    """
    Convert weight to kilograms.

    Args:
        weight_value: Weight value in the specified unit
        unit: "kg" or "lbs"

    Returns:
        Weight in kilograms
    """
    if unit == "kg":
        return weight_value
    elif unit == "lbs":
        return weight_value * 0.453592
    else:
        raise ValueError(f"Unsupported weight unit: {unit}")


def get_bmr(weight: float, height: float, age: int, sex: str) -> float:
    """
    Calculate Basal Metabolic Rate (BMR) using the Revised Harris-Benedict Equation.

    Args:
        weight: Weight in kg
        height: Height in cm
        age: Age in years
        sex: "male" or "female"

    Returns:
        Basal Metabolic Rate in calories per day

    Note:
        Uses the Revised Harris-Benedict Equation:
        - Men: BMR = 88.362 + (13.397 × weight) + (4.799 × height) - (5.677 × age)
        - Women: BMR = 447.593 + (9.247 × weight) + (3.098 × height) - (4.330 × age)
    """
    if sex == "male":
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)


def get_target_calories(
    weight: float, height: float, age: int, sex: str, activity_level: str, goal: str
) -> float:
    bmr = get_bmr(weight, height, age, sex)
    if activity_level == "sedentary":
        active_calories = 0
    elif activity_level == "lightly active":
        active_calories = 200
    elif activity_level == "moderately active":
        active_calories = 400
    elif activity_level == "very active":
        active_calories = 600
    else:
        active_calories = 0
    total_calories = bmr + active_calories

    if goal == "lose":
        calories = total_calories - 500
    elif goal == "maintain":
        calories = total_calories
    elif goal == "gain":
        calories = total_calories + 500
    else:
        calories = total_calories

    return calories


"""
Uses US dietary guidelines:
| Macronutrient     | Recommended Range (% of total calories) |
| ----------------- | --------------------------------------- |
| **Carbohydrates** | **45–65%**                              |
| **Protein**       | **10–35%**                              |
| **Fats**          | **20–35%**                              |
| **Fiber**         | **25-30g**                              |
"""


def get_healthy_carb_range(calories: float) -> Tuple[float, float]:
    return (0.45 * calories / 4, 0.65 * calories / 4)


def get_healthy_protein_range(calories: float) -> Tuple[float, float]:
    return (0.10 * calories / 4, 0.35 * calories / 4)


def get_healthy_fat_range(calories: float) -> Tuple[float, float]:
    return (0.20 * calories / 9, 0.35 * calories / 9)


def get_healthy_fiber_range(calories: float) -> Tuple[float, float]:
    return (25.0, 30.0)
