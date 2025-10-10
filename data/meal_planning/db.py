import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from functools import cached_property
from typing import List, Optional
import pandas as pd
import sys
import os
from data.database import Database


# Global constants
DAYS_OF_THE_WEEK = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]
MEALS = [
    "breakfast",
    "lunch",
    "dinner",
    # "snack",
]
DIETS = [
    "Gluten free",
    "Ketogenic",
    "Vegetarian",
    "Lacto-Vegetarian",
    "Ovo-Vegetarian",
    "Vegan",
    "Pescetarian",
    "Paleo",
    "Primal",
    "Whole30",
    "Clean eating",
    "Mediterranean",
]

INTOLERANCES = [
    "Dairy",
    "Gluten",
    "Soy",
    "Egg",
    "Peanut",
    "Tree Nut",
    "Shellfish",
    "Seafood",
    "Sesame",
    "Sulfite",
    "Wheat",
    "Grain",
]

EQUIPMENT = [
    "Air Fryer",
    "Aluminum Foil",
    "Baking Dish",
    "Baking Pan",
    "Baking Sheet",
    "Baster",
    "Blender",
    "Blow Torch",
    "Box Grater",
    "Bread Knife",
    "Bread Machine",
    "Broiler",
    "Broiler Pan",
    "Butter Curler",
    "Can Opener",
    "Canning Jar",
    "Candy Thermometer",
    "Carving Fork",
    "Casserole Dish",
    "Cake Pop Mold",
    "Cheesecloth",
    "Cherry Pitter",
    "Chocolate Mold",
    "Cleaver",
    "Cocktail Sticks",
    "Colander",
    "Cookie Cutter",
    "Corkscrew",
    "Deep Fryer",
    "Dehydrator",
    "Dough Scraper",
    "Double Boiler",
    "Drinking Straws",
    "Dutch Oven",
    "Egg Slicer",
    "Espresso Machine",
    "Filter",
    "Fillet Knife",
    "Food Processor",
    "Frying Pan",
    "Funnel",
    "Garlic Press",
    "Glass Baking Pan",
    "Glass Casserole Dish",
    "Gravy Boat",
    "Grater",
    "Griddle",
    "Grill",
    "Grill Pan",
    "Hand Mixer",
    "Ice Cream Machine",
    "Ice Cream Scoop",
    "Ice Cube Tray",
    "Immersion Blender",
    "Instant Pot",
    "Juicer",
    "Kitchen Scale",
    "Kitchen Scissors",
    "Kitchen Thermometer",
    "Kitchen Timer",
    "Kitchen Towels",
    "Kitchen Twine",
    "Kugelhopf Mold",
    "Loaf Pan",
    "Lollipop Sticks",
    "Madeleine Mold",
    "Mandoline",
    "Measuring Cup",
    "Measuring Spoon",
    "Meat Grinder",
    "Meat Tenderizer",
    "Melon Baller",
    "Microplane",
    "Mincing Knife",
    "Mortar and Pestle",
    "Muffin Liners",
    "Muffin Tray",
    "Mini Muffin Tray",
    "Oven",
    "Offset Spatula",
    "Palette Knife",
    "Panini Press",
    "Pasta Machine",
    "Pastry Bag",
    "Pastry Brush",
    "Pastry Cutter",
    "Peeler",
    "Pepper Grinder",
    "Pie Mold",
    "Pizza Board",
    "Pizza Cutter",
    "Pizza Pan",
    "Pizza Stone",
    "Plastic Wrap",
    "Popcorn Maker",
    "Popsicle Molds",
    "Popsicle Sticks",
    "Pot",
    "Pot Holder",
    "Potato Masher",
    "Potato Ricer",
    "Poultry Shears",
    "Pressure Cooker",
    "Ramekin",
    "Rice Cooker",
    "Roasting Pan",
    "Rolling Pin",
    "Salad Spinner",
    "Saucepan",
    "Serrated Knife",
    "Sieve",
    "Sifter",
    "Skimmer",
    "Slow Cooker",
    "Slotted Spoon",
    "Springform Pan",
    "Stand Mixer",
    "Steamer Basket",
    "Stove",
    "Tajine Pot",
    "Tart Mold",
    "Teapot",
    "Toaster",
    "Toothpicks",
    "Waffle Iron",
    "Wax Paper",
    "Wire Rack",
    "Wok",
    "Wooden Skewers",
    "Zester",
    "Ziploc Bags",
]


@dataclass
class Recipe:
    """
    Represents a complete recipe with metadata, ingredients, instructions, and nutritional information.

    This dataclass captures all information needed to prepare a meal, including timing,
    serving information, dietary restrictions, and nutritional data. Nutritional values
    are stored as floats representing the actual amounts.

    Required fields:
        title: Recipe name
        ingredients: List of ingredients with quantities
        instructions: Step-by-step cooking instructions
        cuisine: Type of cuisine (e.g., "Vietnamese")

    Fields with defaults:
        prep_time: Preparation time in minutes - defaults to 0
        cook_time: Cooking time in minutes - defaults to 0
        total_time: Total time in minutes - defaults to 0
        num_servings: Number of servings yielded - defaults to 0
        rating: Recipe rating as float - defaults to 0
        num_reviews: Number of reviews as int - defaults to 0
        calories: Calorie content as float - defaults to 0
        protein: Protein content as float - defaults to 0
        total_fat: Fat content in grams as float - defaults to 0
        total_carbohydrate: Carbohydrate content in grams as float - defaults to 0
        diet: Dietary tags like ["Gluten free", "Vegetarian"] - defaults to []
        intolerances: Food intolerances this recipe violates, like ["Dairy", "Egg"] - defaults to []
        equipment: Required cooking equipment - defaults to []

    Optional nutritional fields (with defaults):
        saturated_fat: Saturated fat content in grams as float - defaults to None
        cholesterol: Cholesterol content in mg as float - defaults to None
        sodium: Sodium content in mg as float - defaults to None
        total_sugars: Sugar content in grams as float - defaults to None
        vitamin_c: Vitamin C content in mg as float - defaults to None
        calcium: Calcium content in mg as float - defaults to None
        dietary_fiber: Fiber content in grams as float - defaults to None
        iron: Iron content in mg as float - defaults to None
        potassium: Potassium content in mg as float - defaults to None
        food_type: Type of food (e.g., "Dessert") - defaults to None
    """

    # Required fields
    title: str
    ingredients: List[str]
    instructions: List[str]
    cuisine: str
    prep_time: int = 0
    cook_time: int = 0
    total_time: int = 0
    num_servings: int = 0
    rating: float = 0
    num_reviews: int = 0
    calories: float = 0
    protein: float = 0
    total_fat: float = 0
    total_carbohydrate: float = 0
    diet: List[str] = field(default_factory=list)
    intolerances: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)

    # Optional fields (with defaults)
    saturated_fat: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    total_sugars: Optional[float] = None
    vitamin_c: Optional[float] = None
    calcium: Optional[float] = None
    dietary_fiber: Optional[float] = None
    iron: Optional[float] = None
    potassium: Optional[float] = None
    food_type: Optional[str] = None


DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))


class RecipeDB(Database):
    """
    A database of recipes.
    """

    def __init__(self, file_path: str = f"{DATASET_ROOT}/assets/recipes.csv"):
        df = pd.read_csv(file_path)
        df['ingredients'] = df['ingredients'].apply(eval)
        df['diet'] = df['diet'].apply(eval)
        df['intolerances'] = df['intolerances'].apply(eval)
        df['equipment'] = df['equipment'].apply(eval)
        super().__init__(
            {
                "recipes": (
                    "Database of recipes from AllRecipes.com",
                    df,
                    {
                        "title": "Recipe name",
                        "ingredients": "List of ingredients with quantities",
                        "instructions": "Step-by-step cooking instructions",
                        "cuisine": "Type of cuisine (e.g., 'Vietnamese')",
                        "prep_time": "Preparation time in minutes",
                        "cook_time": "Cooking time in minutes",
                        "total_time": "Total time in minutes",
                        "num_servings": "Number of servings yielded",
                        "rating": "Recipe rating as float",
                        "num_reviews": "Number of reviews as int",
                        "calories": "Calorie content as float",
                        "protein": "Protein content as float",
                        "total_fat": "Fat content in grams as float",
                        "total_carbohydrate": "Carbohydrate content in grams as float",
                        "diet": "COMPATIBLE dietary tags like ['Gluten free', 'Vegetarian']",
                        "intolerances": "Food intolerances this recipe is INCOMPATIBLE with, like ['Dairy', 'Egg']",
                        "equipment": "Required cooking equipment",
                        "saturated_fat": "Saturated fat content in grams as float",
                        "cholesterol": "Cholesterol content in mg as float",
                        "sodium": "Sodium content in mg as float",
                        "total_sugars": "Sugar content in grams as float",
                        "vitamin_c": "Vitamin C content in mg as float",
                        "calcium": "Calcium content in mg as float",
                        "dietary_fiber": "Fiber content in grams as float",
                        "iron": "Iron content in mg as float",
                        "potassium": "Potassium content in mg as float",
                        "food_type": "Type of food (e.g., 'Dessert')",
                    },
                )
            }
        )

    def get_recipe_by_name(self, name: str) -> Recipe:
        """
        Get a recipe by name.
        """
        df = self.tables['recipes']
        matches = df[df['title'] == name]
        if len(matches) == 0:
            matches = df[df['title'] == name.title()]
        if len(matches) == 0:
            return None
        d = matches.iloc[0].to_dict()
        return Recipe(**d)

    @cached_property
    def stats(self):
        print("########### Recipe DB Stats ###########")
        print(f"Number of recipes: {len(self.tables['recipes'])}")
        print(
            f"Distribution of cuisines:\n\t{Counter(recipe.cuisine for recipe in self.tables['recipes'].values())}"
        )

        print(f"Number of recipes that are diet-compatible:\n\t{self.diet_counts}")
        print(
            f"Number of recipes that are intolerance-compatible:\n\t{self.intolerance_counts}"
        )
        print(f"Distribution of equipment:\n\t{self.equipment_counts}")
        print(
            f"Distribution of food types:\n\t{Counter(recipe.food_type for recipe in self.tables['recipes'].values())}"
        )

    @cached_property
    def diet_counts(self):
        diet_counts = {
            diet: sum(
                diet in recipe.diet
                for recipe in self.tables['recipes'].values()
                if recipe.diet is not None
            )
            for diet in DIETS
        }
        return diet_counts

    @cached_property
    def intolerance_counts(self):
        intolerance_counts = {
            intolerance: sum(
                intolerance in recipe.intolerances
                for recipe in self.tables['recipes'].values()
                if recipe.intolerances is not None
            )
            for intolerance in INTOLERANCES
        }
        return intolerance_counts

    @cached_property
    def equipment_counts(self):
        equipment_counts = {
            equipment: sum(
                equipment in recipe.equipment
                for recipe in self.tables['recipes'].values()
                if recipe.equipment is not None
            )
            for equipment in EQUIPMENT
        }
        return equipment_counts
