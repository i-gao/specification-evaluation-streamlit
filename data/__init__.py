from data.travel_planner.data import TravelPlannerDataset
from data.meal_planning.data import MealPlanningDataset
from data.workout_planning.data import WorkoutPlanningDataset
from data.design2code.data import Design2CodeDataset
from data.shopping.data import ShoppingDataset
from data.dataset import Specification, FixedSpecification, CustomSpecification


__all__ = ["DATASETS", "get_dataset", "get_spec"]

DATASETS = [
    "travel_planner",
    "meal_planning",
    "workout_planning",
    "design2code",
    "shopping",
]


def get_dataset(dataset_name: str, **kwargs):
    # Dynamically import the requested dataset class
    if dataset_name == "travel_planner":
        return TravelPlannerDataset(**kwargs)
    elif dataset_name == "meal_planning":
        return MealPlanningDataset(**kwargs)
    elif dataset_name == "workout_planning":
        return WorkoutPlanningDataset(**kwargs)
    elif dataset_name == "design2code":
        return Design2CodeDataset(**kwargs)
    elif dataset_name == "shopping":
        return ShoppingDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_spec(dataset_name: str, index: int, **kwargs):
    print("Getting spec for", dataset_name, index, kwargs)
    dataset = get_dataset(dataset_name, **kwargs)
    return dataset[index]
