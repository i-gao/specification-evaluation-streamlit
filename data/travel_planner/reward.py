"""
Reward functions and constraint creation for travel planning.

This module provides Constraint creation functions to convert the existing
reward_utils evaluation functions into the Constraint-based linear reward system
used by LinearFixedSpecification.
"""

from typing import List, Dict, Tuple
from data.reward import Constraint


# ============================================================================
# Constraint Creation Functions
# ============================================================================


def create_hard_constraints(query_data) -> List[dict]:
    """Create individual hard constraint dictionaries from query_data."""
    constraints = []
    days = query_data.get("days", 1)
    
    # Simple single-check constraints
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"The origin city of Day 1 must be {query_data['org']}",
            extractor="check_first_city_is_origin",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description="The trip must be a round trip",
            extractor="check_closed_circle",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description="All cities mentioned in the travel plan must exist in the city-state map database",
            extractor="check_cities_in_map",
            is_hard=True,
            is_discoverable=False,
            is_minimal=False,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    if query_data.get("days", 1) > 3:
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"All intermediate cities (not the first or last) must be in the destination state: {query_data['dest']}",
                extractor="check_intermediate_cities_in_destination_state",
                is_hard=True,
                is_discoverable=False,
                is_minimal=True,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description="Each accommodation must be booked for at least the minimum number of nights required by that accommodation",
            extractor="check_accommodation_minimum_nights",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"The total cost of the trip must be within the budget of ${query_data.get('budget', 0)}",
            extractor="check_budget",
            is_hard=True,
            is_discoverable=True,
            is_minimal=False,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # Transportation constraint (first day)
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description="Day 1 must have transportation specified (cannot be empty or '-')",
            extractor="check_transportation_first_day",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # Days constraint
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"The travel plan must contain exactly {days} day(s) of itinerary",
            extractor="check_valid_days",
            is_hard=True,
            is_discoverable=False,
            is_minimal=False,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # Visiting city number constraint
    visiting_city_num = query_data.get('visiting_city_number', 'N')
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"The trip must visit exactly {visiting_city_num} unique city/cities (excluding the origin city {query_data['org']})",
            extractor="check_visiting_city_number",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # All days and fields constraints for city matching
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"For all {days} day(s), all specified fields (transportation, breakfast, lunch, dinner, attraction, accommodation) must be in the correct city as indicated by the current_city field",
            extractor="check_all_days_fields_in_correct_city",
            is_hard=True,
            is_discoverable=False,
            is_minimal=False,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # All days and fields constraints for sandbox existence
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"For all {days} day(s), all specified fields (transportation, breakfast, lunch, dinner, attraction, accommodation) must exist in the database (flights must have valid flight numbers and dates, restaurants/attractions/accommodations must exist in the specified cities)",
            extractor="check_all_days_fields_exist_in_sandbox",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # All days and fields constraints for presence
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description=f"For all {days} day(s), all required fields must be present: transportation required for travel days, meals required for single-city days, attractions required for single-city days, accommodation required for all days except the last",
            extractor="check_all_days_fields_present",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # Meals match travel time (first and last day)
    constraints.append(
        Constraint.create_boolean_penalize_false_constraint(
            description="Meals on Day 1 and the last day must match travel times: if arriving after 12pm, breakfast should be empty; if arriving after 3pm, lunch should be empty; if arriving after 11pm, dinner should be empty; if departing before 9am, breakfast should be empty; if departing before 12pm, lunch should be empty; if departing before 5pm, dinner should be empty",
            extractor="check_meals_match_travel_time",
            is_hard=True,
            is_discoverable=False,
            is_minimal=True,
            extractor_kwargs={"query_data": query_data},
            none_val=0.0,
        )
    )
    
    # Hard constraints from hard_constraint.py
    local_constraint = query_data.get("local_constraint", {})
    
    # Room rule constraint
    house_rule = local_constraint.get("house rule")
    if house_rule is not None:
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"All accommodations must allow the house rule '{house_rule}' (accommodations with 'No {house_rule}' are not allowed)",
                extractor="check_room_rule",
                is_hard=True,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    # Transportation restriction constraint
    transportation_restriction = local_constraint.get("transportation")
    if transportation_restriction is not None:
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"Transportation must not use '{transportation_restriction}' (e.g., if restriction is 'no flight', flights are not allowed)",
                extractor="check_transportation_restriction",
                is_hard=True,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    # Room type constraint
    room_type_req = local_constraint.get("room_type")
    if room_type_req is not None:
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"All accommodations must have room type '{room_type_req}' (e.g., 'private room', 'shared room', 'entire room', or 'not shared room')",
                extractor="check_room_type",
                is_hard=True,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    # Attractions per single-city day constraint
    min_attractions_single = local_constraint.get("min_attractions_per_single_city_day")
    max_attractions_single = local_constraint.get("max_attractions_per_single_city_day")
    if min_attractions_single is not None or max_attractions_single is not None:
        min_str = f"at least {min_attractions_single}" if min_attractions_single is not None else ""
        max_str = f"at most {max_attractions_single}" if max_attractions_single is not None else ""
        range_str = " and ".join(filter(None, [min_str, max_str]))
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"For all single-city days (non-travel days), the number of attractions must be {range_str}",
                extractor="check_attractions_per_single_city_day",
                is_hard=False,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    # Attractions per travel day constraint
    min_attractions_travel = local_constraint.get("min_attractions_per_travel_day")
    max_attractions_travel = local_constraint.get("max_attractions_per_travel_day")
    if min_attractions_travel is not None or max_attractions_travel is not None:
        min_str = f"at least {min_attractions_travel}" if min_attractions_travel is not None else ""
        max_str = f"at most {max_attractions_travel}" if max_attractions_travel is not None else ""
        range_str = " and ".join(filter(None, [min_str, max_str]))
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"For all travel days (days with 'from X to Y' in current_city), the number of attractions must be {range_str}",
                extractor="check_attractions_per_travel_day",
                is_hard=False,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
    
    return constraints


def create_soft_constraints(
    query_data, 
    preference_weights: Dict[str, float],
) -> Tuple[List[dict], List[float]]:
    """Create individual soft constraint dictionaries using reward_any_in_set and penalize_any_in_set."""
    constraints = []
    weights = []
    preferences = query_data.get("preferences", {})
    
    # Restaurant tags (liked and disliked) - one constraint per tag
    weight = preference_weights.get("tags", 0) or 0
    if weight > 0:
        liked_tags = preferences.get("liked_tags", [])
        disliked_tags = preferences.get("disliked_tags", [])
        
        # One constraint per liked tag
        for tag in liked_tags:
            if tag:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[tag],
                        description=f"Prefer restaurants that have tag: {tag}",
                        extractor="extract_all_restaurant_tags",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
        
        # One constraint per disliked tag
        for tag in disliked_tags:
            if tag:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[tag],
                        description=f"Prefer restaurants that do not have tag: {tag}",
                        extractor="extract_all_restaurant_tags",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Room types (liked and disliked) - one constraint per type
    weight = preference_weights.get("room_types", 0) or 0
    if weight > 0:
        liked_room_types = preferences.get("liked_room_types", [])
        disliked_room_types = preferences.get("disliked_room_types", [])
        
        # One constraint per liked room type
        for room_type in liked_room_types:
            if room_type:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[room_type],
                        description=f"Prefer accommodations with room type: {room_type}",
                        extractor="extract_all_room_types",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
        
        # One constraint per disliked room type
        for room_type in disliked_room_types:
            if room_type:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[room_type],
                        description=f"Prefer accommodations that do not have room type: {room_type}",
                        extractor="extract_all_room_types",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Attraction types (liked and disliked) - one constraint per type
    weight = preference_weights.get("attraction_types", 0) or 0
    if weight > 0:
        liked_attraction_types = preferences.get("liked_attraction_types", [])
        disliked_attraction_types = preferences.get("disliked_attraction_types", [])
        
        # One constraint per liked attraction type
        for attraction_type in liked_attraction_types:
            if attraction_type:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[attraction_type],
                        description=f"Prefer attractions with type: {attraction_type}",
                        extractor="extract_all_attraction_types",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
        
        # One constraint per disliked attraction type
        for attraction_type in disliked_attraction_types:
            if attraction_type:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[attraction_type],
                        description=f"Prefer attractions that do not have type: {attraction_type}",
                        extractor="extract_all_attraction_types",
                        is_hard=False,
                        is_discoverable=True,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific liked restaurants (one constraint per restaurant)
    weight = preference_weights.get("specific_liked_restaurants", 0) or 0
    if weight > 0:
        specific_liked = preferences.get("specific_liked_restaurants", [])
        for restaurant_name in specific_liked:
            if restaurant_name:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[restaurant_name],
                        description=f"Prefer restaurant: {restaurant_name}",
                        extractor="extract_all_restaurant_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific disliked restaurants (one constraint per restaurant)
    weight = preference_weights.get("specific_disliked_restaurants", 0) or 0
    if weight > 0:
        specific_disliked = preferences.get("specific_disliked_restaurants", [])
        for restaurant_name in specific_disliked:
            if restaurant_name:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[restaurant_name],
                        description=f"Dislike specific restaurant: {restaurant_name}",
                        extractor="extract_all_restaurant_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific liked accommodations (one constraint per accommodation)
    weight = preference_weights.get("specific_liked_accommodations", 0) or 0
    if weight > 0:
        specific_liked = preferences.get("specific_liked_accommodations", [])
        for accommodation_name in specific_liked:
            if accommodation_name:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[accommodation_name],
                        description=f"Prefer accommodation: {accommodation_name}",
                        extractor="extract_all_accommodation_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific disliked accommodations (one constraint per accommodation)
    weight = preference_weights.get("specific_disliked_accommodations", 0) or 0
    if weight > 0:
        specific_disliked = preferences.get("specific_disliked_accommodations", [])
        for accommodation_name in specific_disliked:
            if accommodation_name:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[accommodation_name],
                        description=f"Dislike specific accommodation: {accommodation_name}",
                        extractor="extract_all_accommodation_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific liked attractions (one constraint per attraction)
    weight = preference_weights.get("specific_liked_attractions", 0) or 0
    if weight > 0:
        specific_liked = preferences.get("specific_liked_attractions", [])
        for attraction_name in specific_liked:
            if attraction_name:
                constraints.append(
                    Constraint.create_reward_any_in_set_constraint(
                        good_set=[attraction_name],
                        description=f"Prefer attraction: {attraction_name}",
                        extractor="extract_all_attraction_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Specific disliked attractions (one constraint per attraction)
    weight = preference_weights.get("specific_disliked_attractions", 0) or 0
    if weight > 0:
        specific_disliked = preferences.get("specific_disliked_attractions", [])
        for attraction_name in specific_disliked:
            if attraction_name:
                constraints.append(
                    Constraint.create_penalize_any_in_set_constraint(
                        bad_set=[attraction_name],
                        description=f"Dislike specific attraction: {attraction_name}",
                        extractor="extract_all_attraction_names",
                        is_hard=False,
                        is_discoverable=True,
                        is_discoverable_by_questions=False,
                        is_minimal=False,
                        extractor_kwargs={"query_data": query_data},
                        none_val=0.0,
                    )
                )
                weights.append(weight)
    
    # Restaurant attributes (liked and disliked)
    weight = preference_weights.get("restaurant_attributes", 0) or 0
    if weight > 0:
        liked_attributes = preferences.get("liked_attributes", [])
        disliked_attributes = preferences.get("disliked_attributes", [])
        
        if liked_attributes and len(liked_attributes) > 0:
            constraints.append(
                Constraint.create_reward_any_in_set_constraint(
                    good_set=list(liked_attributes),
                    description=f"Prefer restaurants with these attributes: {', '.join(liked_attributes)}",
                    extractor="extract_all_restaurant_attributes",
                    is_hard=False,
                    is_discoverable=True,
                    is_minimal=False,
                    extractor_kwargs={"query_data": query_data},
                    none_val=0.0,
                )
            )
            weights.append(weight)
        
        if disliked_attributes and len(disliked_attributes) > 0:
            constraints.append(
                Constraint.create_penalize_any_in_set_constraint(
                    bad_set=list(disliked_attributes),
                    description=f"Prefer restaurants that do not have these attributes: {', '.join(disliked_attributes)}",
                    extractor="extract_all_restaurant_attributes",
                    is_hard=False,
                    is_discoverable=True,
                    is_minimal=False,
                    extractor_kwargs={"query_data": query_data},
                    none_val=0.0,
                )
            )
            weights.append(weight)
    
    # Restaurant ratings
    weight = preference_weights.get("restaurant_ratings", 0) or 0
    if weight > 0:
        min_rating = preferences.get("min_rating_restaurants")
        if min_rating is not None:
            constraints.append(
                Constraint.create_penalize_any_in_set_constraint(
                    bad_set=[True],
                    description=f"Restaurant ratings should be at least {min_rating}",
                    extractor="extract_restaurant_rating_below_minimum_flags",
                    is_hard=False,
                    is_discoverable=True,
                    is_minimal=False,
                    extractor_kwargs={"query_data": query_data},
                    none_val=0.0,
                )
            )
            weights.append(weight)
    
    # Accommodation reviews
    weight = preference_weights.get("accommodation_reviews", 0) or 0
    if weight > 0:
        min_reviews = preferences.get("min_num_ratings_accommodations")
        if min_reviews is not None:
            constraints.append(
                Constraint.create_penalize_any_in_set_constraint(
                    bad_set=[True],
                    description=f"Accommodations should have at least {min_reviews} reviews",
                    extractor="extract_accommodation_reviews_below_minimum_flags",
                    is_hard=False,
                    is_discoverable=True,
                    is_minimal=False,
                    extractor_kwargs={"query_data": query_data},
                    none_val=0.0,
                )
            )
            weights.append(weight)
    
    # Restaurant repeats
    weight = preference_weights.get("restaurant_repeats", 0) or 0
    if weight > 0:
        constraints.append(
            Constraint.create_penalize_any_in_set_constraint(
                bad_set=[True],
                description="Ideally, no restaurant should be repeated across meals",
                extractor="extract_restaurant_repeat_flags",
                is_hard=False,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
        weights.append(weight)
    
    # Attraction repeats
    weight = preference_weights.get("attraction_repeats", 0) or 0
    if weight > 0:
        constraints.append(
            Constraint.create_penalize_any_in_set_constraint(
                bad_set=[True],
                description="Ideally, no attraction should be repeated across days",
                extractor="extract_attraction_repeat_flags",
                is_hard=False,
                is_discoverable=True,
                is_minimal=False,
                extractor_kwargs={"query_data": query_data},
                none_val=0.0,
            )
        )
        weights.append(weight)
    
    return constraints, weights
