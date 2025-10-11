import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import os
import shutil
import dill
import pickle
from PIL import Image, ImageDraw
from copy import deepcopy
from collections import Counter

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from clip import load as clip_load

# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================


def _prepare_temp_files(test_id: int, testset_dir: str, temp_dir: str):
    """
    First step of the evaluation process: create a temp dir and save the original HTML, image, and rick.jpg to it
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Load original HTML and rick.jpg, and then save them to a temp dir
    original_html_path = os.path.join(testset_dir, f"{test_id}.html")
    original_img_path = os.path.join(testset_dir, f"{test_id}.png")
    original_blocks_path = os.path.join(testset_dir, f"{test_id}_blocks.pkl")

    if not os.path.exists(original_html_path):
        raise FileNotFoundError(f"Original HTML file not found: {original_html_path}")
    if not os.path.exists(original_img_path):
        raise FileNotFoundError(f"Original image file not found: {original_img_path}")
    if not os.path.exists(original_blocks_path):
        raise FileNotFoundError(f"Original blocks file not found: {original_blocks_path}")

    temp_html_path = os.path.join(temp_dir, f"{test_id}.html")
    temp_img_path = os.path.join(temp_dir, f"{test_id}.png")
    temp_blocks_path = os.path.join(temp_dir, f"{test_id}_blocks.pkl")
    shutil.copy2(original_html_path, temp_html_path)
    shutil.copy2(original_img_path, temp_img_path)
    shutil.copy2(original_blocks_path, temp_blocks_path)
    shutil.copy2(
        os.path.join(testset_dir, "rick.jpg"), os.path.join(temp_dir, "rick.jpg")
    )
    return temp_html_path, temp_img_path, temp_blocks_path


def visual_eval_v3_multi(
    predicted_html: str,
    test_id: int,
    debug: bool = False,
    testset_dir: str = "testset_final",
) -> dict:
    """
    Main evaluation function that compares predicted HTML content against reference HTML content.

    Args:
        predicted_html: HTML content as a string
        test_id: Integer ID corresponding to the test case (e.g., 1 for 1.html and 1.png)
        debug: Whether to print debug information

    Returns:
        dict: Dictionary containing evaluation results with detailed matching information:
            - total_area (float): Total area of all blocks (matched + unmatched) in normalized coordinates
            - combined_score (float): Overall evaluation score (0-1) combining all metrics with equal weights
            - size_score (float): Area coverage score (0-1) measuring how well predicted blocks cover reference areas
            - text_score (float): Text similarity score (0-1) measuring text content accuracy
            - position_score (float): Position accuracy score (0-1) measuring spatial layout similarity
            - color_score (float): Color similarity score (0-1) measuring text color accuracy using CIEDE2000
            - clip_score (float): Semantic similarity score (0-1) measuring overall visual similarity using CLIP
            - predicted_blocks (list): List of text blocks extracted from the predicted HTML screenshot
            - visualization (dict): Visualization data containing images with bounding boxes drawn:
                - predicted_image_with_boxes (numpy.ndarray): Predicted image with matched bounding boxes
                - original_image_with_boxes (numpy.ndarray): Reference image with matched bounding boxes
            - original_blocks (list): Detailed information for each original block:
                - block_index (int): Index of the block in the original sequence
                - text (str): Text content of the block
                - bbox (tuple): Bounding box as (x_ratio, y_ratio, w_ratio, h_ratio) in normalized coordinates
                - color (tuple): RGB color values of the text
                - matched (bool): Whether this block has a matching predicted block
                - predicted_block_index (int): Index of the matched predicted block (if matched)
                - predicted_block (dict): The matched predicted block data (if matched)
                - text_similarity (float): Text similarity score with matched block (if matched)
                - position_similarity (float): Position similarity score with matched block (if matched)
                - color_similarity (float): Color similarity score with matched block (if matched)
    """

    from difflib import SequenceMatcher
    import cv2
    from ocr_free_utils import get_blocks_ocr_free
    from clean_predicted_html import clean_html


    temp_dir = os.path.join(os.getcwd(), "temp_files")
    original_html_path, original_img_path, original_blocks_path = _prepare_temp_files(
        test_id, testset_dir, temp_dir
    )

    # Save predicted HTML to temp dir
    predicted_html = clean_html(predicted_html)
    predicted_html_path = os.path.join(
        temp_dir, f"predicted_{os.getpid()}_{test_id}.html"
    )
    with open(predicted_html_path, "w") as f:
        f.write(predicted_html)

    # Generate a screenshot of the predicted HTML file using an external script
    predict_img = predicted_html_path.replace(".html", ".png")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    screenshot_script = os.path.join(script_dir, "screenshot_single.py")
    os.system(
        f"python3 {screenshot_script} --html {predicted_html_path} --png {predict_img}"
    )

    ###########################################

    # Extract text blocks from the predicted screenshot using OCR-free method
    predict_blocks = get_blocks_ocr_free(predict_img)

    # Load original blocks
    with open(original_blocks_path, "rb") as f:
        original_blocks = pickle.load(f)

    # Parameters for block matching algorithm
    consecutive_bonus, window_size = 0.1, 1

    # Handle case where no blocks were detected in the prediction
    if len(predict_blocks) == 0:
        print("[Warning] No detected blocks in: ", predict_img)
        # Calculate only CLIP similarity as fallback
        remaining_elements_clip_score = calculate_clip_similarity_without_blocks(
            predict_img, original_img_path, predict_blocks, original_blocks
        )
        return {
            "scores": [
                0.0,
                0.2 * remaining_elements_clip_score,
                (0.0, 0.0, 0.0, 0.0, remaining_elements_clip_score),
            ],
            "original_blocks": original_blocks,
            "predicted_blocks": predict_blocks,
            "visualization": None,
            "original_block_details": [],
        }

    # Handle case where no blocks were detected in the reference
    elif len(original_blocks) == 0:
        print("[Warning] No detected blocks in: ", original_img_path)
        remaining_elements_clip_score = calculate_clip_similarity_without_blocks(
            predict_img, original_img_path, predict_blocks, original_blocks
        )
        return {
            "scores": [
                0.0,
                0.2 * remaining_elements_clip_score,
                (0.0, 0.0, 0.0, 0.0, remaining_elements_clip_score),
            ],
            "original_blocks": original_blocks,
            "predicted_blocks": predict_blocks,
            "visualization": None,
            "original_block_details": [],
        }

    # Merge overlapping blocks in the prediction
    predict_blocks = merge_blocks_by_bbox(predict_blocks)

    # Find optimal matching between prediction and reference blocks
    # This includes merging blocks if it improves the overall matching
    predict_blocks_m, original_blocks_m, matching = find_possible_merge(
        predict_blocks,
        deepcopy(original_blocks),
        consecutive_bonus,
        window_size,
        debug=debug,
    )

    # Filter out low-quality matches based on text similarity
    filtered_matching = []
    for i, j in matching:
        text_similarity = SequenceMatcher(
            None, predict_blocks_m[i]["text"], original_blocks_m[j]["text"]
        ).ratio()
        # Only keep matches with at least 50% text similarity
        if text_similarity < 0.5:
            continue
        filtered_matching.append([i, j, text_similarity])
    matching = filtered_matching

    # Extract indices of matched blocks
    indices1 = [item[0] for item in matching]  # Prediction block indices
    indices2 = [item[1] for item in matching]  # Reference block indices

    # Create a mapping from original block index to matching information
    original_block_mapping = {}
    for i, j, text_similarity in matching:
        original_block_mapping[j] = {
            "matched": True,
            "predicted_block_index": i,
            "predicted_block": predict_blocks_m[i],
            "text_similarity": text_similarity,
        }

    #####################################################

    # Initialize lists to store various similarity scores
    matched_list = []
    sum_areas = []
    matched_areas = []
    matched_text_scores = []
    position_scores = []
    text_color_scores = []

    # Calculate area of unmatched blocks in prediction
    unmatched_area_1 = 0.0
    for i in range(len(predict_blocks_m)):
        if i not in indices1:
            unmatched_area_1 += (
                predict_blocks_m[i]["bbox"][2] * predict_blocks_m[i]["bbox"][3]
            )

    # Calculate area of unmatched blocks in reference
    unmatched_area_2 = 0.0
    for j in range(len(original_blocks_m)):
        if j not in indices2:
            unmatched_area_2 += (
                original_blocks_m[j]["bbox"][2] * original_blocks_m[j]["bbox"][3]
            )

    # Total area of unmatched blocks (used for size score calculation)
    sum_areas.append(unmatched_area_1 + unmatched_area_2)

    # Calculate scores for each matched block pair
    for i, j, text_similarity in matching:
        # Calculate combined area of the matched block pair
        sum_block_area = (
            predict_blocks_m[i]["bbox"][2] * predict_blocks_m[i]["bbox"][3]
            + original_blocks_m[j]["bbox"][2] * original_blocks_m[j]["bbox"][3]
        )

        # Calculate position similarity based on center points of blocks
        # Uses maximum distance (either horizontal or vertical) between centers
        position_similarity = 1 - calculate_distance_max_1d(
            predict_blocks_m[i]["bbox"][0]
            + predict_blocks_m[i]["bbox"][2] / 2,  # Pred center x
            predict_blocks_m[i]["bbox"][1]
            + predict_blocks_m[i]["bbox"][3] / 2,  # Pred center y
            original_blocks_m[j]["bbox"][0]
            + original_blocks_m[j]["bbox"][2] / 2,  # Ref center x
            original_blocks_m[j]["bbox"][1]
            + original_blocks_m[j]["bbox"][3] / 2,  # Ref center y
        )

        # Calculate color similarity using CIEDE2000 color difference formula
        text_color_similarity = color_similarity_ciede2000(
            predict_blocks_m[i]["color"], original_blocks_m[j]["color"]
        )

        # Store the matched bounding boxes for visualization
        matched_list.append([predict_blocks_m[i]["bbox"], original_blocks_m[j]["bbox"]])

        # Update the original block mapping with calculated scores
        original_block_mapping[j].update(
            {
                "position_similarity": position_similarity,
                "color_similarity": text_color_similarity,
            }
        )

        # Validation: ensure blocks have non-zero dimensions
        if (
            min(
                predict_blocks_m[i]["bbox"][2],
                original_blocks_m[j]["bbox"][2],
                predict_blocks_m[i]["bbox"][3],
                original_blocks_m[j]["bbox"][3],
            )
            == 0
        ):
            print(f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}")
        assert (
            calculate_ratio(
                predict_blocks_m[i]["bbox"][2], original_blocks_m[j]["bbox"][2]
            )
            > 0
            and calculate_ratio(
                predict_blocks_m[i]["bbox"][3], original_blocks_m[j]["bbox"][3]
            )
            > 0
        ), f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}"

        # Store all the calculated scores
        sum_areas.append(sum_block_area)
        matched_areas.append(sum_block_area)
        matched_text_scores.append(text_similarity)
        position_scores.append(position_similarity)
        text_color_scores.append(text_color_similarity)

    # Generate visualization
    visualization = None
    img1 = cv2.imread(predict_img)
    img2 = cv2.imread(original_img_path)
    img1_with_boxes, img2_with_boxes = draw_matched_bboxes(img1, img2, matched_list)
    visualization = {
        "predicted_image_with_boxes": img1_with_boxes,
        "original_image_with_boxes": img2_with_boxes,
    }

    # Calculate final scores if there are matched blocks
    if len(matched_areas) > 0:
        total_area = np.sum(sum_areas)

        # Calculate individual component scores:
        final_size_score = np.sum(matched_areas) / np.sum(sum_areas)  # Area coverage
        final_matched_text_score = np.mean(matched_text_scores)  # Text similarity
        final_position_score = np.mean(position_scores)  # Position accuracy
        final_text_color_score = np.mean(text_color_scores)  # Color accuracy

        # Calculate CLIP similarity (semantic similarity of the images)
        remaining_elements_clip_score = calculate_clip_similarity_without_blocks(
            predict_img, original_img_path, predict_blocks, original_blocks
        )

        # Combine all scores with equal weights (0.2 each)
        final_score = 0.2 * (
            final_size_score
            + final_matched_text_score
            + final_position_score
            + final_text_color_score
            + remaining_elements_clip_score
        )

        # Create detailed information for each original block
        original_block_details = []
        for j, original_block in enumerate(original_blocks_m):
            match_info = original_block_mapping.get(j, {})
            original_block_details.append(
                {
                    "block_index": j,
                    **original_block,
                    **match_info,
                }
            )
    else:
        # No matched blocks found - use only CLIP score as fallback
        print("[Warning] No matched blocks in: ", predict_img)
        remaining_elements_clip_score = calculate_clip_similarity_without_blocks(
            predict_img, original_img_path, predict_blocks, original_blocks
        )
        total_area = 0.0
        final_score = 0.2 * remaining_elements_clip_score
        final_size_score = 0.0
        final_matched_text_score = 0.0
        final_position_score = 0.0
        final_text_color_score = 0.0
        original_block_details = original_blocks_m

    # Clean up temporary files
    if not debug:
        try:
            os.remove(predicted_html_path)
            os.remove(predict_img)
            os.remove(original_html_path)
            os.remove(original_img_path)
            os.remove(original_blocks_path)
        except:
            pass  # Ignore cleanup errors

    result = {
        "predicted_blocks": predict_blocks_m,
        "original_blocks": original_block_details,
        "total_area": total_area,
        "final_score": final_score,
        "size_score": final_size_score,
        "text_score": final_matched_text_score,
        "position_score": final_position_score,
        "color_score": final_text_color_score,
        "remaining_elements_clip_score": remaining_elements_clip_score,
        "visualization": visualization,
    }
    if debug:
        dill.dump(result, open(os.path.join(temp_dir, f"{test_id}_result.pkl"), "wb"))
    return result


# ============================================================================
# BLOCK MATCHING AND OPTIMIZATION FUNCTIONS
# ============================================================================


def find_possible_merge(A, B, consecutive_bonus, window_size, debug=False):
    """
    Find optimal block merges to improve matching between sequences A and B.

    This function iteratively tries merging consecutive blocks in both sequences
    to find configurations that improve the overall matching quality. It uses
    a greedy approach, always choosing the merge that provides the best improvement.

    Args:
        A: List of blocks from the prediction
        B: List of blocks from the reference
        consecutive_bonus: Bonus parameter for context adjustment
        window_size: Window size for context adjustment
        debug: Whether to print debug information

    Returns:
        tuple: (optimized_A, optimized_B, final_matching)
            - optimized_A: Optimized version of sequence A
            - optimized_B: Optimized version of sequence B
            - final_matching: Optimal matching between optimized sequences
    """
    merge_bonus = 0.0
    merge_windows = 1

    def sortFn(value):
        return value[2]  # Sort by improvement score

    while True:
        A_changed = False
        B_changed = False

        # Find current optimal matching
        matching, current_cost, cost_matrix = find_maximum_matching(
            A, B, merge_bonus, merge_windows
        )
        if debug:
            print("Current cost of the solution:", current_cost)

        # Try merging consecutive blocks in sequence A
        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                # Create a copy and try merging blocks i and i+1
                new_A = deepcopy(A)
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)

                # Check if this merge improves the matching
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(
                    new_A, B, merge_bonus, merge_windows
                )
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:  # Only merge if improvement is significant
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_A[i]["text"], diff)

            # Sort merges by improvement score and apply the best ones
            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(
                    A, B, merge_bonus, merge_windows
                )
                if debug:
                    print("Cost after optimization A:", current_cost)

        # Try merging consecutive blocks in sequence B
        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                # Create a copy and try merging blocks i and i+1
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)

                # Check if this merge improves the matching
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(
                    A, new_B, merge_bonus, merge_windows
                )
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:  # Only merge if improvement is significant
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_B[i]["text"], diff)

            # Sort merges by improvement score and apply the best ones
            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(
                    A, B, merge_bonus, merge_windows
                )
                if debug:
                    print("Cost after optimization B:", current_cost)

        # Stop if no improvements were made in this iteration
        if not A_changed and not B_changed:
            break

    # Return final optimized sequences and matching
    matching, _, _ = find_maximum_matching(A, B, consecutive_bonus, window_size)
    return A, B, matching


def find_maximum_matching(A, B, consecutive_bonus, window_size):
    """
    Find the optimal matching between blocks from sequence A and sequence B.

    Uses the Hungarian algorithm (linear_sum_assignment) to find the minimum cost
    matching, with context-based cost adjustments to favor consecutive matches.

    Args:
        A: List of blocks from the prediction
        B: List of blocks from the reference
        consecutive_bonus: Bonus parameter for context adjustment
        window_size: Window size for context adjustment

    Returns:
        tuple: (matching_indices, total_cost, cost_matrix)
            - matching_indices: List of (i, j) pairs indicating matched blocks
            - total_cost: Sum of costs for the optimal matching
            - cost_matrix: The adjusted cost matrix used for matching
    """
    cost_matrix = create_cost_matrix(A, B)
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_cost = calculate_current_cost(cost_matrix, row_ind, col_ind)
    return list(zip(row_ind, col_ind)), current_cost, cost_matrix


def create_cost_matrix(A, B):
    """
    Create a cost matrix for matching blocks from sequence A to sequence B.

    The cost is the negative of the similarity, so lower (more negative) costs
    indicate better matches.

    Args:
        A: List of blocks from the prediction
        B: List of blocks from the reference

    Returns:
        numpy.ndarray: Cost matrix where cost[i,j] = -similarity(A[i], B[j])
    """
    n = len(A)
    m = len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = -calculate_similarity(A[i], B[j])
    return cost_matrix


def adjust_cost_for_context(cost_matrix, consecutive_bonus=1.0, window_size=20):
    """
    Adjust the cost matrix to favor consecutive matches by adding bonuses based on nearby costs.

    This function encourages matching blocks that are close to each other in both sequences,
    which helps maintain the spatial layout consistency.

    Args:
        cost_matrix: 2D numpy array of costs between prediction and reference blocks
        consecutive_bonus: Multiplier for the bonus calculation
        window_size: Size of the window around each position to consider for bonus

    Returns:
        numpy.ndarray: Adjusted cost matrix with context-based bonuses
    """
    if window_size <= 0:
        return cost_matrix

    n, m = cost_matrix.shape
    adjusted_cost_matrix = np.copy(cost_matrix)

    for i in range(n):
        for j in range(m):
            bonus = 0
            # Skip if the current cost is already good (less negative than -0.5)
            if adjusted_cost_matrix[i][j] >= -0.5:
                continue
            # Get a window of nearby costs around position (i,j)
            nearby_matrix = cost_matrix[
                max(0, i - window_size) : min(n, i + window_size + 1),
                max(0, j - window_size) : min(m, j + window_size + 1),
            ]
            flattened_array = nearby_matrix.flatten()
            sorted_array = np.sort(flattened_array)[::-1]
            # Remove the current cost from consideration
            sorted_array = np.delete(
                sorted_array, np.where(sorted_array == cost_matrix[i, j])[0][0]
            )
            # Take the worst (most negative) costs from the window
            top_k_elements = sorted_array[-window_size * 2 :]
            sum_top_k = np.sum(top_k_elements)
            # Add bonus proportional to the sum of nearby bad costs
            bonus = consecutive_bonus * sum_top_k
            adjusted_cost_matrix[i][j] += bonus
    return adjusted_cost_matrix


def difference_of_means(list1, list2):
    """
    Calculate the difference of means between two lists after removing common elements.

    This function is used to determine if merging blocks improves the overall
    matching quality by comparing the cost distributions.

    Args:
        list1: First list of numerical values
        list2: Second list of numerical values

    Returns:
        float: Difference between means of unique elements (list1_mean - list2_mean)
    """
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    # Remove common elements from both lists
    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    # Get remaining unique elements
    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    # Calculate means of remaining elements
    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    # Return difference with additional validation
    if mean_list1 - mean_list2 > 0:
        if min(unique_list1) > min(unique_list2):
            return mean_list1 - mean_list2
        else:
            return 0.0
    else:
        return mean_list1 - mean_list2


# ============================================================================
# BLOCK MANIPULATION FUNCTIONS
# ============================================================================


def merge_blocks_by_bbox(blocks):
    """
    Merge blocks that have identical bounding boxes.

    This function combines text blocks that occupy the same spatial location,
    which can happen when multiple text elements are detected in the same area.

    Args:
        blocks: List of text blocks with 'text', 'bbox', and 'color' fields

    Returns:
        list: List of blocks with overlapping bounding boxes merged
    """
    merged_blocks = {}

    # Traverse and merge blocks with identical bounding boxes
    for block in blocks:
        bbox = tuple(block["bbox"])  # Convert bbox to tuple for hashability
        if bbox in merged_blocks:
            # Merge with existing block at the same location
            existing_block = merged_blocks[bbox]
            existing_block["text"] += " " + block["text"]
            existing_block["color"] = [
                (ec + c) / 2 for ec, c in zip(existing_block["color"], block["color"])
            ]
        else:
            # Add new block
            merged_blocks[bbox] = block

    return list(merged_blocks.values())


def merge_blocks_wo_check(block1, block2):
    """
    Merge two text blocks into a single block without validation checks.

    This function combines the text, bounding box, and color information from
    two blocks into one. Used during the block matching optimization process.

    Args:
        block1: First text block with 'text', 'bbox', and 'color' fields
        block2: Second text block with 'text', 'bbox', and 'color' fields

    Returns:
        dict: Merged block with combined text, bounding box, and averaged color
    """
    # Concatenate text with a space separator
    merged_text = block1["text"] + " " + block2["text"]

    # Calculate the bounding box that encompasses both blocks
    x_min = min(block1["bbox"][0], block2["bbox"][0])
    y_min = min(block1["bbox"][1], block2["bbox"][1])
    x_max = max(
        block1["bbox"][0] + block1["bbox"][2], block2["bbox"][0] + block2["bbox"][2]
    )
    y_max = max(
        block1["bbox"][1] + block1["bbox"][3], block2["bbox"][1] + block2["bbox"][3]
    )
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Average the colors of both blocks
    merged_color = tuple(
        (color1 + color2) // 2
        for color1, color2 in zip(block1["color"], block2["color"])
    )

    return {"text": merged_text, "bbox": merged_bbox, "color": merged_color}


def merge_blocks_by_list(blocks, merge_list):
    """
    Merge blocks according to a list of merge operations.

    This function performs multiple merge operations on a list of blocks,
    handling conflicts where blocks might be involved in multiple merges.

    Args:
        blocks: List of text blocks to merge
        merge_list: List of [i, j] pairs indicating which blocks to merge

    Returns:
        list: Modified list of blocks after all merges are applied
    """
    pop_list = []
    while True:
        if len(merge_list) == 0:
            remove_indices(blocks, pop_list)
            return blocks

        i = merge_list[0][0]
        j = merge_list[0][1]

        # Merge blocks i and j, storing result in position i
        blocks[i] = merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)  # Mark block j for removal

        merge_list.pop(0)  # Remove the processed merge operation
        if len(merge_list) > 0:
            # Update remaining merge operations to account for removed block j
            new_merge_list = []
            for k in range(len(merge_list)):
                # Keep merge operations that don't involve the merged blocks
                if (
                    merge_list[k][0] != i
                    and merge_list[k][1] != i
                    and merge_list[k][0] != j
                    and merge_list[k][1] != j
                ):
                    new_merge_list.append(merge_list[k])
            merge_list = new_merge_list


def remove_indices(lst, indices):
    """
    Remove elements from a list at specified indices.

    Indices are processed in reverse order to avoid index shifting issues
    when removing multiple elements.

    Args:
        lst: List to modify
        indices: List of indices to remove

    Returns:
        list: Modified list with specified indices removed
    """
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst


# ============================================================================
# SIMILARITY CALCULATION FUNCTIONS
# ============================================================================


def calculate_similarity(block1, block2, max_distance=1.42):
    """
    Calculate text similarity between two text blocks.

    Args:
        block1: First text block dictionary containing 'text' field
        block2: Second text block dictionary containing 'text' field
        max_distance: Maximum distance threshold (unused in current implementation)

    Returns:
        float: Text similarity ratio between 0 and 1
    """
    from difflib import SequenceMatcher

    text_similarity = SequenceMatcher(None, block1["text"], block2["text"]).ratio()
    return text_similarity


def calculate_distance_max_1d(x1, y1, x2, y2):
    """
    Calculate the maximum distance between two 2D points in either x or y direction.

    This is used for position similarity calculation, where we care about the
    maximum deviation in either horizontal or vertical direction.

    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point

    Returns:
        float: Maximum of absolute differences in x and y directions
    """
    distance = max(abs(x2 - x1), abs(y2 - y1))
    return distance


def calculate_ratio(h1, h2):
    """
    Calculate the ratio between two values, always returning a value >= 1.

    This is used to validate that block dimensions are reasonable when comparing
    predicted and reference blocks.

    Args:
        h1: First value
        h2: Second value

    Returns:
        float: Ratio of larger value to smaller value (always >= 1)
    """
    return max(h1, h2) / min(h1, h2)


# ============================================================================
# COLOR PROCESSING FUNCTIONS
# ============================================================================


def color_similarity_ciede2000(rgb1, rgb2):
    """
    Calculate the color similarity between two RGB colors using the CIEDE2000 formula.

    CIEDE2000 is a color difference formula that provides perceptually uniform
    color differences. This function converts it to a similarity score.

    Args:
        rgb1: First RGB color tuple (R, G, B) in range [0, 255]
        rgb2: Second RGB color tuple (R, G, B) in range [0, 255]

    Returns:
        float: Similarity score between 0 and 1, where 1 means identical colors
    """
    # Convert RGB colors to Lab color space for better difference calculation
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    # Calculate the Delta E using CIEDE2000 formula
    delta_e = delta_e_cie2000(lab1, lab2)

    # Normalize the Delta E value to get a similarity score
    # Note: The normalization method here is arbitrary and can be adjusted based on your needs.
    # A delta_e of 0 means identical colors. Higher values indicate more difference.
    # For visualization purposes, we consider a delta_e of 100 to be completely different.
    similarity = max(0, 1 - (delta_e / 100))

    return similarity


def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.

    Lab color space is perceptually uniform, making it better for color difference
    calculations than RGB.

    Args:
        rgb: Tuple of (R, G, B) values in range [0, 255]

    Returns:
        LabColor: Color in Lab color space
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)

    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)

    return lab_color


# ============================================================================
# IMAGE PROCESSING FUNCTIONS (FOR CLIP SIMILARITY)
# ============================================================================


def calculate_clip_similarity_without_blocks(
    image_path1, image_path2, blocks1, blocks2
):
    """
    Calculate CLIP similarity between two images with text blocks removed.

    This function computes semantic similarity between images by removing text
    content and using CLIP to compare the remaining visual elements.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        blocks1: Text blocks to remove from first image
        blocks2: Text blocks to remove from second image

    Returns:
        float: CLIP similarity score between 0 and 1
    """

    import numpy as np
    
    # Load and preprocess images with text blocks removed
    image1 = (
        preprocess(rescale_and_mask(image_path1, [block["bbox"] for block in blocks1]))
        .unsqueeze(0)
        .to(device)
    )
    image2 = (
        preprocess(rescale_and_mask(image_path2, [block["bbox"] for block in blocks2]))
        .unsqueeze(0)
        .to(device)
    )

    # Calculate CLIP features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize features for cosine similarity
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features1 @ image_features2.T).item()

    return similarity


def rescale_and_mask(image_path, blocks):
    """
    Load an image, mask out text blocks, and resize to a square format.

    This function prepares images for CLIP processing by removing text content
    and ensuring consistent square dimensions.

    Args:
        image_path: Path to the image file
        blocks: List of text blocks with bounding boxes to mask out

    Returns:
        PIL.Image: Processed image with text removed and resized to square
    """
    # Load the image
    with Image.open(image_path) as img:
        if len(blocks) > 0:
            # Use inpainting to remove text blocks
            img = mask_bounding_boxes_with_inpainting(img, blocks)

        width, height = img.size

        # Determine which side is shorter and create a square
        if width < height:
            # Width is shorter, scale height to match the width
            new_size = (width, width)
        else:
            # Height is shorter, scale width to match the height
            new_size = (height, height)

        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized


def mask_bounding_boxes_with_inpainting(image, bounding_boxes):
    """
    Remove text blocks from an image using inpainting.

    This function masks out the areas occupied by text blocks and fills them
    using OpenCV's inpainting algorithm, which creates a more natural-looking
    background for CLIP similarity calculation.

    Args:
        image: PIL Image to process
        bounding_boxes: List of bounding boxes as (x_ratio, y_ratio, w_ratio, h_ratio)

    Returns:
        PIL.Image: Image with text blocks removed and filled using inpainting
    """
    import cv2

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a black mask
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    height, width = image_cv.shape[:2]

    # Draw white rectangles on the mask for each bounding box
    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        w = int(w_ratio * width)
        h = int(h_ratio * height)
        mask[y : y + h, x : x + w] = 255

    # Use inpainting to fill the masked areas
    inpainted_image = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)

    # Convert back to PIL format
    inpainted_image_pil = Image.fromarray(
        cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
    )

    return inpainted_image_pil


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_current_cost(cost_matrix, row_ind, col_ind):
    """
    Calculate the total cost of a matching between rows and columns.

    Args:
        cost_matrix: 2D numpy array of costs
        row_ind: Row indices of the matching
        col_ind: Column indices of the matching

    Returns:
        list: List of costs for the matched pairs
    """
    return cost_matrix[row_ind, col_ind].tolist()


# ============================================================================
# DEBUGGING VISUALIZATION FUNCTIONS
# ============================================================================


def draw_matched_bboxes(img1, img2, matched_bboxes):
    import cv2
    
    # Create copies of images to draw on
    img1_drawn = img1.copy()
    img2_drawn = img2.copy()

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Iterate over matched bounding boxes
    for bbox_pair in matched_bboxes:
        # Random color for each pair
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Ensure that the bounding box coordinates are integers
        bbox1 = [
            int(bbox_pair[0][0] * w1),
            int(bbox_pair[0][1] * h1),
            int(bbox_pair[0][2] * w1),
            int(bbox_pair[0][3] * h1),
        ]
        bbox2 = [
            int(bbox_pair[1][0] * w2),
            int(bbox_pair[1][1] * h2),
            int(bbox_pair[1][2] * w2),
            int(bbox_pair[1][3] * h2),
        ]

        # Draw bbox on the first image
        top_left_1 = (bbox1[0], bbox1[1])
        bottom_right_1 = (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
        img1_drawn = cv2.rectangle(img1_drawn, top_left_1, bottom_right_1, color, 2)

        # Draw bbox on the second image
        top_left_2 = (bbox2[0], bbox2[1])
        bottom_right_2 = (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        img2_drawn = cv2.rectangle(img2_drawn, top_left_2, bottom_right_2, color, 2)

    return img1_drawn, img2_drawn
