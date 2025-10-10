import difflib
import os
import re
from bs4 import BeautifulSoup, NavigableString, Comment


# ============================================================================
# HTML PREPROCESSING FUNCTIONS
# ============================================================================


def clean_html(html_content: str) -> str:
    """
    Pre-process predicted HTML code to fix common issues.

    This function performs several cleaning steps:
    1. Ensures proper HTML structure
    2. Truncates excessive repeated elements
    3. Replace image src with rick.jpg

    Args:
        html_content: HTML code as a string

    Returns:
        str: Pre-processed HTML code
    """
    html_content = make_html(html_content)
    soup = BeautifulSoup(html_content, "html.parser")
    soup = replace_image_src(soup)
    soup = truncate_repeated_html_elements(soup)
    return soup.prettify(formatter="html5")


def make_html(content):
    """
    Ensure a file contains valid HTML structure.

    If the file doesn't contain HTML tags, wrap the content in basic HTML structure.

    Args:
        filename: Path to the file to process
    """
    new_content = content
    # Check if content already has HTML structure
    if not re.search(r"<html[^>]*>", content, re.IGNORECASE):
        # Wrap content in basic HTML structure
        new_content = f"<html><body><p>{content}</p></body></html>"
    return new_content


def map_positions(clean_text, original_text):
    """
    Maps the positions from the clean text back to the original text.
    """
    map_clean_to_original = []
    original_idx = 0

    for clean_char in clean_text:
        while original_text[original_idx] != clean_char:
            original_idx += 1
        map_clean_to_original.append(original_idx)
        original_idx += 1

    return map_clean_to_original


def check_repetitive_content(
    file_path,
    chunk_size=100,
    repetition_threshold=5,
    similarity_threshold=0.8,
    debug=False,
):
    """
    Checks for repetitive content in a text file, considering both exact and similar chunks,
    ignoring HTML tags but keeping the original position reference.

    :param file_path: Path to the text file.
    :param chunk_size: The size of each chunk for comparison.
    :param repetition_threshold: Minimum number of repetitions to consider it as repetitive content.
    :param similarity_threshold: The threshold for considering two chunks as similar (0 to 1).
    :return: A tuple indicating if repetitive content was found and the position where it starts in the original file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Clean HTML content and keep a map of positions
    content_no_html = re.sub("<.*?>", "", content)
    position_map = map_positions(content_no_html, content)

    # Split content into chunks
    chunks = [
        content_no_html[i : i + chunk_size]
        for i in range(0, len(content_no_html), chunk_size)
    ]

    # Check for repetitive and similar chunks
    seen = {}
    repetitive_start = len(content_no_html)
    for i, chunk in enumerate(chunks):
        for seen_chunk, indexes in seen.items():
            similarity = difflib.SequenceMatcher(None, chunk, seen_chunk).ratio()
            if similarity >= similarity_threshold:
                indexes.append(i)
                if len(indexes) >= repetition_threshold:
                    clean_start = min(repetitive_start, indexes[0] * chunk_size)
                    c_repetitive_start = (
                        position_map[clean_start]
                        if clean_start < len(position_map)
                        else len(content)
                    )
                    if c_repetitive_start < repetitive_start:
                        repetitive_start = c_repetitive_start
                break
        else:
            seen[chunk] = [i]

    repetitive, start_position = (
        repetitive_start != len(content_no_html),
        repetitive_start,
    )

    if repetitive:
        print(
            f"[Warning] Repetitive content found in {file_path}, start at {start_position}"
        )
        print(
            f"[Warning] You might want to manually check whether the automatic repetition removal is correct."
        )
        if not debug:
            os.rename(file_path, file_path.replace(".html", "_old.txt"))
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content[:start_position])
        else:
            with open(
                file_path.replace(".html", "_new.html"), "w", encoding="utf-8"
            ) as file:
                file.write(content[:start_position])


def replace_image_src(soup):
    """
    Replaces the image src with rick.jpg.
    """
    for img in soup.find_all("img"):
        img["src"] = "rick.jpg"
    return soup


def truncate_repeated_html_elements(soup, max_count=50):
    """
    Remove excessive repeated HTML elements from a BeautifulSoup object.

    This function prevents HTML files from having too many identical elements,
    which can cause issues during processing and evaluation.

    Args:
        soup: BeautifulSoup object to process
        max_count: Maximum number of identical elements to keep

    Returns:
        str: HTML string with excessive repetitions removed
    """
    content_counts = {}

    for element in soup.find_all(True):
        if isinstance(element, (NavigableString, Comment)):
            continue

        try:
            element_html = str(element)
        except:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1

        # Remove element if it appears too many times
        if content_counts[element_html] > max_count:
            element.decompose()

    return soup
