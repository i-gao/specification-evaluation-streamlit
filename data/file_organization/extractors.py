from typing import List, Dict, Optional
import re
from data.file_organization.reward import apply_policy


# Extractors for file organization constraints
def check_file_folder_clustering(
    policy: Dict,
    original_filename: str,
    correct_cluster_files: List[str],
    files_data: List[Dict],
) -> List[str]:
    """
    Extract the list of files that are in the same folder as the original file.
    Returns a list of original filenames (including the original file itself).
    """
    try:
        organized = apply_policy(files_data, policy)

        # Find which folder the original file ended up in
        file_folder = None
        for folder, files in organized.items():
            for f in files:
                if f.get("original_filename") == original_filename:
                    file_folder = folder
                    break
            if file_folder:
                break

        if file_folder is None:
            return [], "File not found in any folder"

        # Return all original filenames in the same folder
        files_in_same_folder = []
        for f in organized.get(file_folder, []):
            orig_name = f.get("original_filename")
            if orig_name:
                files_in_same_folder.append(orig_name)

        return (
            files_in_same_folder,
            f"Folder: {file_folder} has contents: {files_in_same_folder}",
        )
    except Exception:
        return [], "Error checking file folder clustering"


def check_folder_cluster(
    policy: Dict,
    expected_cluster_files: List[str],
    expected_folder_name: Optional[str],
    files_data: List[Dict],
) -> List[str]:
    """
    Extract the list of files that are clustered together in the same folder.
    This checks if all expected_cluster_files are together, regardless of which folder they're in
    or what the folder is named.

    Args:
        policy: The file organization policy to apply
        expected_cluster_files: List of original filenames that should be in the same folder
        expected_folder_name: Optional folder name for reference only (not used in checking)
        files_data: The original files data

    Returns:
        List of original filenames that are actually in the same folder as any of the expected files.
        If files are split across folders, returns the files from the first folder found.
        The folder name is irrelevant - only that the files are together matters.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Find which folder(s) contain the expected files
        folders_with_expected_files = set()
        for folder, files in organized.items():
            for f in files:
                orig_name = f.get("original_filename")
                if orig_name in expected_cluster_files:
                    folders_with_expected_files.add(folder)

        if folders_with_expected_files:
            # If all expected files are in the same folder, return all files from that folder
            # Otherwise, return files from the first folder found (files are split)
            if len(folders_with_expected_files) == 1:
                # All expected files are in the same folder - good!
                folder = list(folders_with_expected_files)[0]
                files_in_folder = []
                for f in organized.get(folder, []):
                    orig_name = f.get("original_filename")
                    if orig_name:
                        files_in_folder.append(orig_name)
                return files_in_folder, f"Folder '{folder}' contains: {files_in_folder}"
            else:
                # Files are split across multiple folders - return files from first folder
                folder = list(folders_with_expected_files)[0]
                files_in_folder = []
                for f in organized.get(folder, []):
                    orig_name = f.get("original_filename")
                    if orig_name:
                        files_in_folder.append(orig_name)
                return (
                    files_in_folder,
                    f"Files split across folders. Folder '{folder}' contains: {files_in_folder}",
                )

        return [], "Expected files not found in any folder"
    except Exception:
        return [], "Error checking folder cluster"


def check_filename_matches_policy(
    policy: Dict,
    original_filename: str,
    expected_filename: str,
    files_data: List[Dict],
) -> bool:
    """
    Check if the filename matches the naming policy.
    Returns True if the organized filename matches the expected filename.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Find the organized filename for this file
        for folder, files in organized.items():
            for f in files:
                if f.get("original_filename") == original_filename:
                    organized_filename = f.get("filename", "")
                    # Compare normalized (case-insensitive, ignoring whitespace differences)
                    return (
                        organized_filename.lower().strip()
                        == expected_filename.lower().strip()
                    )

        return False
    except Exception:
        return False


def _extract_base_name_from_filename(filename: str) -> str:
    """Extract the base name (without extension, date, version) from a filename."""
    # Remove extension
    name_part = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Remove version indicators, dates, FINAL, DRAFT, etc.
    name_part = re.sub(
        r"[-_\s]*(v|ver|final|draft|copy)[-\s_]*\d*",
        "",
        name_part,
        flags=re.IGNORECASE,
    )

    # Try removing all datetime formats and pick the shortest result
    results = []
    for fmt in [
        r"\d{4}[-_\s]\d{2}[-_\s]\d{2}",
        r"\d{2}[-_\s]\d{2}[-_\s]\d{2}",
        r"\d{2}[-_\s]\d{2}",
        r"\d{4}",
    ]:
        results.append(re.sub(fmt, "", name_part))
    name_part = min(results, key=len)
    name_part = name_part.strip("-_ ")
    return name_part


def _check_case(filename: str, expected_case: Optional[str]) -> bool:
    """Check if filename matches the expected case."""
    if expected_case is None:
        return True  # No-op, always matches

    base_name = _extract_base_name_from_filename(filename)
    if not base_name:
        return True

    # Extract just the base name part (before any date/version)
    # Split on common delimiters to get words
    parts = re.split(r"[-_\s]+", base_name)
    if not parts or not parts[0]:
        return True

    if expected_case == "lower":
        return base_name.islower() or not any(c.isalpha() for c in base_name)
    elif expected_case == "upper":
        return base_name.isupper() or not any(c.isalpha() for c in base_name)
    elif expected_case == "title":
        # Check if each word is title case
        return all(
            part.istitle() or not any(c.isalpha() for c in part)
            for part in parts
            if part
        )
    elif expected_case == "camel":
        # Check if it's camelCase (first word lowercase, subsequent words capitalized)
        if len(parts) == 1:
            return parts[0][0].islower() if parts[0] else True
        return (
            parts[0][0].islower()
            if parts[0]
            else True and all(p[0].isupper() if p else True for p in parts[1:])
        )
    elif expected_case == "snake":
        # Should use underscores as delimiter
        return "_" in base_name or len(parts) == 1
    elif expected_case == "spaces":
        # Should use spaces as delimiter
        return " " in base_name or len(parts) == 1

    return True


def _check_delimiter(filename: str, expected_delimiter: Optional[str]) -> bool:
    """Check if filename uses the expected delimiter."""
    if expected_delimiter is None:
        return True  # No-op, always matches

    base_name = _extract_base_name_from_filename(filename)
    if not base_name:
        return True

    # Check if the delimiter appears in the base name
    if expected_delimiter == "_":
        # Should use underscores, not dashes or spaces
        return "_" in base_name or ("-" not in base_name and " " not in base_name)
    elif expected_delimiter == "-":
        # Should use dashes, not underscores or spaces
        return "-" in base_name or ("_" not in base_name and " " not in base_name)
    elif expected_delimiter == " ":
        # Should use spaces, not underscores or dashes
        return " " in base_name or ("_" not in base_name and "-" not in base_name)

    return True


def _check_date_inclusion(
    filename: str,
    expected_include_date: Optional[bool],
    expected_date_format: Optional[str],
    create_date: Optional[str],
) -> bool:
    """Check if filename includes date according to the expected policy."""
    if expected_include_date is None or not expected_include_date:
        # Date should not be included
        # Check if there's a date pattern in the filename
        date_patterns = [
            r"\d{4}[-_\s]\d{2}[-_\s]\d{2}",  # YYYY-MM-DD
            r"\d{2}[-_\s]\d{2}[-_\s]\d{2}",  # MM-DD-YY
            r"\d{2}[-_\s]\d{2}",  # MM-DD
            r"\d{4}",  # YYYY
        ]
        for pattern in date_patterns:
            if re.search(pattern, filename):
                return False
        return True

    # Date should be included
    if not create_date or not expected_date_format:
        return True  # Can't verify without date info

    # Check if date appears in filename
    try:
        from datetime import datetime

        date_obj = datetime.strptime(create_date, "%Y-%m-%d %H:%M:%S")
        expected_date_str = date_obj.strftime(expected_date_format)
        # Normalize delimiters (date format might use - but filename might use _ or space)
        expected_date_str_normalized = (
            expected_date_str.replace("-", "").replace("_", "").replace(" ", "")
        )
        filename_normalized = (
            filename.replace("-", "").replace("_", "").replace(" ", "")
        )
        return expected_date_str_normalized in filename_normalized
    except Exception:
        return True  # If we can't parse, assume it's okay


def _check_version_format(filename: str, expected_version: Optional[str]) -> bool:
    """Check if filename uses the expected version format."""
    if expected_version is None:
        # Version should not be included
        # Check if there's a version pattern
        version_patterns = [
            r"[-_\s]v\d+",
            r"[-_\s]\(v?\d+\)",
            r"[-_\s]ver\s*\d+",
        ]
        for pattern in version_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False
        return True

    # Version should be included - check if it matches the format
    # This is a simplified check - we just verify a version pattern exists
    version_patterns = [
        r"[-_\s]v\d+",
        r"[-_\s]\(v?\d+\)",
        r"[-_\s]ver\s*\d+",
    ]
    for pattern in version_patterns:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False


def check_naming_case(
    policy: Dict,
    expected_case: Optional[str],
    files_data: List[Dict],
) -> bool:
    """
    Check if all files use the expected case in their filenames.
    Returns True if all files match the expected case.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Check all files
        for folder, files in organized.items():
            for f in files:
                filename = f.get("filename", "")
                if filename and not _check_case(filename, expected_case):
                    return False

        return True
    except Exception:
        return False


def check_naming_delimiter(
    policy: Dict,
    expected_delimiter: Optional[str],
    files_data: List[Dict],
) -> bool:
    """
    Check if all files use the expected delimiter in their filenames.
    Returns True if all files match the expected delimiter.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Check all files
        for folder, files in organized.items():
            for f in files:
                filename = f.get("filename", "")
                if filename and not _check_delimiter(filename, expected_delimiter):
                    return False

        return True
    except Exception:
        return False


def check_naming_date(
    policy: Dict,
    expected_include_date: Optional[bool],
    expected_date_format: Optional[str],
    files_data: List[Dict],
) -> bool:
    """
    Check if all files include dates according to the expected policy.
    Returns True if all files match the expected date inclusion policy.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Check all files
        for folder, files in organized.items():
            for f in files:
                filename = f.get("filename", "")
                create_date = f.get("create_date")
                if filename and not _check_date_inclusion(
                    filename, expected_include_date, expected_date_format, create_date
                ):
                    return False

        return True
    except Exception:
        return False


def check_naming_version(
    policy: Dict,
    expected_version: Optional[str],
    files_data: List[Dict],
) -> bool:
    """
    Check if all files use the expected version format in their filenames.
    Returns True if all files match the expected version format.
    """
    try:
        organized = apply_policy(files_data, policy)

        # Check all files
        for folder, files in organized.items():
            for f in files:
                filename = f.get("filename", "")
                if filename and not _check_version_format(filename, expected_version):
                    return False

        return True
    except Exception:
        return False
