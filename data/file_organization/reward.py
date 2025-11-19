from typing import List, Dict
import re
from collections import defaultdict
from datetime import datetime


class NamingPolicy:
    """Defines a per-user organized filename + date policy."""

    CASES = ["lower", "upper", "title", "camel", "snake", "spaces"]
    DATE_FORMATS = [None, "%Y-%m-%d", "%m-%d-%y", "%m-%d", "%Y"]
    DELIMITERS = ["_", "-", " "]
    VERSION_FORMATS = [None, "_v{n}", " ({n})", " v{n}", "-{n}"]

    def __init__(
        self,
        case: str,
        delimiter: str,
        include_date: bool,
        date_format: str,
        version: str,
    ):
        self.case = case
        self.delimiter = delimiter
        self.include_date = include_date
        self.date_format = date_format
        self.version = version

    def canonicalize(self, base, ext, date=None, version=None):
        """Return filename in this policy."""
        name = base
        # Normalize: split on any delimiter (_, -, or space) to get parts
        # Note: camelCase doesn't use delimiters - it's concatenated
        parts = re.split(r"[-_\s]+", name)
        if self.case == "camel":
            parts = [parts[0].lower()] + [p.capitalize() for p in parts[1:]]
            name = "".join(parts)
        elif self.delimiter is not None:
            name = self.delimiter.join(parts)
        else:
            name = name

        # Handle case
        if self.case == "lower":
            name = name.lower()
        elif self.case == "upper":
            name = name.upper()
        elif self.case == "title":
            name = name.title()

        # Place date and/or version
        date_str = ""
        if (self.include_date) and date and self.date_format:
            date_str = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime(
                self.date_format
            )
            # Normalize date format separators to use chosen delimiter
            if self.delimiter is not None and self.delimiter != "-":
                date_str = date_str.replace("-", self.delimiter)

        name_pieces = [name]
        if date_str:
            name_pieces.append(date_str)

        if self.version and version is not None:
            # Normalize version format to use chosen delimiter
            version_str = self.version.format(n=version)
            # Replace common separators in version format with chosen delimiter
            if self.delimiter == " ":
                # For spaces, replace underscores and dashes with spaces
                version_str = version_str.replace("_", " ").replace("-", " ")
            elif self.delimiter is not None:
                # For non-space delimiters, replace spaces and other delimiters
                version_str = version_str.replace(" ", self.delimiter)
                if self.delimiter != "_":
                    version_str = version_str.replace("_", self.delimiter)
                if self.delimiter != "-":
                    version_str = version_str.replace("-", self.delimiter)
            name_pieces.append(version_str)

        # Join all pieces with the delimiter
        out = (
            self.delimiter.join([p for p in name_pieces if p])
            if self.delimiter is not None
            else "-".join(name_pieces)
        )

        # Final consistency check: ensure all separators match the chosen delimiter
        if self.delimiter == " ":
            # Replace any remaining underscores or dashes with spaces
            out = out.replace("_", " ").replace("-", " ")
            # Normalize multiple spaces to single space
            out = re.sub(r"\s+", " ", out).strip()
        elif self.delimiter is not None:
            # Replace any remaining spaces or other delimiters with chosen delimiter
            out = out.replace(" ", self.delimiter)
            if self.delimiter != "_":
                out = out.replace("_", self.delimiter)
            if self.delimiter != "-":
                out = out.replace("-", self.delimiter)

        if "." not in ext:
            ext = "." + ext
        return f"{out}{ext}"


def apply_moving_rules(
    working_files: List[Dict], moving_rules: List[Dict]
) -> List[str]:
    # Step 1: Apply moving rules to match files and move them to folders
    for rule in moving_rules:
        conditions = rule.get("conditions", {})
        folder = rule.get("folder", "Unsorted")

        for wf in working_files:
            if wf["_matched"]:
                continue

            matches = False

            # content_contains (look inside preview if available)
            if "content_contains" in conditions:
                preview = wf.get("file_contents_preview") or ""
                preview_lower = preview.lower()
                # Normalize preview by removing spaces, dashes, underscores for robust matching
                preview_normalized = (
                    preview_lower.replace("-", "").replace("_", "").replace(" ", "")
                )
                contains_any = any(
                    keyword.lower().replace("-", "").replace("_", "").replace(" ", "")
                    in preview_normalized
                    for keyword in conditions["content_contains"]
                )
                if contains_any:
                    matches = True

            # name_contains (optional, for matching by filename)
            if not matches and "name_contains" in conditions:
                name_lower = wf["filename"].lower()
                for keyword in conditions["name_contains"]:
                    k = (
                        keyword.lower()
                        .replace("-", "")
                        .replace("_", "")
                        .replace(" ", "")
                    )
                    fname_stripped = (
                        name_lower.replace("-", "").replace("_", "").replace(" ", "")
                    )
                    if k in fname_stripped:
                        matches = True
                        break

            if matches:
                wf["_folder"] = folder
                wf["_matched"] = True

    return working_files


def apply_naming_policy(working_files: List[Dict], user_pol: Dict) -> List[Dict]:
    for wf in working_files:
        name_part, ext = (
            wf["filename"].rsplit(".", 1)
            if "." in wf["filename"]
            else (wf["filename"], "")
        )
        
        # Split camelCase into words by inserting a delimiter before uppercase letters
        # Pattern: lowercase letter followed by uppercase letter -> insert space
        # This handles camelCase like "consentPendingList" -> "consent Pending List"
        if re.search(r'[a-z][A-Z]', name_part):
            # Insert a space before each uppercase letter that follows a lowercase letter or digit
            name_part = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name_part)
        
        # Remove version indicators, dates, FINAL, DRAFT, etc.
        name_part = re.sub(
            r"[-_\s]*(v|ver|final|draft|copy)[-\s_]*\d*",
            "",
            name_part,
            flags=re.IGNORECASE,
        )

        # try removing all datetime formats  and pick the shortest result
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
        base_name = name_part
        wf["_base_name"] = base_name
        wf["_extension"] = ext

    naming_policy_obj = NamingPolicy(
        case=user_pol.get("case"),
        delimiter=user_pol.get("delimiter"),
        include_date=user_pol.get("include_date"),
        date_format=user_pol.get("date_format"),
        version=user_pol.get("version"),
    )

    # Step 3: Add version numbers if version format is specified
    folder_groups = defaultdict(list)
    for idx, wf in enumerate(working_files):
        folder = wf.get("_folder", "Unsorted")
        key = (folder, wf.get("_base_name"), wf.get("_extension"))
        folder_groups[key].append(idx)

    for (folder, base_name, ext), group in folder_groups.items():
        if len(group) > 1:
            # Sort by create_date and assign version numbers based on sorted position
            sorted_indices = sorted(group, key=lambda x: working_files[x]["create_date"])
            for version_num, idx in enumerate(sorted_indices):
                working_files[idx]["_new_name"] = naming_policy_obj.canonicalize(
                    base_name, ext, date=working_files[idx].get("create_date"), version=version_num
                )
        else:
            working_files[group[0]]["_new_name"] = naming_policy_obj.canonicalize(
                base_name, ext, date=working_files[group[0]].get("create_date")
            )
    return working_files


def apply_policy(files: List[Dict], policy: Dict) -> Dict[str, List[Dict]]:
    """
    Apply moving rules first to organize files into folders, then apply naming policy to all files.
    """
    working_files = []
    for f in files:
        wf = f.copy()
        wf["_original_filename"] = f["filename"]
        wf["_matched"] = False
        wf["_folder"] = None
        working_files.append(wf)

    working_files = apply_moving_rules(working_files, policy.get("moving_rules", []))
    working_files = apply_naming_policy(working_files, policy.get("naming_policy", {}))
    # Output files organized by folders
    output = defaultdict(list)
    for wf in working_files:
        folder = wf.get("_folder") or "Unsorted"
        result_file = {k: v for k, v in wf.items() if not k.startswith("_")}
        result_file["filename"] = wf["_new_name"]
        result_file["original_filename"] = wf["_original_filename"]
        output[folder].append(result_file)
    return dict(output)


# ============================================================================
# Constraint Creation Functions
# ============================================================================


def clean_theme_name(theme: str) -> str:
    """Clean theme name: lowercase and replace underscores with spaces."""
    return theme.lower().replace("_", " ")


def create_file_organization_constraints(
    organized_data: Dict[str, List[Dict]],
    gold_policy: Dict,
    files_data: List[Dict],
) -> List[Dict]:
    """
    Create all constraints for file organization.
    
    Args:
        organized_data: Dictionary mapping folder names to lists of file dictionaries
        gold_policy: The gold standard policy dictionary
        files_data: List of original file dictionaries
    
    Returns:
        List of constraint dictionaries ready to be converted to Constraint objects
    """
    constraints = []
    
    # Build folder-to-files mapping for clustering constraints
    folder_to_files = {}
    # Build a mapping from original filename to theme
    # Try from files_data first (if from gold directory), then from organized_data
    filename_to_theme = {}
    
    # First, try to get themes from files_data (if it's from the gold directory)
    for file_info in files_data:
        filename = file_info.get("filename", "")
        theme = file_info.get("theme") or file_info.get("_theme")
        if filename and theme:
            filename_to_theme[filename] = theme
    
    # Then, extract from organized_data (this will overwrite if there are conflicts, but organized_data is authoritative)
    for folder, files in organized_data.items():
        original_filenames = [
            f.get("original_filename")
            for f in files
            if f.get("original_filename")
        ]
        folder_to_files[folder] = original_filenames
        
        # Extract theme information from organized_data
        for f in files:
            orig_name = f.get("original_filename")
            theme = f.get("theme") or f.get("_theme")
            if orig_name and theme:
                filename_to_theme[orig_name] = theme
    
    # Constraint: Folder clustering (one constraint per unique cluster)
    # Only create constraints for folders with multiple files
    for folder, cluster_files in folder_to_files.items():
        if len(cluster_files) > 1:
            # Create one constraint per cluster checking all files are together
            # Note: folder name is only used for reference - the constraint checks
            # that files are clustered together regardless of folder name
            cluster_files_sorted = sorted(cluster_files)
            
            # Extract theme(s) for this cluster
            themes = set()
            for filename in cluster_files_sorted:
                theme = filename_to_theme.get(filename)
                if theme:
                    themes.add(theme)
            
            # Build description using theme if available, otherwise fall back to filenames
            if themes:
                if len(themes) == 1:
                    theme_name = clean_theme_name(list(themes)[0])
                    description = f"Files related to {theme_name} should be clustered together"
                else:
                    # Multiple themes in one cluster - list them
                    themes_sorted = sorted([clean_theme_name(t) for t in themes])
                    description = f"Files related to themes {', '.join(f'\"{t}\"' for t in themes_sorted)} should be clustered together"
            else:
                # Fallback to filenames if no theme information available
                description = f"Files should be clustered together: {', '.join(cluster_files_sorted)}"
            
            constraints.append(
                {
                    "type": "multiset_jaccard",
                    "description": description,
                    "is_hard": False,
                    "is_discoverable": True,
                    "is_minimal": False,
                    "extractor": "check_folder_cluster",
                    "extractor_kwargs": {
                        "expected_cluster_files": cluster_files_sorted,
                        "expected_folder_name": folder,  # Used for reference only
                        "files_data": files_data,
                    },
                    "true_set": cluster_files_sorted,
                }
            )
    
    # High-level naming policy constraints (one per aspect, checking all files)
    naming_policy = gold_policy.get("naming_policy", {})
    
    # Constraint: Case
    expected_case = naming_policy.get("case")
    if expected_case is not None:
        # Convert technical case names to user-friendly descriptions
        case_descriptions = {
            "lower": "all lowercase letters (e.g., 'my file name')",
            "upper": "all uppercase letters (e.g., 'MY FILE NAME')",
            "title": "first letter of each word capitalized (e.g., 'My File Name')",
            "camel": "first word lowercase, then each word capitalized with no spaces (e.g., 'myFileName')",
            "snake": "all lowercase with underscores between words (e.g., 'my_file_name')",
            "spaces": "words separated by spaces",
        }
        case_desc = case_descriptions.get(expected_case, expected_case)
        constraints.append(
            {
                "type": "boolean_reward_true",
                "description": f"All filenames should use {case_desc}",
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "check_naming_case",
                "extractor_kwargs": {
                    "expected_case": expected_case,
                    "files_data": files_data,
                },
                "none_val": 0,
            }
        )
    
    # Constraint: Delimiter
    expected_delimiter = naming_policy.get("delimiter")
    if expected_delimiter is not None:
        # Convert delimiter to user-friendly description
        delimiter_descriptions = {
            "_": "underscores",
            "-": "dashes",
            " ": "spaces",
        }
        delimiter_desc = delimiter_descriptions.get(expected_delimiter, f"'{expected_delimiter}'")
        constraints.append(
            {
                "type": "boolean_reward_true",
                "description": f"All filenames should separate words using {delimiter_desc}",
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "check_naming_delimiter",
                "extractor_kwargs": {
                    "expected_delimiter": expected_delimiter,
                    "files_data": files_data,
                },
                "none_val": 0,
            }
        )
    
    # Constraint: Date inclusion and format
    expected_include_date = naming_policy.get("include_date")
    expected_date_format = naming_policy.get("date_format")
    if expected_include_date is not None:
        if expected_include_date:
            # Convert date format codes to user-friendly examples
            date_format_examples = {
                "%Y-%m-%d": "'2024-03-15'",
                "%m-%d-%y": "'03-15-24'",
                "%m-%d": "'03-15'",
                "%Y": "'2024'",
            }
            if expected_date_format and expected_date_format in date_format_examples:
                date_desc = f"All filenames should include dates in the format {date_format_examples[expected_date_format]}"
            elif expected_date_format:
                date_desc = f"All filenames should include dates in format '{expected_date_format}'"
            else:
                date_desc = "All filenames should include dates"
        else:
            date_desc = "All filenames should not include dates"
        constraints.append(
            {
                "type": "boolean_reward_true",
                "description": date_desc,
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "check_naming_date",
                "extractor_kwargs": {
                    "expected_include_date": expected_include_date,
                    "expected_date_format": expected_date_format,
                    "files_data": files_data,
                },
                "none_val": 0,
            }
        )
    
    # Constraint: Version format
    expected_version = naming_policy.get("version")
    if expected_version is not None:
        # Convert version format codes to user-friendly examples
        version_format_examples = {
            "_v{n}": "underscore followed by 'v' and number (e.g., '_v1', '_v2')",
            " ({n})": "space, then number in parentheses (e.g., ' (1)', ' (2)')",
            " v{n}": "space, then 'v' and number (e.g., ' v1', ' v2')",
            "-{n}": "dash followed by number (e.g., '-1', '-2')",
        }
        if expected_version in version_format_examples:
            version_desc = f"All filenames should include version numbers using {version_format_examples[expected_version]}"
        else:
            version_desc = f"All filenames should use version format '{expected_version}'"
        constraints.append(
            {
                "type": "boolean_reward_true",
                "description": version_desc,
                "is_hard": False,
                "is_discoverable": True,
                "is_minimal": False,
                "extractor": "check_naming_version",
                "extractor_kwargs": {
                    "expected_version": expected_version,
                    "files_data": files_data,
                },
                "none_val": 0,
            }
        )
    
    return constraints
