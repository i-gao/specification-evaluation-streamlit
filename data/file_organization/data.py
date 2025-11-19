from typing import List, Optional, Dict, Tuple
import os
import json
from data.dataset import (
    SpecificationCollection,
    LinearFixedSpecification,
    CustomSpecification,
)
from data.actions import get_jupyter_actions, Action
from data.reward import Constraint
from langchain_core.tools import tool
from data.file_organization.extractors import (
    check_file_folder_clustering,
    check_filename_matches_policy,
    check_naming_case,
    check_naming_delimiter,
    check_naming_date,
    check_naming_version,
    check_folder_cluster,
)
from data.file_organization.reward import (
    apply_policy,
    create_file_organization_constraints,
)
from data.file_organization.streamlit_render import (
    render_file_policy_results,
    render_file_policy_results_txt,
    render_eval as render_file_eval,
)
from data.file_organization.streamlit_search_interface import render_search_interface as render_file_search_interface
from data.file_organization.parser import (
    parse_policy,
    parse_file_organization_solutions,
)
from utils.streamlit_types import FormElement, DisplayElement
import streamlit as st
import pandas as pd

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
COMMONSENSE_DESCRIPTION = """A file organization policy is a tuple (moving_rules, naming_policy). First, the moving rules are applied: they are an ordered list of rules which select files in the working directory using filters and then move them into folders. Then, all files are renamed according to the naming policy. The goal is to clean up the working directory, clustering the correct files together and standardizing the filenames. The resulting directory will be 
- Uncategorized/
- folder1/
- folder2/ (not the real name of the folder)
There are no nested folders.

Eac moving rule is a tuple (conditions, folder). The conditions specify filters on the files to select them (namely: name_contains, which checks if the filename contains a string literal, and content_contains, which checks if the file contents contain a string literal). The filters are OR'd together. The folder specifies the folder to move the selected files into. Note that no file will be moved twice: once a rule has selected a file, it will not be selected by any other rule.

MovingRule(
    conditions: {
        name_contains: List[str],
        content_contains: List[str],
    }
    folder: str,
)

The naming policy is a tuple (case, delimiter, include_date, date_format, version). The case specifies the capitalization scheme of the filename. The delimiter specifies the separator between words in the filename. The include_date specifies whether to include the file's creation date in the filename. The date_format specifies the format of the date. The version specifies the format of version numbers in the filename (e.g. v1, v2).

All of these are optional. If a field is not specified, it is a no-op (no operation) - the original value is preserved or the feature is skipped.

NamingPolicy(
    case: Optional[Literal["lower", "upper", "title", "camel", "snake", "spaces"]],
    delimiter: Optional[Literal["_", "-", " "]],
    include_date: Optional[bool],
    date_format: Optional[Literal["%Y-%m-%d", "%m-%d-%y", "%m-%d", "%Y"]],
    version: Optional[Literal["_v{n}", " ({n})", " v{n}", "-{n}"]],
)

The policy will be applied to the raw files, and the result will be evaluated based on:
- Whether files are clustered correctly (grouped with the right other files)
- Whether filenames match the expected naming policy
"""

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to organize files according to a policy.** You will see a collection of unorganized files, and you need to create a policy that will organize them correctly.

### The task
You will see files with their original filenames, creation dates, edit dates, and content previews. Your job is to create a policy (as a JSON object) that:
1. Groups related files into folders based on their content or filenames
2. Renames files according to a consistent naming policy

The policy will be automatically applied to the files, and you'll be scored based on how well the organized result matches the desired organization.
"""

PREDICTION_FMT_INSTRUCTIONS = """Return the policy as a JSON object wrapped in <policy></policy> tags:

<policy>
{
  "moving_rules": [
    {
      "conditions": {
        "content_contains": ["keyword1", "keyword2"], // optional
        "name_contains": ["keyword3"]  // optional
      },
      "folder": "Folder Name"
    },
    ...
  ],
  "naming_policy": {
    "case": Optional[Literal["lower", "upper", "title", "camel", "snake", "spaces"]],
    "delimiter": Optional[Literal["_", "-", " "]],
    "include_date": Optional[bool],
    "date_format": Optional[Literal["%Y-%m-%d", "%m-%d-%y", "%m-%d", "%Y"]],
    "version": Optional[Literal["_v{n}", " ({n})", " v{n}", "-{n}"]],
  }
}
</policy>

The conditions in each moving rule are OR'd together.
"""


def render_fixed_task_explanation():
    """Render the fixed task explanation for file organization."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


def render_custom_task_explanation():
    """Render the custom task explanation for file organization."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


def check_file_policy_validity(yhat: str, raise_errors: bool = False) -> Tuple[bool, dict]:
    """
    Check if a file organization policy is valid.
    
    Args:
        yhat: The policy string (may be wrapped in <policy></policy> tags)
        raise_errors: If True, raise an error if invalid
    
    Returns:
        Tuple of (is_valid, metadata)
    """
    policy = parse_policy(yhat, raise_errors=raise_errors)
    if policy is None:
        if raise_errors:
            raise ValueError(
                "Could not parse the policy. Wrap the JSON policy in <policy></policy> tags."
            )
        return False, {"error": "Could not parse the policy"}
    
    # Validate that policy is a dict
    if not isinstance(policy, dict):
        if raise_errors:
            raise ValueError(
                f"Policy must be a dictionary, but got {type(policy).__name__}."
            )
        return False, {"error": f"Policy must be a dictionary, but got {type(policy).__name__}"}
    
    # Validate moving_rules if present
    if "moving_rules" in policy:
        moving_rules = policy["moving_rules"]
        if not isinstance(moving_rules, list):
            if raise_errors:
                raise ValueError(f"'moving_rules' must be a list, but got {type(moving_rules).__name__}.")
            return False, {"error": f"'moving_rules' must be a list, but got {type(moving_rules).__name__}"}
        for i, rule in enumerate(moving_rules):
            if not isinstance(rule, dict):
                if raise_errors:
                    raise ValueError(f"Moving rule at index {i} must be a dictionary, but got {type(rule).__name__}.")
                return False, {"error": f"Moving rule at index {i} must be a dictionary"}
            if "folder" not in rule:
                if raise_errors:
                    raise ValueError(f"Moving rule at index {i} is missing the 'folder' field.")
                return False, {"error": f"Moving rule at index {i} is missing 'folder'"}
            if "conditions" not in rule:
                if raise_errors:
                    raise ValueError(f"Moving rule at index {i} is missing the 'conditions' field.")
                return False, {"error": f"Moving rule at index {i} is missing 'conditions'"}
    
    # Validate naming_policy if present
    if "naming_policy" in policy:
        naming_policy = policy["naming_policy"]
        if not isinstance(naming_policy, dict):
            if raise_errors:
                raise ValueError(f"'naming_policy' must be a dictionary, but got {type(naming_policy).__name__}.")
            return False, {"error": f"'naming_policy' must be a dictionary, but got {type(naming_policy).__name__}"}
    
    return True, {}


class FileOrganizationDataset(SpecificationCollection):
    @property
    def dataset_name(self) -> str:
        return "file_organization"

    @property
    def dataset_pretty_name(self) -> str:
        return "File Organization"

    @property
    def dataset_description(self) -> str:
        return "Organize files according to a policy that groups related files and applies consistent naming."

    @property
    def assets_file_id(self) -> str:
        return None  # No Google Drive assets

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return None

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create user expertise form elements."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with file organization and naming conventions?",
                options=[
                    "Not familiar at all",
                    "Somewhat familiar",
                    "Moderately familiar",
                    "Very familiar",
                    "Expert",
                ],
                default="Moderately familiar",
                required=True,
                help="This helps us understand your experience level with file organization",
            )
        ]

    def __init__(
        self,
        dev: bool = False,
        fixed_indexes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Find all UUIDs in the assets directory for the fixed specs
        policy_dir = os.path.join(DATASET_ROOT, "assets", "dataset_policy")
        if not os.path.exists(policy_dir):
            self._intents = {}
            self.fixed_length = 0
        else:
            policy_files = [
                f
                for f in os.listdir(policy_dir)
                if f.endswith("_policy.json") and not f.startswith("custom_")
            ]
            # Extract UUIDs (filename format: {uuid}_policy.json)
            uuids = [f.replace("_policy.json", "") for f in policy_files]
            self._uuids = sorted(uuids)
            self.fixed_length = len(self._uuids)

        # Load the custom spec
        if os.path.exists(os.path.join(policy_dir, "custom_policy.json")):
            with open(os.path.join(policy_dir, "custom_policy.json"), "r") as f:
                self._custom_policy = json.load(f)
            self.custom_length = 1
        else:
            self.custom_length = 0

        # Build extractor lookup
        self._extractor_lookup = {
            "check_file_folder_clustering": check_file_folder_clustering,
            "check_filename_matches_policy": check_filename_matches_policy,
            "check_naming_case": check_naming_case,
            "check_naming_delimiter": check_naming_delimiter,
            "check_naming_date": check_naming_date,
            "check_naming_version": check_naming_version,
            "check_folder_cluster": check_folder_cluster,
        }

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)

    def _load_files_data(self, uuid_or_custom: str) -> Optional[List[Dict]]:
        """
        Load files data from JSON file, trying gold directory first, then regular.
        
        Args:
            uuid_or_custom: UUID string for fixed specs, or "custom" for custom specs
        
        Returns:
            List of file dictionaries, or None if files don't exist
        """
        if uuid_or_custom == "custom":
            files_gold_path = os.path.join(
                DATASET_ROOT, "assets", "dataset_files_gold", "custom_files.json"
            )
            files_path = os.path.join(
                DATASET_ROOT, "assets", "dataset_files", "custom_files.json"
            )
        else:
            files_gold_path = os.path.join(
                DATASET_ROOT, "assets", "dataset_files_gold", f"{uuid_or_custom}_files.json"
            )
            files_path = os.path.join(
                DATASET_ROOT, "assets", "dataset_files", f"{uuid_or_custom}_files.json"
            )
        
        # Try to load from gold directory first (has theme info), otherwise use regular files
        if os.path.exists(files_gold_path):
            with open(files_gold_path, "r") as f:
                return json.load(f)
        elif os.path.exists(files_path):
            with open(files_path, "r") as f:
                return json.load(f)
        return None

    def _build_file_fields(self, sample_file: Dict) -> Dict[str, str]:
        """
        Build file field descriptions from a sample file.
        
        Returns:
            Dictionary mapping field names to descriptions
        """
        file_fields = {}
        for key in sample_file.keys():
            if key == "filename":
                file_fields[key] = "Original filename of the file"
            elif key == "create_date":
                file_fields[key] = "Creation date of the file (format: YYYY-MM-DD HH:MM:SS)"
            elif key == "edit_date":
                file_fields[key] = "Last edit date of the file (format: YYYY-MM-DD HH:MM:SS)"
            elif key == "file_contents_preview":
                file_fields[key] = "Preview of the file contents (first portion of the file)"
        return file_fields

    def _create_ls_output(self, files_data: List[Dict], uuid_or_custom: str) -> List[Dict]:
        """
        Create ls_output description for files JSON file.
        
        Args:
            files_data: List of file dictionaries
            uuid_or_custom: UUID string for fixed specs, or "custom" for custom specs
        
        Returns:
            List with single dict describing the files JSON file
        """
        sample_file = files_data[0] if files_data else {}
        file_fields = self._build_file_fields(sample_file)
        
        if uuid_or_custom == "custom":
            files_json_path = os.path.join("dataset_files_gold", "custom_files.json")
        else:
            files_json_path = os.path.join("dataset_files", f"{uuid_or_custom}_files.json")
        
        return [
            {
                "filename": files_json_path,
                "description": f"JSON file containing {len(files_data)} files to organize. Each entry in the JSON array represents one file with its metadata. Use json and pandas to load and investigate this file (e.g., `import json; files_data = json.load(open('{files_json_path}'))` or `import pandas as pd; files_df = pd.read_json('{files_json_path}')`).",
                "columns": file_fields,
            }
        ]

    def _create_apply_policy_tool(self, files_data: List[Dict]) -> Action:
        """
        Create the apply_policy_and_see_results tool.
        
        Returns:
            Action object for the tool
        """
        @tool(parse_docstring=True)
        def apply_policy_and_see_results(policy_json: str) -> str:
            """
            Apply a file organization policy to the files and see the results.
            This tool shows which files are organized into which folders and how they are renamed after applying the policy.

            Args:
                policy_json: A JSON string representing the policy. The policy should be a dictionary with
                             "moving_rules" (list of rule objects) and "naming_policy" (dictionary with naming options).
                             Optionally wrapped in <policy></policy> tags.
            """
            try:
                policy = parse_policy(policy_json, raise_errors=False)

                if policy is None:
                    return "Error: Could not parse the policy. Please provide a valid JSON object with 'moving_rules' and 'naming_policy', optionally wrapped in <policy></policy> tags."

                if not isinstance(policy, dict):
                    return f"Error: Policy must be a dictionary, but got {type(policy).__name__}."

                # Validate moving_rules if present
                if "moving_rules" in policy:
                    moving_rules = policy["moving_rules"]
                    if not isinstance(moving_rules, list):
                        return f"Error: 'moving_rules' must be a list, but got {type(moving_rules).__name__}."
                    for i, rule in enumerate(moving_rules):
                        if not isinstance(rule, dict):
                            return f"Error: Moving rule at index {i} must be a dictionary, but got {type(rule).__name__}."
                        if "folder" not in rule:
                            return f"Error: Moving rule at index {i} is missing the 'folder' field."
                        if "conditions" not in rule:
                            return f"Error: Moving rule at index {i} is missing the 'conditions' field."

                organized = apply_policy(files_data, policy)

                result_lines = []
                result_lines.append(
                    "File organization results after applying the policy:\n"
                )
                result_lines.append("=" * 80 + "\n")

                sorted_folders = sorted(organized.keys())

                for folder in sorted_folders:
                    files = organized[folder]
                    if not files:
                        continue

                    result_lines.append(f"\n📁 {folder} ({len(files)} file(s)):")
                    result_lines.append("-" * 80)

                    for file in files:
                        original_filename = file.get("original_filename", "")
                        new_filename = file.get("filename", "")
                        create_date = file.get("create_date", "")
                        if len(original_filename) > 60:
                            original_filename = original_filename[:57] + "..."
                        if len(new_filename) > 60:
                            new_filename = new_filename[:57] + "..."

                        result_lines.append(f"  • Original: {original_filename}")
                        result_lines.append(f"    New name: {new_filename}")
                        if create_date:
                            result_lines.append(f"    Created: {create_date}")
                        result_lines.append("")

                if "Unsorted" in organized and organized["Unsorted"]:
                    result_lines.append(
                        f"\n⚠️  Unsorted ({len(organized['Unsorted'])} file(s)):"
                    )
                    result_lines.append("-" * 80)
                    for file in organized["Unsorted"]:
                        original_filename = file.get("original_filename", "")
                        if len(original_filename) > 60:
                            original_filename = original_filename[:57] + "..."
                        result_lines.append(f"  • {original_filename}")

                return "\n".join(result_lines)

            except Exception as e:
                return f"Error applying policy: {str(e)}"

        return Action(
            fn=apply_policy_and_see_results,
            is_public=True,
            is_human=False,
            name="Apply file organization policy and see results",
        )

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, LinearFixedSpecification]:
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            if ix >= len(self._uuids):
                continue

            uuid = self._uuids[ix]

            # Load the files data
            files_data = self._load_files_data(uuid)
            if files_data is None:
                continue

            # Load the policy and organized data
            policy_path = os.path.join(
                DATASET_ROOT, "assets", "dataset_policy", f"{uuid}_policy.json"
            )
            organized_path = os.path.join(
                DATASET_ROOT,
                "assets",
                "dataset_organized",
                f"{uuid}_organized.json",
            )

            with open(policy_path, "r") as f:
                gold_policy = json.load(f)
            with open(organized_path, "r") as f:
                organized_data = json.load(f)

            # Build a mapping from original filename to expected organized filename
            file_to_expected_filename = {}
            for folder, files in organized_data.items():
                for f in files:
                    orig_name = f.get("original_filename")
                    if orig_name:
                        file_to_expected_filename[orig_name] = f.get("filename", "")

            # Create constraints using the reward module
            features_dicts = create_file_organization_constraints(
                organized_data=organized_data,
                gold_policy=gold_policy,
                files_data=files_data,
            )

            # Convert to Constraint objects
            features: List[Constraint] = [
                Constraint.from_dict(fd, extractor_lookup=self._extractor_lookup)
                for fd in features_dicts
            ]

            # Weights: equal importance for all constraints
            weights: List[float] = [1.0] * len(features)

            # Build specification text
            signature = f"Organize the {len(files_data)} files into folders, and rename files for clarity."

            # Create ls_output and get Jupyter actions
            ls_output = self._create_ls_output(files_data, uuid)
            filename, actions = get_jupyter_actions(
                docker_image=None,
                docker_container_id=None,
                ls_output=ls_output,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Add the apply policy tool
            actions.append(self._create_apply_policy_tool(files_data))

            # Create the ystar (gold policy as JSON string)
            ystar = "<policy>" + json.dumps(gold_policy) + "</policy>"

            spec = LinearFixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                initial_specification=signature,
                commonsense_description=COMMONSENSE_DESCRIPTION,
                features=features,
                weights=weights,
                parse_y_fn=parse_policy,
                validity_fn_tool_name="check_file_organization_policy_validity",
                validity_fn_tool_description="Check if the file organization policy is valid JSON",
                reward_fn_tool_name="score_file_organization_policy",
                reward_fn_tool_description="Score the file organization policy based on folder clustering and filename matching",
                ystar=ystar,
                render_task_explanation=render_fixed_task_explanation,
                actions=actions,
                msg_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=render_file_policy_results,
                render_msg_fn_txt=render_file_policy_results_txt,
                render_msg_kwargs=["files_data"],
                name=f"file_organization_{uuid}",
                state_files=[filename] if filename else [],
                files_to_clean=[filename] if filename else [],
                container_ids=[],
                user_expertise_form=self._create_user_expertise_form(),
                parse_solutions_fn=parse_file_organization_solutions,
                initial_shared_state=[
                    (
                        "Files to organize",
                        DisplayElement(
                            input_type="dataframe",
                            value=pd.DataFrame(files_data),
                            hide_index=True,
                        ),
                    ),
                ],
                files_data=files_data,
            )

            specs[ix] = spec

        return specs

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        if indexes is None:
            return {}

        # Check if custom policy exists
        if not hasattr(self, "_custom_policy") or self._custom_policy is None:
            return {}

        specs = {}
        for ix in indexes:
            if ix >= self.custom_length:
                continue

            # Load the custom files
            files_data = self._load_files_data("custom")
            if files_data is None:
                continue

            # Load the custom policy (already loaded in __init__)
            gold_policy = self._custom_policy

            # Build specification text
            signature = f"Organize the {len(files_data)} files according to a policy that groups related files and applies consistent naming."

            # Create ls_output and get Jupyter actions
            ls_output = self._create_ls_output(files_data, "custom")
            filename, actions = get_jupyter_actions(
                docker_image=None,
                docker_container_id=None,
                ls_output=ls_output,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Add the apply policy tool
            actions.append(self._create_apply_policy_tool(files_data))

            # Create the y0 (gold policy as JSON string)
            y0 = "<policy>" + json.dumps(gold_policy) + "</policy>"

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification=signature,
                current_specification=signature,
                commonsense_description=COMMONSENSE_DESCRIPTION,
                validity_fn=check_file_policy_validity,
                validity_kwargs={},
                validity_fn_tool_name="check_file_organization_policy_validity",
                validity_fn_tool_description="Check if the file organization policy is valid JSON",
                y0=y0,
                render_task_explanation=render_custom_task_explanation,
                actions=actions,
                msg_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=render_file_policy_results,
                render_msg_fn_txt=render_file_policy_results_txt,
                render_msg_kwargs=["files_data"],
                name="file_organization_custom",
                state_files=[filename] if filename else [],
                files_to_clean=[filename] if filename else [],
                container_ids=[],
                user_expertise_form=self._create_user_expertise_form(),
                parse_solutions_fn=parse_file_organization_solutions,
                initial_shared_state=[
                    (
                        "Files to organize",
                        DisplayElement(
                            input_type="dataframe",
                            value=pd.DataFrame(files_data),
                            hide_index=True,
                        ),
                    ),
                ],
                files_data=files_data,
                render_evaluation_fn=render_file_eval,
                render_evaluation_kwargs=["files_data"],
                render_search_interface_fn=render_file_search_interface,
                render_search_interface_kwargs={"files_data": files_data},
            )

            specs[ix] = spec

        return specs
