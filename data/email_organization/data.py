from typing import List, Optional, Dict, Tuple
import os
import json
import pandas as pd
from collections import defaultdict
from data.dataset import (
    SpecificationCollection,
    LinearFixedSpecification,
    CustomSpecification,
)
from data.actions import get_jupyter_actions
from data.reward import Constraint
from utils.streamlit_types import FormElement, DisplayElement
import streamlit as st

from data.email_organization.reward import (
    check_email_folder_assignment,
    apply_email_policy,
    check_rule_satisfaction,
)
from data.email_organization.streamlit_render import (
    render_email_policy_results,
    render_email_policy_results_txt,
    render_eval as render_email_eval,
)
from data.email_organization.streamlit_search_interface import render_search_interface as render_email_search_interface
from data.email_organization.parser import (
    parse_policy,
    parse_email_organization_solutions,
)
from data.actions import Action
from langchain_core.tools import tool


DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
COMMONSENSE_DESCRIPTION = """An email organization policy is an ordered list of rules that select emails based on their email ID, subject, sender, or content keywords, and then sort them into appropriate folders.

[
  {
    "conditions": "subject_contains \"meeting\" OR email_id \"0\"",
    "folder": "Meetings"
  },
  {
    "conditions": "(subject_contains \"project\" AND from_contains \"boss\") AND NOT content_contains \"cancelled\"",
    "folder": "Active Projects"
  }
]

Syntax (DSL)
- Field conditions: `field_name "value"`
  - `email_id "0"` - Exact match by email ID
  - `subject_contains "keyword"` - Keywords in email subject (normalized matching)
  - `from_contains "keyword"` - Keywords in sender's email address (normalized matching)
  - `content_contains "keyword"` - Keywords in email message body (normalized matching)

- Boolean operators: `AND`, `OR`, `NOT`
- Parentheses: `(...)` for grouping
- Operator precedence: NOT > AND > OR

**Examples:**

Simple OR:
```
subject_contains "meeting" OR email_id "0"
```

AND condition:
```
subject_contains "meeting" AND from_contains "boss"
```

NOT condition:
```
NOT content_contains "spam"
```

Complex nested:
```
(subject_contains "project" AND from_contains "team") OR (email_id "5" AND NOT content_contains "cancelled")
```

**Normalization:**
For keyword matching (subject_contains, from_contains, content_contains), text is:
- Converted to lowercase
- Spaces, dashes (-), and underscores (_) are removed before comparison
- This makes matching robust to formatting variations (e.g., "meeting-notes" matches "meeting notes")

The policy will be applied to the emails, and the result will be evaluated based on whether each email is sorted to the correct folder.
"""

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to organize emails according to a policy.** You will see a collection of unorganized emails, and you need to create a policy that will organize them correctly into the specified folders.

### The task
You will see emails with their email ID, subject, sender, date, and content. Your job is to create a policy (as a JSON object) that sorts emails into the appropriate folders based on their email ID, content, subject, or sender.

The folders you should organize emails into are listed below. The policy will be automatically applied to the emails, and you'll be scored based on how well each email is sorted to the correct folder.
"""

PREDICTION_FMT_INSTRUCTIONS = """Return the policy as a JSON array wrapped in <policy></policy> tags.

**DSL format:**
'<policy>[
    {
        "conditions": "subject_contains \"meeting\" OR email_id \"0\"",
        "folder": "Meetings"
    },
    {
        "conditions": "(subject_contains \"project\" AND from_contains \"boss\") AND NOT content_contains \"cancelled\"",
        "folder": "Active Projects"
    }
]</policy>'

**DSL Syntax:**
- Field conditions: `field_name "value"`
  - `email_id "0"` - Exact match by email ID
  - `subject_contains "keyword"` - Keywords in email subject (normalized matching)
  - `from_contains "keyword"` - Keywords in sender's email address (normalized matching)
  - `content_contains "keyword"` - Keywords in email message body (normalized matching)

- Boolean operators: `AND`, `OR`, `NOT`
- Parentheses: `(...)` for grouping
- Operator precedence: NOT > AND > OR
"""

MSG_FMT_INSTRUCTIONS = "When mentioning specific emails in your communication, wrap their email_id in <email></email> tags, e.g.: '<email>0</email>'. This will display the email to the user for easy reference."


def render_fixed_task_explanation():
    """Render the fixed task explanation for email organization."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


def render_custom_task_explanation(emails_data: List[Dict] = None):
    """Render the custom task explanation for email organization."""
    from data.email_organization.streamlit_render import render_custom_task_explanation as render_custom_explanation
    render_custom_explanation(emails_data=emails_data)


def check_email_policy_validity(
    yhat: str, raise_errors: bool = False
) -> Tuple[bool, dict]:
    """
    Check if an email organization policy is valid.

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

    # Validate that policy is a list
    if not isinstance(policy, list):
        if raise_errors:
            raise ValueError(
                f"Policy must be a list of rule objects, but got {type(policy).__name__}."
            )
        return False, {
            "error": f"Policy must be a list, but got {type(policy).__name__}"
        }

    # Validate that each rule has the required fields
    for i, rule in enumerate(policy):
        if not isinstance(rule, dict):
            if raise_errors:
                raise ValueError(
                    f"Rule at index {i} must be a dictionary, but got {type(rule).__name__}."
                )
            return False, {"error": f"Rule at index {i} must be a dictionary"}
        if "conditions" not in rule:
            if raise_errors:
                raise ValueError(
                    f"Rule at index {i} is missing the 'conditions' field."
                )
            return False, {"error": f"Rule at index {i} is missing 'conditions'"}
        if "folder" not in rule:
            if raise_errors:
                raise ValueError(f"Rule at index {i} is missing the 'folder' field.")
            return False, {"error": f"Rule at index {i} is missing 'folder'"}

    return True, {}


class EmailOrganizationDataset(SpecificationCollection):
    @property
    def dataset_name(self) -> str:
        return "email_organization"

    @property
    def dataset_pretty_name(self) -> str:
        return "Email Organization"

    @property
    def dataset_description(self) -> str:
        return "Organize emails according to a policy that sorts them into appropriate folders."

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
                label="How familiar are you with email organization and filtering?",
                options=[
                    "Not familiar at all",
                    "Somewhat familiar",
                    "Moderately familiar",
                    "Very familiar",
                    "Expert",
                ],
                default="Moderately familiar",
                required=True,
                help="This helps us understand your experience level with email organization",
            )
        ]

    def __init__(
        self,
        dev: bool = False,
        fixed_indexes: Optional[List[int]] = None,
        eval_num_comparisons: int = 5,
        eval_num_items_per_comparison: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)
        self.eval_num_comparisons = eval_num_comparisons
        self.eval_num_items_per_comparison = eval_num_items_per_comparison

        # Find all fixed email CSV files in the assets directory
        emails_dir = os.path.join(DATASET_ROOT, "assets", "emails")
        solutions_dir = os.path.join(DATASET_ROOT, "assets", "solutions")
        if not os.path.exists(emails_dir):
            self._email_files = []
            self.fixed_length = 0
        else:
            email_files = [
                f
                for f in os.listdir(emails_dir)
                if f.endswith(".csv")
                and f != "All Enron Worldwide.csv"
                and f != "_custom.csv"
            ]
            self._email_files = sorted(email_files)
            self.fixed_length = len(self._email_files)
            solution_files = [
                f
                for f in os.listdir(solutions_dir)
                if f.endswith(".csv")
                and f != "All Enron Worldwide.csv"
                and f != "_custom.csv"
            ]
            self._solution_files = sorted(solution_files)

        # Find custom email CSV file in the assets directory
        custom_email_file = os.path.join(
            DATASET_ROOT, "assets", "emails", "_custom.csv"
        )
        if os.path.exists(custom_email_file):
            self._custom_email_file = custom_email_file
            self.custom_length = 1
        else:
            self._custom_email_file = None
            self.custom_length = 0

        # Build extractor lookup
        self._extractor_lookup = {
            "check_email_folder_assignment": check_email_folder_assignment,
            "check_rule_satisfaction": check_rule_satisfaction,
        }

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)

    def _load_email_data(
        self, email_path: str, solution_path: str
    ) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        Load email data from CSV files.

        Returns:
            Tuple of (df, emails_data_no_folder, emails_data)
        """
        df = pd.read_csv(email_path)
        emails_data_no_folder = df.to_dict("records")
        solution_df = pd.read_csv(solution_path)
        df = df.merge(solution_df, on="email_id", how="left")
        emails_data = df.to_dict("records")
        return df, emails_data_no_folder, emails_data

    def _build_folder_mappings(
        self, emails_data: List[Dict]
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Build mappings from email ID to folder and folder to email IDs.

        Returns:
            Tuple of (email_to_folder, folder_to_emails)
        """
        email_to_folder = {}
        folder_to_emails = defaultdict(list)
        for email in emails_data:
            email_id = email.get("email_id", "")
            folder = email.get("folder_pretty", "")
            if email_id != "" and folder != "":
                email_to_folder[email_id] = folder
                folder_to_emails[folder].append(email_id)
        return email_to_folder, folder_to_emails

    def _load_rules_and_create_constraints(
        self, rules_file: str, emails_data: List[Dict], email_to_folder: Dict[str, str]
    ) -> Tuple[List[Dict], Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Load rules from file and create constraint dictionaries.

        Returns:
            Tuple of (features_dicts, valid_rules, edge_cases)
        """
        features_dicts = []
        valid_rules = None
        edge_cases = None

        if os.path.exists(rules_file):
            try:
                with open(rules_file, "r") as f:
                    rules_data = json.load(f)

                valid_rules = rules_data.get("valid_rules", [])
                edge_cases = rules_data.get("edge_cases", [])

                # Create constraints for valid rules (general constraints)
                for rule in valid_rules:
                    rule_name = rule.get("name", "Unnamed rule")
                    rule_conditions = rule.get("conditions", "")
                    rule_folder = rule.get("folder", "")

                    if rule_conditions and rule_folder:
                        features_dicts.append(
                            {
                                "type": "boolean_reward_true",
                                "description": rule_name,
                                "is_hard": False,
                                "is_discoverable": True,
                                "is_minimal": False,
                                "extractor": "check_rule_satisfaction",
                                "extractor_kwargs": {
                                    "rule_name": rule_name,
                                    "rule_conditions": rule_conditions,
                                    "rule_folder": rule_folder,
                                    "emails_data": emails_data,
                                },
                                "none_val": 0,
                            }
                        )

                # Create constraints for edge cases (specific email constraints)
                for edge_case in edge_cases:
                    email_id = edge_case.get("email_id", "")
                    folder = edge_case.get("folder", "")

                    if email_id and folder:
                        features_dicts.append(
                            {
                                "type": "boolean_reward_true",
                                "description": f"Email ID {email_id} should be sorted to folder '{folder}'",
                                "is_hard": False,
                                "is_discoverable": True,
                                "is_minimal": False,
                                "extractor": "check_email_folder_assignment",
                                "extractor_kwargs": {
                                    "email_id": email_id,
                                    "correct_folder": folder,
                                    "emails_data": emails_data,
                                },
                                "none_val": 0,
                            }
                        )
            except Exception as e:
                print(f"Warning: Failed to load rules from {rules_file}: {e}")
                print("Falling back to individual email constraints.")
                features_dicts = []

        # Fall back to individual email constraints if no rules file or loading failed
        if not features_dicts:
            for email in emails_data:
                email_id = email.get("email_id", "")
                if email_id == "":
                    continue

                expected_folder = email_to_folder.get(email_id, "")
                if expected_folder:
                    features_dicts.append(
                        {
                            "type": "boolean_reward_true",
                            "description": f"Email ID {email_id} should be sorted to folder '{expected_folder}'",
                            "is_hard": False,
                            "is_discoverable": True,
                            "is_minimal": False,
                            "extractor": "check_email_folder_assignment",
                            "extractor_kwargs": {
                                "email_id": email_id,
                                "correct_folder": expected_folder,
                                "emails_data": emails_data,
                            },
                            "none_val": 0,
                        }
                    )

        return features_dicts, valid_rules, edge_cases

    def _create_apply_policy_tool(self, emails_data: List[Dict]) -> Action:
        """
        Create the apply_policy_and_see_results tool.

        Returns:
            Action object for the tool
        """

        @tool(parse_docstring=True)
        def apply_policy_and_see_results(policy_json: str) -> str:
            """
            Apply an email organization policy to the emails and see the results.
            This tool shows which emails are sorted into which folders after applying the policy.

            Args:
                policy_json: A JSON string representing the policy. The policy should be a list of rule objects,
                             each with "conditions" (DSL string) and "folder" (folder name).
            """
            try:
                policy = parse_policy(policy_json, raise_errors=False)

                if policy is None:
                    return "Error: Could not parse the policy. Please provide a valid JSON array of rule objects, optionally wrapped in <policy></policy> tags."

                if not isinstance(policy, list):
                    return f"Error: Policy must be a list of rule objects, but got {type(policy).__name__}."

                for i, rule in enumerate(policy):
                    if not isinstance(rule, dict):
                        return f"Error: Rule at index {i} must be a dictionary, but got {type(rule).__name__}."
                    if "conditions" not in rule:
                        return f"Error: Rule at index {i} is missing the 'conditions' field."
                    if "folder" not in rule:
                        return (
                            f"Error: Rule at index {i} is missing the 'folder' field."
                        )

                organized = apply_email_policy(emails_data, policy)

                result_lines = []
                result_lines.append(
                    "Email organization results after applying the policy:\n"
                )
                result_lines.append("=" * 80 + "\n")

                sorted_folders = sorted(organized.keys())

                for folder in sorted_folders:
                    emails = organized[folder]
                    if not emails:
                        continue

                    result_lines.append(f"\n📁 {folder} ({len(emails)} email(s)):")
                    result_lines.append("-" * 80)

                    for email in emails:
                        email_id = email.get("email_id", "")
                        subject = email.get("subject", "")
                        from_addr = email.get("from", "")
                        if len(subject) > 60:
                            subject = subject[:57] + "..."
                        if len(from_addr) > 40:
                            from_addr = from_addr[:37] + "..."

                        result_lines.append(f"  • Email ID: {email_id}")
                        result_lines.append(f"    Subject: {subject}")
                        result_lines.append(f"    From: {from_addr}")
                        result_lines.append("")

                if "Unsorted" in organized and organized["Unsorted"]:
                    result_lines.append(
                        f"\n⚠️  Unsorted ({len(organized['Unsorted'])} email(s)):"
                    )
                    result_lines.append("-" * 80)
                    for email in organized["Unsorted"]:
                        email_id = email.get("email_id", "")
                        result_lines.append(f"  • Email ID: {email_id}")

                return "\n".join(result_lines)

            except Exception as e:
                return f"Error applying policy: {str(e)}"

        return Action(
            fn=apply_policy_and_see_results,
            is_public=True,
            is_human=False,
            name="Apply email policy and see results",
        )

    def _create_gold_policy(
        self,
        valid_rules: Optional[List[Dict]],
        edge_cases: Optional[List[Dict]],
        unique_folders: List[str],
        emails_data: List[Dict],
    ) -> List[Dict]:
        """
        Create gold policy from saved rules or synthetic policy.

        Returns:
            List of rule dicts
        """
        if valid_rules is not None and edge_cases is not None:
            gold_policy = []
            for rule in valid_rules:
                gold_policy.append(
                    {
                        "conditions": rule.get("conditions", ""),
                        "folder": rule.get("folder", ""),
                    }
                )
            for edge_case in edge_cases:
                email_id = edge_case.get("email_id", "")
                folder = edge_case.get("folder", "")
                if email_id and folder:
                    gold_policy.append(
                        {
                            "conditions": f'email_id "{email_id}"',
                            "folder": folder,
                        }
                    )
            return gold_policy
        else:
            gold_policy = []
            for folder in unique_folders:
                folder_emails = [
                    e for e in emails_data if e.get("folder_pretty") == folder
                ]
                if folder_emails:
                    folder_rules = _create_synthetic_policy_for_folder(
                        folder, folder_emails, emails_data
                    )
                    gold_policy.extend(folder_rules)
            return gold_policy

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, LinearFixedSpecification]:
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            if ix >= len(self._email_files):
                continue

            email_file = self._email_files[ix]
            solution_file = self._solution_files[ix]
            email_path = os.path.join(DATASET_ROOT, "assets", "emails", email_file)
            solution_path = os.path.join(
                DATASET_ROOT, "assets", "solutions", solution_file
            )

            # Load the emails
            df, emails_data_no_folder, emails_data = self._load_email_data(
                email_path, solution_path
            )

            # Build mappings
            email_to_folder, folder_to_emails = self._build_folder_mappings(emails_data)
            unique_folders = sorted(folder_to_emails.keys())

            # Load rules and create constraints
            rules_file = os.path.join(
                DATASET_ROOT,
                "assets",
                "rules",
                email_file.replace(".csv", "_rules.json"),
            )
            features_dicts, valid_rules, edge_cases = (
                self._load_rules_and_create_constraints(
                    rules_file, emails_data, email_to_folder
                )
            )

            # Convert to Constraint objects
            features: List[Constraint] = [
                Constraint.from_dict(fd, extractor_lookup=self._extractor_lookup)
                for fd in features_dicts
            ]

            # Weights: equal importance for all constraints
            weights: List[float] = [1.0] * len(features)

            # Build specification text
            signature = "Looking for a set of email sorting rules which sorts emails into the following folders:\n\n"
            for folder in unique_folders:
                signature += f"- {folder}\n"

            # Create ls_output to describe the email CSV file
            # Get column descriptions from the DataFrame
            email_columns = {
                "email_id": "Unique identifier for the email (index-based)",
                "subject": "Subject line of the email",
                "from": "Sender's email address",
                "date": "Date when the email was sent",
                "message": "Full content/body of the email message",
            }

            # The filename should be relative to the root_dir (assets directory)
            email_file_path = os.path.join("emails", email_file)

            ls_output = [
                {
                    "filename": email_file_path,
                    "description": f"CSV file containing {len(df)} emails to organize. Each row represents one email with its metadata and content. Use pandas to load and investigate this file (e.g., `email_df = pd.read_csv('{email_file_path}')`).",
                    "columns": email_columns,
                }
            ]

            # Get Jupyter actions with ls_output pointing to the emails directory
            filename, actions = get_jupyter_actions(
                docker_image=None,
                docker_container_id=None,
                ls_output=ls_output,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Add the apply policy tool
            actions.append(self._create_apply_policy_tool(emails_data))

            # Create gold policy
            gold_policy = self._create_gold_policy(
                valid_rules, edge_cases, unique_folders, emails_data
            )

            ystar = "<policy>" + json.dumps(gold_policy) + "</policy>"
            spec = LinearFixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                initial_specification=signature,
                commonsense_description=COMMONSENSE_DESCRIPTION,
                features=features,
                weights=weights,
                parse_y_fn=parse_policy,
                validity_fn_tool_name="check_email_organization_policy_validity",
                validity_fn_tool_description="Check if the email organization policy is valid JSON",
                reward_fn_tool_name="score_email_organization_policy",
                reward_fn_tool_description="Score the email organization policy based on folder assignment accuracy",
                ystar=ystar,
                render_task_explanation=render_fixed_task_explanation,
                actions=actions,
                msg_fmt_instructions=MSG_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=render_email_policy_results,
                render_msg_fn_txt=render_email_policy_results_txt,
                render_msg_kwargs=["emails_data"],
                name=f"email_organization_{email_file.replace('.csv', '')}",
                state_files=[filename] if filename else [],
                files_to_clean=[filename] if filename else [],
                container_ids=[],
                user_expertise_form=self._create_user_expertise_form(),
                parse_solutions_fn=parse_email_organization_solutions,
                initial_shared_state=[
                    (
                        "Emails to organize",
                        DisplayElement(
                            input_type="dataframe",
                            value=pd.DataFrame(emails_data_no_folder),
                            hide_index=True,
                        ),
                    ),
                ],
                emails_data=emails_data,
            )

            specs[ix] = spec

        return specs

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        if indexes is None:
            return {}

        # Check if custom email file exists
        if self._custom_email_file is None:
            return {}

        specs = {}
        for ix in indexes:
            if ix >= self.custom_length:
                continue

            # Use the custom email file
            email_file = "_custom.csv"
            solution_file = "_custom.csv"
            email_path = os.path.join(DATASET_ROOT, "assets", "emails", email_file)
            solution_path = os.path.join(
                DATASET_ROOT, "assets", "solutions", solution_file
            )

            # Load the emails
            df, emails_data_no_folder, emails_data = self._load_email_data(
                email_path, solution_path
            )

            # Build mappings
            email_to_folder, folder_to_emails = self._build_folder_mappings(emails_data)
            unique_folders = sorted(folder_to_emails.keys())

            # Load rules (for gold policy creation, not for constraints in custom specs)
            rules_file = os.path.join(
                DATASET_ROOT,
                "assets",
                "rules",
                "_custom_rules.json",
            )
            _, valid_rules, edge_cases = self._load_rules_and_create_constraints(
                rules_file, emails_data, email_to_folder
            )

            # Build specification text
            signature = "Set up rules to sort the emails into tidy folders."

            # Create ls_output to describe the email CSV file
            # Get column descriptions from the DataFrame
            email_columns = {
                "email_id": "Unique identifier for the email (index-based)",
                "subject": "Subject line of the email",
                "from": "Sender's email address",
                "date": "Date when the email was sent",
                "message": "Full content/body of the email message",
            }

            # The filename should be relative to the root_dir (assets directory)
            email_file_path = os.path.join("emails", email_file)

            ls_output = [
                {
                    "filename": email_file_path,
                    "description": f"CSV file containing {len(df)} emails to organize. Each row represents one email with its metadata and content. Use pandas to load and investigate this file (e.g., `email_df = pd.read_csv('{email_file_path}')`).",
                    "columns": email_columns,
                }
            ]

            # Get Jupyter actions with ls_output pointing to the emails directory
            filename, actions = get_jupyter_actions(
                docker_image=None,
                docker_container_id=None,
                ls_output=ls_output,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Add the apply policy tool
            actions.append(self._create_apply_policy_tool(emails_data))

            # Create gold policy
            gold_policy = self._create_gold_policy(
                valid_rules, edge_cases, unique_folders, emails_data
            )

            y0 = "<policy>" + json.dumps(gold_policy) + "</policy>"
            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification=signature,
                current_specification=signature,
                commonsense_description=COMMONSENSE_DESCRIPTION,
                validity_fn=check_email_policy_validity,
                validity_kwargs={},
                validity_fn_tool_name="check_email_organization_policy_validity",
                validity_fn_tool_description="Check if the email organization policy is valid JSON",
                y0=y0,
                render_task_explanation=render_custom_task_explanation,
                actions=actions,
                msg_fmt_instructions=MSG_FMT_INSTRUCTIONS,
                prediction_fmt_instructions=PREDICTION_FMT_INSTRUCTIONS,
                render_msg_fn=render_email_policy_results,
                render_msg_fn_txt=render_email_policy_results_txt,
                render_msg_kwargs=["emails_data"],
                name="email_organization_custom",
                state_files=[filename] if filename else [],
                files_to_clean=[filename] if filename else [],
                container_ids=[],
                user_expertise_form=self._create_user_expertise_form(),
                parse_solutions_fn=parse_email_organization_solutions,
                initial_shared_state=[
                    (
                        "Emails to organize",
                        DisplayElement(
                            input_type="dataframe",
                            value=pd.DataFrame(emails_data_no_folder),
                            hide_index=True,
                        ),
                    ),
                ],
                emails_data=emails_data,
                render_evaluation_fn=render_email_eval,
                render_evaluation_kwargs={
                    "y0": y0,
                    "emails_data": emails_data,
                    "num_comparisons": self.eval_num_comparisons,
                    "num_items_per_comparison": self.eval_num_items_per_comparison,
                },
                render_search_interface_fn=render_email_search_interface,
                render_search_interface_kwargs={"emails_data": emails_data},
            )

            specs[ix] = spec

        return specs


def _create_synthetic_policy_for_folder(
    folder: str,
    folder_emails: List[Dict],
    all_emails: List[Dict],
) -> List[Dict]:
    """
    Create synthetic policy rules that exactly match each email by ID.
    Returns a list of rule dicts using DSL format:
    [{"conditions": "email_id \"0\"", "folder": "..."}, ...]
    Creates one rule per email using email_id for exact matching.
    """
    rules = []

    # Create a rule for each email individually using email_id
    for email in folder_emails:
        email_id = email.get("email_id", "")
        if email_id == "":
            continue

        # Use DSL format: email_id "value"
        conditions = f'email_id "{email_id}"'

        # Create rule in DSL format
        rules.append({"conditions": conditions, "folder": folder})

    return rules
