from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import json
import random
from datetime import datetime
from utils.misc import parse_for_answer_tags, replace_tags_with_link
from data.email_organization.reward import apply_email_policy
from data.email_organization.parser import parse_policy

# Session state key prefixes used by this render module that should be cleared between rounds
RENDER_SESSION_STATE_KEY_PREFIXES = [
    "email_comparison_",
]


def render_email_policy_results(msg: str, emails_data: List[Dict], show_correct_folder: bool = True) -> None:
    """
    Render the email policy results grouped by folder, showing emails within each folder.

    Args:
        msg: The message containing the policy
        emails_data: List of email dictionaries
        show_correct_folder: Whether to show the correct folder column (for fixed specs). 
                            Set to False for custom specs where there's no "correct" folder.
    """
    # Build mapping from email_id to full email data and folder
    email_to_data = {}  
    email_to_folder = {}
    for email in emails_data:
        email_id = str(email.get("email_id", ""))
        if email_id:
            email_to_data[email_id] = email
            folder = email.get("folder_pretty", "")
            if folder:
                email_to_folder[email_id] = folder

    # Parse email tags from the message
    mentioned_emails = parse_for_answer_tags(
        msg, keyword="email", return_all=True, return_none_if_not_found=True
    )

    # Extract policy and find its position in the message
    policy_str, start_end = parse_for_answer_tags(
        msg, keyword="policy", return_start_end=True, return_none_if_not_found=True
    )

    # Generate unique ID for this message to avoid conflicts when multiple messages are rendered
    message_hash = str(hash(msg))[:8]
    unique_id = f"mentioned-emails-{message_hash}"

    # Display message with email tags replaced by links
    if start_end is not None:
        # Message has policy tags - display everything except the policy
        before_policy = msg[: start_end[0]]
        after_policy = msg[start_end[1] :]
        message_without_policy = before_policy + after_policy
        if message_without_policy.strip():
            display_msg = replace_tags_with_link(
                message_without_policy, tag="email", href=f"#{unique_id}"
            )
            st.markdown(display_msg, unsafe_allow_html=True)
    else:
        # No policy tags found - display the full message
        display_msg = replace_tags_with_link(msg, tag="email", href=f"#{unique_id}")
        st.markdown(display_msg, unsafe_allow_html=True)

    # Display mentioned emails with their full information
    if mentioned_emails:
        email_info_list = []
        for email_group in mentioned_emails:
            # Handle comma-separated email IDs
            email_ids = [eid.strip() for eid in email_group.split(",") if eid.strip()]
            for email_id in email_ids:
                if email_id in email_to_data:
                    email_data = email_to_data[email_id]
                    email_info = {
                        "email_id": email_id,
                        "to": email_data.get("to", ""),
                        "from": email_data.get("from", ""),
                        "subject": email_data.get("subject", ""),
                        "message": email_data.get("message", "")[:500]
                        + (
                            "..."
                            if len(email_data.get("message", "")) > 500
                            else ""
                        ),
                    }
                    # Only add correct_folder for fixed specs
                    if show_correct_folder:
                        email_info["correct_folder"] = email_to_folder.get(email_id, "Unknown")
                    email_info_list.append(email_info)

        if email_info_list:
            with st.expander("Email Information", expanded=True):
                st.markdown(f'<div id="{unique_id}"></div>', unsafe_allow_html=True)
                df = pd.DataFrame(email_info_list)
                st.dataframe(df, hide_index=True)

    # Parse the policy from the message
    policy = parse_policy(msg, raise_errors=False)

    if policy is None:
        # If we can't parse the policy, we've already displayed the message
        return

    # Reorder policy to prioritize email_id direct matches (edge cases)
    # These should be checked first
    priority_rules = []
    regular_rules = []

    for rule in policy:
        conditions = rule.get("conditions", "")
        # Check if this rule has an email_id condition (exact match)
        if "email_id" in conditions:
            priority_rules.append(rule)
        else:
            regular_rules.append(rule)

    # Combine: priority rules first, then regular rules
    reordered_policy = priority_rules + regular_rules

    # Apply the policy to all emails
    try:
        organized = apply_email_policy(emails_data, reordered_policy)
    except Exception as e:
        st.error(f"Error applying policy: {str(e)}")
        return

    # Display organized emails grouped by folder
    if not organized:
        return

    with st.container(border=True, height=700):
        st.markdown("### 📁 Organized Emails")

        # Create tabs for each folder
        folder_names = sorted(organized.keys())
        if len(folder_names) > 0:
            tabs = st.tabs(folder_names)

            for tab, folder_name in zip(tabs, folder_names):
                with tab:
                    emails_in_folder = organized[folder_name]

                    if not emails_in_folder:
                        st.info(f"No emails in {folder_name}")
                        continue

                    # Sort emails by date if available
                    try:
                        emails_in_folder = sorted(
                            emails_in_folder,
                            key=lambda x: x.get("date", ""),
                            reverse=True,
                        )
                    except Exception:
                        pass

                    for email in emails_in_folder:
                        email_id = str(email.get("email_id", ""))
                        from_addr = email.get("from", "")
                        to_addr = email.get("to", "")
                        subject = email.get("subject", "")
                        date = email.get("date", "")
                        message = email.get("message", "")

                        # Build the main part of the expander label
                        main_part = (
                            f"{subject}"
                            if subject
                            else f"Email {email_id}"
                        )

                        # Add date suffix if available
                        date_suffix = ""
                        if date:
                            try:
                                # Try to parse and format the date
                                date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                                date_suffix = f" ({date_obj.strftime('%b %d, %Y')})"
                            except Exception:
                                date_suffix = f" ({date})"

                        expander_label = main_part + date_suffix

                        with st.expander(expander_label, expanded=False):
                            # Email header information
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**From:** {from_addr}")
                                if to_addr:
                                    st.markdown(f"**To:** {to_addr}")
                            with col2:
                                st.markdown(f"**Email ID:** `{email_id}`")
                                st.markdown(f"**Date:** {date}")

                            # Email subject
                            if subject:
                                st.markdown(f"**Subject:** {subject}")

                            # Email body - use markdown with reduced whitespace
                            st.markdown("**Message:**")
                            # Remove excessive whitespace from message
                            message_clean = "\n".join(
                                line.rstrip() for line in message.split("\n")
                            )
                            st.markdown(
                                f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>",
                                unsafe_allow_html=True,
                            )


def render_email_policy_results_txt(msg: str, emails_data: List[Dict], show_correct_folder: bool = True) -> str:
    """
    Render the email policy results as JSON organized by folder, with emails listed under each folder.

    Args:
        msg: The message containing the policy
        emails_data: List of email dictionaries
        show_correct_folder: Whether to show the correct folder column (for fixed specs). 
                            Set to False for custom specs where there's no "correct" folder.

    Returns:
        JSON string with folders as keys and arrays of email objects as values
    """
    # Build mapping from email_id to full email data and folder
    email_to_data = {}
    email_to_folder = {}
    for email in emails_data:
        email_id = str(email.get("email_id", ""))
        if email_id:
            email_to_data[email_id] = email
            folder = email.get("folder_pretty", "")
            if folder:
                email_to_folder[email_id] = folder

    # Parse email tags from the message
    mentioned_emails = parse_for_answer_tags(
        msg, keyword="email", return_all=True, return_none_if_not_found=True
    )

    # Helper function to build email info list
    def build_email_info_list():
        email_info_list = []
        for email_group in mentioned_emails:
            # Handle comma-separated email IDs
            email_ids = [eid.strip() for eid in email_group.split(",") if eid.strip()]
            for email_id in email_ids:
                if email_id in email_to_data:
                    email_data = email_to_data[email_id]
                    email_info = {
                        "email_id": email_id,
                        "to": email_data.get("to", ""),
                        "from": email_data.get("from", ""),
                        "subject": email_data.get("subject", ""),
                        "message": email_data.get("message", "")[:500]
                        + (
                            "..."
                            if len(email_data.get("message", "")) > 500
                            else ""
                        ),
                    }
                    # Only add correct_folder for fixed specs
                    if show_correct_folder:
                        email_info["correct_folder"] = email_to_folder.get(email_id, "Unknown")
                    email_info_list.append(email_info)
        return email_info_list

    # Parse the policy from the message
    policy = parse_policy(msg, raise_errors=False)

    if policy is None:
        # If we can't parse the policy, return the raw message with email info if available
        if mentioned_emails:
            email_info_list = build_email_info_list()
            if email_info_list:
                return (
                    msg
                    + "\n\n------- Information about mentioned emails ----------\n\n"
                    + json.dumps(email_info_list, indent=2)
                )

        return msg

    # Apply the policy to all emails
    try:
        organized = apply_email_policy(emails_data, policy)
    except Exception as e:
        # Return error message as JSON
        error_msg = (
            msg
            + "\n\n-----------Inbox state after running the policy:-----------\n\n"
            + json.dumps({"error": f"Error applying policy: {str(e)}"})
        )
        # Add email info if available
        if mentioned_emails:
            email_info_list = build_email_info_list()
            if email_info_list:
                error_msg += (
                    "\n\n------- Information about mentioned emails ----------\n\n"
                    + json.dumps(email_info_list, indent=2)
                )
        return error_msg

    # Truncate each message to 500 characters
    for folder, emails in organized.items():
        for email in emails:
            email["message"] = email["message"][:500] + "..."
            email.pop("folder_pretty", None)

    result = (
        msg
        + "\n\n-----------Inbox state after running the policy:-----------\n\n"
        + json.dumps(organized)
    )

    # Add email info if available
    if mentioned_emails:
        email_info_list = build_email_info_list()
        if email_info_list:
            result += (
                "\n\n------- Information about mentioned emails ----------\n\n"
                + json.dumps(email_info_list, indent=2)
            )

    return result


def render_eval(
    *,
    final_prediction: str,
    y0: str,
    emails_data: List[Dict],
    num_comparisons: int = 5,
    num_items_per_comparison: int = 5,
) -> Tuple[bool, Optional[Dict]]:
    """
    Render evaluation UI comparing final_prediction against y0.
    Shows side-by-side email organization comparisons and collects user preferences.

    Args:
        final_prediction: The assistant's policy prediction
        y0: The gold/reference policy
        emails_data: List of email dictionaries
        num_comparisons: Number of comparison iterations to run
        num_items_per_comparison: Number of items to sample per comparison

    Returns:
        Tuple of (completed: bool, feedback: Optional[Dict])
    """
    # Initialize session state for tracking comparisons
    if "final_evaluation" not in st.session_state.form_results:
        st.session_state.form_results["final_evaluation"] = {}
    if "comparison_results" not in st.session_state.form_results["final_evaluation"]:
        st.session_state.form_results["final_evaluation"]["comparison_results"] = []

    comparison_results = st.session_state.form_results["final_evaluation"][
        "comparison_results"
    ]

    # Parse both policies
    pred_policy = parse_policy(final_prediction, raise_errors=False)
    y0_policy = parse_policy(y0, raise_errors=False)

    if pred_policy is None:
        st.error("Could not parse final_prediction policy.")
        return False, None
    if y0_policy is None:
        st.error("Could not parse y0 policy.")
        return False, None

    # Apply both policies to get organization
    try:
        pred_organized = apply_email_policy(emails_data, pred_policy)
        y0_organized = apply_email_policy(emails_data, y0_policy)
    except Exception as e:
        st.error(f"Error applying policies: {str(e)}")
        return False, None

    # Build email_id to email mapping
    email_id_to_email = {
        str(e.get("email_id", "")): e for e in emails_data if e.get("email_id", "")
    }
    all_email_ids = list(email_id_to_email.keys())

    # Run comparisons
    current_comparison = len(comparison_results)

    if current_comparison >= num_comparisons:
        # All comparisons done
        return True, {"comparison_results": comparison_results}

    # Sample emails for this comparison (stable across reruns)
    sample_key = f"email_comparison_{current_comparison}_sample"
    if sample_key not in st.session_state:
        if len(all_email_ids) > num_items_per_comparison:
            sampled_ids = random.sample(all_email_ids, num_items_per_comparison)
        else:
            sampled_ids = all_email_ids
        st.session_state[sample_key] = sampled_ids
    sampled_ids = st.session_state[sample_key]

    # Randomize which policy goes on which side (stable across reruns)
    side_key = f"email_comparison_{current_comparison}_side_assignment"
    if side_key not in st.session_state:
        # Randomly assign: True means pred on left, False means y0 on left
        st.session_state[side_key] = random.choice([True, False])
    pred_on_left = st.session_state[side_key]

    # Get folders for sampled emails in both policies
    pred_folders = {}
    y0_folders = {}

    for email_id in sampled_ids:
        # Find which folder this email is in for pred policy
        for folder, emails in pred_organized.items():
            if any(str(e.get("email_id", "")) == email_id for e in emails):
                pred_folders[email_id] = folder
                break

        # Find which folder this email is in for y0 policy
        for folder, emails in y0_organized.items():
            if any(str(e.get("email_id", "")) == email_id for e in emails):
                y0_folders[email_id] = folder
                break

    # Assign folders to left/right based on randomization
    left_folders = pred_folders if pred_on_left else y0_folders
    right_folders = y0_folders if pred_on_left else pred_folders

    # Group emails by folder for both sides
    left_emails_by_folder = {}
    right_emails_by_folder = {}
    
    for email_id in sampled_ids:
        # Left side grouping
        left_folder = left_folders.get(email_id, "Uncategorized")
        if left_folder not in left_emails_by_folder:
            left_emails_by_folder[left_folder] = []
        left_emails_by_folder[left_folder].append(email_id)
        
        # Right side grouping
        right_folder = right_folders.get(email_id, "Uncategorized")
        if right_folder not in right_emails_by_folder:
            right_emails_by_folder[right_folder] = []
        right_emails_by_folder[right_folder].append(email_id)

    # Display comparison
    st.markdown(f"### Comparison {current_comparison + 1} of {num_comparisons}")
    st.markdown("Compare how emails are organized in Policy A vs Policy B:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Policy A")
        # Display folders in sorted order
        for folder in sorted(left_emails_by_folder.keys()):
            with st.container(border=True):
                st.markdown(f"**📁 {folder}**")
                for email_id in left_emails_by_folder[folder]:
                    email = email_id_to_email.get(email_id, {})
                    from_addr = email.get("from", "")
                    to_addr = email.get("to", "")
                    subject = email.get("subject", f"Email {email_id}")
                    date = email.get("date", "")
                    message = email.get("message", "")

                    # Build the expander label (matching render_email_policy_results style)
                    main_part = subject if subject else f"Email {email_id}"
                    date_suffix = ""
                    if date:
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                            date_suffix = f" ({date_obj.strftime('%b %d, %Y')})"
                        except Exception:
                            date_suffix = f" ({date})"
                    
                    expander_label = f"{main_part}{date_suffix}"

                    with st.expander(expander_label, expanded=False):
                        # Email header information
                        col1a, col1b = st.columns([3, 1])
                        with col1a:
                            st.markdown(f"**From:** {from_addr}")
                            if to_addr:
                                st.markdown(f"**To:** {to_addr}")
                        with col1b:
                            st.markdown(f"**Email ID:** `{email_id}`")
                            st.markdown(f"**Date:** {date}")

                        # Email subject
                        if subject:
                            st.markdown(f"**Subject:** {subject}")

                        # Email body
                        st.markdown("**Message:**")
                        message_clean = "\n".join(
                            line.rstrip() for line in message.split("\n")
                        )
                        st.markdown(
                            f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>",
                            unsafe_allow_html=True,
                        )

    with col2:
        st.markdown("#### Policy B")
        # Display folders in sorted order
        for folder in sorted(right_emails_by_folder.keys()):
            with st.container(border=True):
                st.markdown(f"**📁 {folder}**")
                for email_id in right_emails_by_folder[folder]:
                    email = email_id_to_email.get(email_id, {})
                    from_addr = email.get("from", "")
                    to_addr = email.get("to", "")
                    subject = email.get("subject", f"Email {email_id}")
                    date = email.get("date", "")
                    message = email.get("message", "")

                    # Build the expander label (matching render_email_policy_results style)
                    main_part = subject if subject else f"Email {email_id}"
                    date_suffix = ""
                    if date:
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                            date_suffix = f" ({date_obj.strftime('%b %d, %Y')})"
                        except Exception:
                            date_suffix = f" ({date})"
                    
                    expander_label = f"{main_part}{date_suffix}"

                    with st.expander(expander_label, expanded=False):
                        # Email header information
                        col2a, col2b = st.columns([3, 1])
                        with col2a:
                            st.markdown(f"**From:** {from_addr}")
                            if to_addr:
                                st.markdown(f"**To:** {to_addr}")
                        with col2b:
                            st.markdown(f"**Email ID:** `{email_id}`")
                            st.markdown(f"**Date:** {date}")

                        # Email subject
                        if subject:
                            st.markdown(f"**Subject:** {subject}")

                        # Email body
                        st.markdown("**Message:**")
                        message_clean = "\n".join(
                            line.rstrip() for line in message.split("\n")
                        )
                        st.markdown(
                            f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>",
                            unsafe_allow_html=True,
                        )

    # Collect preference
    preference = st.radio(
        "Which organization do you prefer?",
        ["-", "A", "neutral", "B"],
        key=f"comparison_{current_comparison}",
    )

    if st.button("Submit", key=f"submit_comparison_{current_comparison}"):
        # Store preference relative to which policy is which
        # If pred_on_left: A=pred, B=y0. If not: A=y0, B=pred
        # Convert preference to always be relative to pred vs y0
        if pred_on_left:
            # A is pred, B is y0, so preference is already correct
            pred_preference = preference
        else:
            # A is y0, B is pred, so flip the preference
            if preference == "A":
                pred_preference = "B"
            elif preference == "B":
                pred_preference = "A"
            else:
                pred_preference = "neutral"
        
        comparison_results.append(
            {
                "comparison_index": current_comparison,
                "sampled_email_ids": sampled_ids,
                "preference": pred_preference,
                "pred_on_left": pred_on_left,  # Store for reference
            }
        )
        st.session_state.form_results["final_evaluation"]["comparison_results"] = (
            comparison_results
        )
        st.rerun()

    return False, None


def _render_unsorted_inbox(emails: List[Dict]):
    """
    Render an unsorted inbox showing emails in a simple list format.
    Matches the email rendering format used in render_email_policy_results.
    
    Args:
        emails: List of email dictionaries
    """
    st.markdown("### 📥 Your Inbox")
    st.markdown("These are your unorganized emails:")
    
    for email in emails:
        email_id = str(email.get("email_id", ""))
        from_addr = email.get("from", "")
        to_addr = email.get("to", "")
        subject = email.get("subject", "")
        date = email.get("date", "")
        message = email.get("message", "")

        # Build the main part of the expander label
        main_part = (
            f"{subject}"
            if subject
            else f"Email {email_id}"
        )

        # Add date suffix if available
        date_suffix = ""
        if date:
            try:
                # Try to parse and format the date
                date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                date_suffix = f" ({date_obj.strftime('%b %d, %Y')})"
            except Exception:
                date_suffix = f" ({date})"

        expander_label = main_part + date_suffix

        with st.expander(expander_label, expanded=False):
            # Email header information
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**From:** {from_addr}")
                if to_addr:
                    st.markdown(f"**To:** {to_addr}")
            with col2:
                st.markdown(f"**Email ID:** `{email_id}`")
                st.markdown(f"**Date:** {date}")

            # Email subject
            if subject:
                st.markdown(f"**Subject:** {subject}")

            # Email body - use markdown with reduced whitespace
            st.markdown("**Message:**")
            # Remove excessive whitespace from message
            message_clean = "\n".join(
                line.rstrip() for line in message.split("\n")
            )
            st.markdown(
                f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>",
                unsafe_allow_html=True,
            )


def render_custom_task_explanation(emails_data: List[Dict] = None):
    """
    Render the custom task explanation for email organization.
    Shows what an inbox is, what parts of an email are, and an example of a policy.

    Args:
        emails_data: Optional list of email dictionaries for examples. If None, creates sample data.
    """
    st.markdown("### What you need to prompt the assistant to do")
    st.markdown(
        "In this task, **your goal is to get the assistant to organize your emails.** To do so, the assistant will write some rules that automatically filter your emails into folders. "
    )

    # Create example emails if none provided
    if emails_data is None:
        example_emails = [
            {
                "email_id": "0",
                "from": "boss@company.com",
                "to": "you@company.com",
                "subject": "Meeting tomorrow at 3pm",
                "date": "2024-01-15 10:30:00",
                "message": "Hi, let's meet tomorrow at 3pm to discuss the project.",
                "folder_pretty": "Meetings",
            },
            {
                "email_id": "1",
                "from": "team@company.com",
                "to": "you@company.com",
                "subject": "Project update",
                "date": "2024-01-15 11:00:00",
                "message": "Here's the latest update on our active project.",
                "folder_pretty": "Active Projects",
            },
            {
                "email_id": "2",
                "from": "newsletter@example.com",
                "to": "you@company.com",
                "subject": "Weekly newsletter",
                "date": "2024-01-15 12:00:00",
                "message": "Check out this week's newsletter with all the latest news.",
                "folder_pretty": "Newsletters",
            },
        ]
    else:
        # Use first few emails from provided data
        example_emails = emails_data[:3] if len(emails_data) >= 3 else emails_data

    
    st.markdown("### What are the parts of an email?")
    st.markdown(
        "Each email has several parts you can use to create organization rules:"
    )
    st.markdown(
        "- **From:** The sender's email address\n"
        "- **To:** The recipient's email address\n"
        "- **Subject:** The subject line of the email\n"
        "- **Date:** When the email was sent\n"
        "- **Message:** The body/content of the email\n"
        "- **Email ID:** A unique identifier for each email"
    )
    st.markdown("For example, you might want to ask the assistant to filter all emails from your boss at `boss@company.com` into a folder titled `Important`.")
    
    with st.container(border=True):
        _render_unsorted_inbox(example_emails)


    st.markdown("### Example: Organizing emails with a policy")
    st.markdown(
        "Here's an example of how a policy can organize your emails into folders. "
        "The assistant will create a policy that looks for patterns in your emails and sorts them accordingly."
    )

    with st.container(border=True):
        example_policy = [
            {
                "conditions": 'subject_contains "meeting" OR email_id "0"',
                "folder": "Meetings",
            },
            {
                "conditions": 'subject_contains "project" AND from_contains "team"',
                "folder": "Active Projects",
            },
        ]
        example_msg = f"Here's my email organization policy:\n\n<policy>{json.dumps(example_policy)}</policy>"

        st.info(
            "*Example:* An email organization policy that sorts emails into folders"
        )
        render_email_policy_results(example_msg, example_emails, show_correct_folder=False)

    st.markdown(
        "Think about how you want to organize your emails. The assistant should create a policy that sorts "
        "emails into meaningful folders based on patterns you identify in the email content, subjects, or senders."
    )
