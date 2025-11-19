from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import json
import re
import random
from datetime import datetime
from utils.misc import parse_for_answer_tags, replace_tags_with_link
from data.email_organization.reward import apply_email_policy
from data.email_organization.parser import parse_policy


def render_email_policy_results(msg: str, emails_data: List[Dict]) -> None:
    """
    Render the email policy results grouped by folder, showing emails within each folder.

    Args:
        msg: The message containing the policy
        emails_data: List of email dictionaries
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

        # Replace email tags with links
        if mentioned_emails:
            message_without_policy = replace_tags_with_link(
                message_without_policy, "email", f"#{unique_id}"
            )

        if message_without_policy.strip():
            st.markdown(message_without_policy, unsafe_allow_html=True)
    else:
        # No policy tags found - display the full message with email tags replaced
        display_msg = msg
        if mentioned_emails:
            display_msg = replace_tags_with_link(display_msg, "email", f"#{unique_id}")
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
                    email_info_list.append(
                        {
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
                            "correct_folder": email_to_folder.get(email_id, "Unknown"),
                        }
                    )

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
        # Check if conditions contain direct email_id match pattern: email_id "X"
        if re.search(r'email_id\s+"[^"]+"', conditions):
            priority_rules.append(rule)
        else:
            regular_rules.append(rule)
    
    # Reorder: priority rules first, then regular rules
    reordered_policy = priority_rules + regular_rules

    # Apply the policy to all emails (with email_id rules prioritized)
    try:
        organized = apply_email_policy(emails_data, reordered_policy)
    except Exception as e:
        st.error(f"Error applying policy: {str(e)}")
        return

    # Display emails grouped by folder using tabs (like an inbox)
    # Filter out empty folders and sort folders
    folders_with_emails = {
        folder: emails
        for folder, emails in organized.items()
        if emails and folder != "Unsorted"
    }
    unsorted_emails = organized.get("Unsorted", [])
    
    # Create tab labels with email counts
    tab_labels = []
    tab_contents = []
    
    # Add regular folders
    for folder in sorted(folders_with_emails.keys()):
        emails = folders_with_emails[folder]
        tab_labels.append(f"{folder} ({len(emails)})")
        tab_contents.append(emails)
    
    # Add Unsorted folder if it has emails
    if unsorted_emails:
        tab_labels.insert(0, f"Unsorted ({len(unsorted_emails)})")
        tab_contents.insert(0, unsorted_emails)
    
    if not tab_labels:
        st.info("No emails to display.")
        return

    # Add button to view rules in a dialog
    @st.dialog("Email Organization Rules", width="large")
    def show_rules_dialog():
        st.markdown("### Policy Rules")
        st.info("Rules are applied in order. Each email is sorted into the **first folder** whose conditions match. Rules with direct email_id matches are always checked first.")
        st.markdown("")
        
        # Separate rules into priority (email_id direct matches) and regular rules
        priority_rules = []
        regular_rules = []
        
        for rule in policy:
            conditions = rule.get("conditions", "")
            # Check if conditions contain direct email_id match pattern: email_id "X"
            if re.search(r'email_id\s+"[^"]+"', conditions):
                priority_rules.append(rule)
            else:
                regular_rules.append(rule)
        
        # Combine: priority rules first, then regular rules
        sorted_policy = priority_rules + regular_rules
        
        for idx, rule in enumerate(sorted_policy, 1):
            conditions = rule.get("conditions", "")
            folder = rule.get("folder", "")
            
            # Wrap each rule in a bordered box
            with st.container(border=True):
                st.markdown(f"{idx}. **If email satisfies:**")
                st.code(conditions, language=None)
                st.markdown(f"**Sort to folder:** `{folder}`")
            
            # Add spacing between rule boxes
            if idx < len(sorted_policy):
                st.markdown("")
    
    st.button("View Email Organization Rules", on_click=show_rules_dialog, use_container_width=False)

    with st.container(border=True, height=700):
        st.subheader("Email Inbox")
    
        # Create tabs for each folder
        tabs = st.tabs(tab_labels)
        
        for tab_idx, (tab, emails) in enumerate(zip(tabs, tab_contents)):
            with tab:
                if not emails:
                    st.info("No emails in this folder.")
                    continue
                
                # Sort emails by date (newest first)
                def parse_email_date(date_str):
                    """Parse email date string to datetime for sorting."""
                    if not date_str:
                        return datetime.min
                    try:
                        # Try common date formats
                        # Format: "YYYY-MM-DD HH:MM:SS" or "Mon, DD MMM YYYY HH:MM:SS"
                        # Handle comma-separated dates
                        if "," in date_str:
                            # Format like "Mon, DD MMM YYYY HH:MM:SS"
                            date_part = date_str.split(",")[1].strip() if "," in date_str else date_str
                            # Try parsing with common formats
                            for fmt in ["%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                try:
                                    return datetime.strptime(date_part, fmt)
                                except ValueError:
                                    continue
                        else:
                            # Try direct parsing
                            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]:
                                try:
                                    return datetime.strptime(date_str.strip(), fmt)
                                except ValueError:
                                    continue
                    except Exception:
                        pass
                    return datetime.min
                
                # Sort emails by date (newest first)
                sorted_emails = sorted(emails, key=lambda e: parse_email_date(e.get("date", "")), reverse=True)
                
                # Display each email as an expandable item
                for email in sorted_emails:
                    email_id = str(email.get("email_id", ""))
                    subject = email.get("subject", "")
                    from_addr = email.get("from", "")
                    date = email.get("date", "")
                    message = email.get("message", "")
                    to_addr = email.get("to", "")
                    
                    # Create a short date format for the label
                    date_short = ""
                    if date:
                        # Try to get just the date part (before any time)
                        if "," in date:
                            date_short = date.split(",")[0]
                        else:
                            date_short = date
                    
                    # Create expander label with full sender, subject, and date on RHS
                    # Format: "sender | subject                    date"
                    max_label_length = 120
                    date_suffix = f" | {date_short}" if date_short else ""
                    date_suffix_len = len(date_suffix)
                    
                    if subject:
                        main_part = f"{from_addr} | {subject}"
                    else:
                        main_part = f"{from_addr} | (No Subject)"
                    
                    # Calculate available space for main part
                    available_len = max_label_length - date_suffix_len
                    if len(main_part) > available_len:
                        # Truncate the main part
                        truncate_len = available_len - 3  # 3 for "..."
                        if subject:
                            # Try to truncate just the subject
                            sender_part_len = len(from_addr) + 3  # " | "
                            if truncate_len > sender_part_len:
                                max_subject_len = truncate_len - sender_part_len
                                truncated_subject = subject[:max_subject_len] + "..."
                                main_part = f"{from_addr} | {truncated_subject}"
                            else:
                                main_part = main_part[:truncate_len] + "..."
                        else:
                            main_part = main_part[:truncate_len] + "..."
                    
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
                        message_clean = "\n".join(line.rstrip() for line in message.split("\n"))
                        st.markdown(f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>", unsafe_allow_html=True)


def render_email_policy_results_txt(msg: str, emails_data: List[Dict]) -> str:
    """
    Render the email policy results as JSON organized by folder, with emails listed under each folder.

    Args:
        msg: The message containing the policy
        emails_data: List of email dictionaries

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
                    email_info_list.append(
                        {
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
                            "correct_folder": email_to_folder.get(email_id, "Unknown"),
                        }
                    )
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
    Shows side-by-side clustering comparisons and collects user preferences.
    
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
    
    comparison_results = st.session_state.form_results["final_evaluation"]["comparison_results"]
    
    # Parse both policies
    pred_policy = parse_policy(final_prediction, raise_errors=False)
    y0_policy = parse_policy(y0, raise_errors=False)
    
    if pred_policy is None:
        st.error("Could not parse final_prediction policy.")
        return False, None
    if y0_policy is None:
        st.error("Could not parse y0 policy.")
        return False, None
    
    # Apply both policies to get clustering
    try:
        pred_organized = apply_email_policy(emails_data, pred_policy)
        y0_organized = apply_email_policy(emails_data, y0_policy)
    except Exception as e:
        st.error(f"Error applying policies: {str(e)}")
        return False, None
    
    # Build email_id to email mapping
    email_id_to_email = {str(e.get("email_id", "")): e for e in emails_data if e.get("email_id", "")}
    all_email_ids = list(email_id_to_email.keys())
    
    # Run comparisons
    current_comparison = len(comparison_results)
    
    if current_comparison < num_comparisons:
        # Sample items for this comparison (stable across reruns)
        sample_key = f"comparison_{current_comparison}_sample"
        if sample_key not in st.session_state:
            # Sample unique items
            sample_size = min(num_items_per_comparison, len(all_email_ids))
            st.session_state[sample_key] = random.sample(all_email_ids, sample_size)
        sampled_ids = st.session_state[sample_key]
        
        # Randomize which policy goes on which side (stable across reruns)
        side_key = f"comparison_{current_comparison}_side_assignment"
        if side_key not in st.session_state:
            # Randomly assign: True means pred on left, False means y0 on left
            st.session_state[side_key] = random.choice([True, False])
        pred_on_left = st.session_state[side_key]
        
        st.markdown(f"### Comparison {current_comparison + 1} of {num_comparisons}")
        st.markdown(f"Comparing how **{len(sampled_ids)} emails** are organized by each policy.")
        
        # Get clustering for sampled items under each policy
        pred_clusters = {}
        y0_clusters = {}
        
        for email_id in sampled_ids:
            # Find which folder each email goes to in each policy
            for folder, emails in pred_organized.items():
                if any(str(e.get("email_id", "")) == email_id for e in emails):
                    pred_clusters[email_id] = folder
                    break
            else:
                pred_clusters[email_id] = "Unsorted"
            
            for folder, emails in y0_organized.items():
                if any(str(e.get("email_id", "")) == email_id for e in emails):
                    y0_clusters[email_id] = folder
                    break
            else:
                y0_clusters[email_id] = "Unsorted"
        
        # Assign clusters to left/right based on randomization
        left_clusters = pred_clusters if pred_on_left else y0_clusters
        right_clusters = y0_clusters if pred_on_left else pred_clusters
        
        # Helper function to parse email date for sorting
        def parse_email_date(date_str):
            """Parse email date string to datetime for sorting."""
            if not date_str:
                return datetime.min
            try:
                # Try common date formats
                if "," in date_str:
                    date_part = date_str.split(",")[1].strip() if "," in date_str else date_str
                    for fmt in ["%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            return datetime.strptime(date_part, fmt)
                        except ValueError:
                            continue
                else:
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]:
                        try:
                            return datetime.strptime(date_str.strip(), fmt)
                        except ValueError:
                            continue
            except Exception:
                pass
            return datetime.min
        
        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("#### **Policy A**")
                left_folders = {}
                for email_id in sampled_ids:
                    folder = left_clusters[email_id]
                    if folder not in left_folders:
                        left_folders[folder] = []
                    left_folders[folder].append(email_id)
                
                # Display each folder as a section with emails as expandable items
                for folder in sorted(left_folders.keys()):
                    emails_in_folder = left_folders[folder]
                    # Get full email objects and sort by date
                    folder_emails = [email_id_to_email.get(eid, {}) for eid in emails_in_folder]
                    sorted_emails = sorted(folder_emails, key=lambda e: parse_email_date(e.get("date", "")), reverse=True)
                    
                    st.markdown(f"**📁 {folder}** ({len(emails_in_folder)} emails)")
                    
                    # Display each email as an expandable item
                    for email in sorted_emails:
                        email_id = str(email.get("email_id", ""))
                        subject = email.get("subject", "")
                        from_addr = email.get("from", "")
                        date = email.get("date", "")
                        message = email.get("message", "")
                        to_addr = email.get("to", "")
                        
                        # Create a short date format for the label
                        date_short = ""
                        if date:
                            if "," in date:
                                date_short = date.split(",")[0]
                            else:
                                date_short = date
                        
                        # Create expander label with full sender, subject, and date on RHS
                        max_label_length = 120
                        date_suffix = f" | {date_short}" if date_short else ""
                        date_suffix_len = len(date_suffix)
                        
                        if subject:
                            main_part = f"{from_addr} | {subject}"
                        else:
                            main_part = f"{from_addr} | (No Subject)"
                        
                        # Calculate available space for main part
                        available_len = max_label_length - date_suffix_len
                        if len(main_part) > available_len:
                            truncate_len = available_len - 3
                            if subject:
                                sender_part_len = len(from_addr) + 3
                                if truncate_len > sender_part_len:
                                    max_subject_len = truncate_len - sender_part_len
                                    truncated_subject = subject[:max_subject_len] + "..."
                                    main_part = f"{from_addr} | {truncated_subject}"
                                else:
                                    main_part = main_part[:truncate_len] + "..."
                            else:
                                main_part = main_part[:truncate_len] + "..."
                        
                        expander_label = main_part + date_suffix
                        
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
                            
                            # Email body - use markdown with reduced whitespace
                            st.markdown("**Message:**")
                            message_clean = "\n".join(line.rstrip() for line in message.split("\n"))
                            st.markdown(f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>", unsafe_allow_html=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("#### **Policy B**")
                right_folders = {}
                for email_id in sampled_ids:
                    folder = right_clusters[email_id]
                    if folder not in right_folders:
                        right_folders[folder] = []
                    right_folders[folder].append(email_id)
                
                # Display each folder as a section with emails as expandable items
                for folder in sorted(right_folders.keys()):
                    emails_in_folder = right_folders[folder]
                    # Get full email objects and sort by date
                    folder_emails = [email_id_to_email.get(eid, {}) for eid in emails_in_folder]
                    sorted_emails = sorted(folder_emails, key=lambda e: parse_email_date(e.get("date", "")), reverse=True)
                    
                    st.markdown(f"**📁 {folder}** ({len(emails_in_folder)} emails)")
                    
                    # Display each email as an expandable item
                    for email in sorted_emails:
                        email_id = str(email.get("email_id", ""))
                        subject = email.get("subject", "")
                        from_addr = email.get("from", "")
                        date = email.get("date", "")
                        message = email.get("message", "")
                        to_addr = email.get("to", "")
                        
                        # Create a short date format for the label
                        date_short = ""
                        if date:
                            if "," in date:
                                date_short = date.split(",")[0]
                            else:
                                date_short = date
                        
                        # Create expander label with full sender, subject, and date on RHS
                        max_label_length = 120
                        date_suffix = f" | {date_short}" if date_short else ""
                        date_suffix_len = len(date_suffix)
                        
                        if subject:
                            main_part = f"{from_addr} | {subject}"
                        else:
                            main_part = f"{from_addr} | (No Subject)"
                        
                        # Calculate available space for main part
                        available_len = max_label_length - date_suffix_len
                        if len(main_part) > available_len:
                            truncate_len = available_len - 3
                            if subject:
                                sender_part_len = len(from_addr) + 3
                                if truncate_len > sender_part_len:
                                    max_subject_len = truncate_len - sender_part_len
                                    truncated_subject = subject[:max_subject_len] + "..."
                                    main_part = f"{from_addr} | {truncated_subject}"
                                else:
                                    main_part = main_part[:truncate_len] + "..."
                            else:
                                main_part = main_part[:truncate_len] + "..."
                        
                        expander_label = main_part + date_suffix
                        
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
                            
                            # Email body - use markdown with reduced whitespace
                            st.markdown("**Message:**")
                            message_clean = "\n".join(line.rstrip() for line in message.split("\n"))
                            st.markdown(f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>", unsafe_allow_html=True)
        
        # Collect user preference
        with st.form(key=f"comparison_form_{current_comparison}"):
            preference = st.radio(
                "Which policy organizes these emails better?",
                options=["Policy A", "Policy B", "No preference"],
                key=f"preference_{current_comparison}",
            )
            submit = st.form_submit_button("Submit", type="primary")
            
            if submit:
                # Map user preference back to actual policies
                # If pred_on_left: Policy A = pred, Policy B = y0
                # If not pred_on_left: Policy A = y0, Policy B = pred
                if preference == "Policy A":
                    actual_preference = "pred" if pred_on_left else "y0"
                elif preference == "Policy B":
                    actual_preference = "y0" if pred_on_left else "pred"
                else:
                    actual_preference = "tie"
                
                comparison_results.append({
                    "comparison_index": current_comparison,
                    "sampled_email_ids": sampled_ids,
                    "preference": preference,
                    "actual_preference": actual_preference,  # Internal tracking
                    "pred_on_left": pred_on_left,  # Track assignment for this comparison
                    "pred_clusters": pred_clusters,
                    "y0_clusters": y0_clusters,
                })
                st.session_state.form_results["final_evaluation"]["comparison_results"] = comparison_results
                st.rerun()
        
        return False, None
    
    # Calculate summary statistics based on actual preferences
    pred_wins = sum(1 for r in comparison_results if r.get("actual_preference") == "pred")
    y0_wins = sum(1 for r in comparison_results if r.get("actual_preference") == "y0")
    ties = sum(1 for r in comparison_results if r.get("actual_preference") == "tie")
    
    # Calculate win rate score for final_prediction
    # Each pred win = 1 point, each tie = 0.5 points, each y0 win = 0 points
    # Win rate = (pred_wins + 0.5 * ties) / total_comparisons
    score = (pred_wins + 0.5 * ties) / num_comparisons if num_comparisons > 0 else 0.0
    
    # Store score in session state
    st.session_state.form_results["final_evaluation"]["score"] = score
    
    return True, {
        "comparison_results": comparison_results,
        "summary": {
            "pred_wins": pred_wins,
            "y0_wins": y0_wins,
            "ties": ties,
            "total": num_comparisons,
            "score": score,
        }
    }

