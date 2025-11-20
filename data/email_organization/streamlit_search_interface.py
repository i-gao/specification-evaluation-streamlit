"""
Streamlit search interface for email organization.

Allows users to design their ideal email organization (y*) by:
- Showing emails one at a time or in batches
- Asking users to assign each email to a folder
- Allowing users to create new folders dynamically
- Allowing users to edit/rename existing folders
"""

from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "email_organization_folders",
    "email_organization_assignments",
    "email_organization_current_index",
    "email_organization_show_all",
]

def parse_email_date(date_str):
    """Parse email date string to datetime for sorting."""
    if not date_str:
        return datetime.min
    try:
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


def render_search_interface(emails_data: Optional[List[Dict]] = None, **kwargs):
    """
    Main function to render the email organization interface.
    
    Args:
        emails_data: List of email dictionaries. If None, should be passed via kwargs.
        **kwargs: Additional arguments (may include emails_data)
    """
    # Get emails_data from kwargs if not provided directly
    if emails_data is None:
        emails_data = kwargs.get("emails_data", [])
    
    if not emails_data:
        st.info("No emails available to organize.")
        return
    
    # Initialize session state
    if "email_organization_folders" not in st.session_state:
        st.session_state.email_organization_folders = ["Unsorted"]
    
    if "email_organization_assignments" not in st.session_state:
        st.session_state.email_organization_assignments = {}
    
    if "email_organization_current_index" not in st.session_state:
        st.session_state.email_organization_current_index = 0
    
    if "email_organization_show_all" not in st.session_state:
        st.session_state.email_organization_show_all = False
    
    folders = st.session_state.email_organization_folders
    assignments = st.session_state.email_organization_assignments
    
    st.markdown("Assign each email to a folder. You can create new folders or use existing ones.")
    
    # Folder management section
    with st.expander("📁 Manage Folders", expanded=True):
        st.markdown("### Existing Folders")
        for i, folder in enumerate(folders):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_name = st.text_input(
                    f"Folder {i+1}",
                    value=folder,
                    key=f"folder_edit_{i}",
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("Delete", key=f"folder_delete_{i}", disabled=(folder == "Unsorted")):
                    # Remove folder and reassign its emails to Unsorted
                    for email_id in list(assignments.keys()):
                        if assignments[email_id] == folder:
                            assignments[email_id] = "Unsorted"
                    folders.remove(folder)
                    st.rerun()
            
            if new_name != folder and new_name.strip():
                # Rename folder
                old_name = folder
                folders[i] = new_name.strip()
                # Update all assignments
                for email_id in assignments:
                    if assignments[email_id] == old_name:
                        assignments[email_id] = new_name.strip()
                st.rerun()
        
        st.markdown("### Add New Folder")
        new_folder = st.text_input("New folder name", key="new_folder_input")
        if st.button("Add Folder", key="add_folder_button"):
            if new_folder.strip() and new_folder.strip() not in folders:
                folders.append(new_folder.strip())
                st.rerun()
    
    # Display mode toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        show_all = st.checkbox("Show all emails at once", value=st.session_state.email_organization_show_all)
        if show_all != st.session_state.email_organization_show_all:
            st.session_state.email_organization_show_all = show_all
            st.rerun()
    
    # Sort emails by date (newest first)
    sorted_emails = sorted(emails_data, key=lambda e: parse_email_date(e.get("date", "")), reverse=True)
    
    if st.session_state.email_organization_show_all:
        # Show all emails at once
        st.markdown(f"### Assign {len(sorted_emails)} Emails to Folders")
        
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
            
            # Create expander label
            max_label_length = 120
            date_suffix = f" | {date_short}" if date_short else ""
            date_suffix_len = len(date_suffix)
            
            if subject:
                main_part = f"{from_addr} | {subject}"
            else:
                main_part = f"{from_addr} | (No Subject)"
            
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
                # Email details
                col1a, col1b = st.columns([3, 1])
                with col1a:
                    st.markdown(f"**From:** {from_addr}")
                    if to_addr:
                        st.markdown(f"**To:** {to_addr}")
                with col1b:
                    st.markdown(f"**Email ID:** `{email_id}`")
                    st.markdown(f"**Date:** {date}")
                
                if subject:
                    st.markdown(f"**Subject:** {subject}")
                
                st.markdown("**Message:**")
                message_clean = "\n".join(line.rstrip() for line in message.split("\n"))
                st.markdown(f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>", unsafe_allow_html=True)
                
                # Folder assignment
                current_folder = assignments.get(email_id, "Unsorted")
                selected_folder = st.selectbox(
                    "Assign to folder:",
                    options=folders,
                    index=folders.index(current_folder) if current_folder in folders else 0,
                    key=f"folder_select_{email_id}",
                    accept_new_options=True,
                    placeholder="Select or create a folder"
                )
                
                if selected_folder != current_folder:
                    assignments[email_id] = selected_folder
                    # Add new folder if it doesn't exist
                    if selected_folder not in folders:
                        folders.append(selected_folder)
                    st.rerun()
    else:
        # Show one email at a time
        current_index = st.session_state.email_organization_current_index
        
        if current_index >= len(sorted_emails):
            st.success("✅ You've organized all emails!")
            st.markdown("### Summary")
            folder_counts = {}
            for email_id, folder in assignments.items():
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            for folder in sorted(folders):
                count = folder_counts.get(folder, 0)
                st.markdown(f"- **{folder}**: {count} emails")
            
            if st.button("Start Over", key="restart_organization"):
                st.session_state.email_organization_current_index = 0
                st.session_state.email_organization_assignments = {}
                st.rerun()
            return
        
        email = sorted_emails[current_index]
        email_id = str(email.get("email_id", ""))
        subject = email.get("subject", "")
        from_addr = email.get("from", "")
        date = email.get("date", "")
        message = email.get("message", "")
        to_addr = email.get("to", "")
        
        st.markdown(f"### Email {current_index + 1} of {len(sorted_emails)}")
        
        # Email details
        with st.container(border=True):
            col1a, col1b = st.columns([3, 1])
            with col1a:
                st.markdown(f"**From:** {from_addr}")
                if to_addr:
                    st.markdown(f"**To:** {to_addr}")
            with col1b:
                st.markdown(f"**Email ID:** `{email_id}`")
                st.markdown(f"**Date:** {date}")
            
            if subject:
                st.markdown(f"**Subject:** {subject}")
            
            st.markdown("**Message:**")
            message_clean = "\n".join(line.rstrip() for line in message.split("\n"))
            st.markdown(f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>{message_clean}</div>", unsafe_allow_html=True)
        
        # Folder assignment
        current_folder = assignments.get(email_id, "Unsorted")
        selected_folder = st.selectbox(
            "Assign to folder:",
            options=folders,
            index=folders.index(current_folder) if current_folder in folders else 0,
            key="current_email_folder_select",
            accept_new_options=True,
            placeholder="Select or create a folder"
        )
        
        # Update assignment
        if selected_folder != current_folder:
            assignments[email_id] = selected_folder
            # Add new folder if it doesn't exist
            if selected_folder not in folders:
                folders.append(selected_folder)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("← Previous", disabled=(current_index == 0)):
                st.session_state.email_organization_current_index = current_index - 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary"):
                # Save current assignment
                assignments[email_id] = selected_folder
                if selected_folder not in folders:
                    folders.append(selected_folder)
                st.session_state.email_organization_current_index = current_index + 1
                st.rerun()
        with col3:
            progress = (current_index + 1) / len(sorted_emails)
            st.progress(progress, text=f"{current_index + 1} / {len(sorted_emails)} emails organized")
    
    # Store assignments in a format that can be retrieved
    # The assignments dict maps email_id -> folder_name
    # This can be used to construct y* later


def assignments_to_policy(assignments: Dict[str, str]) -> str:
    """
    Convert email assignments dict to policy format.
    
    Args:
        assignments: Dict mapping email_id -> folder_name
        
    Returns:
        Policy string wrapped in <policy></policy> tags
    """
    import json
    policy_rules = []
    for email_id, folder_name in assignments.items():
        policy_rules.append({
            "conditions": f'email_id "{email_id}"',
            "folder": folder_name
        })
    user_policy = json.dumps(policy_rules)
    return f"<policy>{user_policy}</policy>"


def render_user_policy(assignments: Dict[str, str], emails_data: List[Dict]):
    """
    Render the user's organization policy (similar to how liked items are rendered).
    
    Args:
        assignments: Dict mapping email_id -> folder_name
        emails_data: List of email dictionaries
    """
    if not assignments:
        st.markdown("*No organization assignments were made during exploration.*")
        return
    
    st.markdown(f"You organized {len(assignments)} email(s) during exploration.")
    
    # Convert assignments to policy and render it
    user_policy_str = assignments_to_policy(assignments)
    
    # Use the spec's render_msg_fn to display the policy results
    # This will show the emails organized by folder
    from data.email_organization.streamlit_render import render_email_policy_results
    render_email_policy_results(user_policy_str, emails_data)

