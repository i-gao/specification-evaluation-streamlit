"""
Streamlit search interface for file organization.

Allows users to design their ideal file organization (y*) by:
- Showing files one at a time or in batches
- Asking users to assign each file to a folder
- Allowing users to create new folders dynamically
- Allowing users to edit/rename existing folders
"""

from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime

# Session state keys used by this search interface that should be cleared between rounds
SEARCH_INTERFACE_SESSION_STATE_KEYS = [
    "file_organization_folders",
    "file_organization_assignments",
    "file_organization_current_index",
    "file_organization_show_all",
]


def parse_file_date(file_dict):
    """Parse file date string to datetime for sorting."""
    # Prefer edit_date, fallback to create_date
    date_str = file_dict.get("edit_date", "") or file_dict.get("create_date", "")
    if not date_str:
        return datetime.min
    try:
        # Try common date formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
    except Exception:
        pass
    return datetime.min


def render_search_interface(files_data: Optional[List[Dict]] = None, **kwargs):
    """
    Main function to render the file organization interface.

    Args:
        files_data: List of file dictionaries. If None, should be passed via kwargs.
        **kwargs: Additional arguments (may include files_data)
    """
    # Get files_data from kwargs if not provided directly
    if files_data is None:
        files_data = kwargs.get("files_data", [])

    if not files_data:
        st.info("No files available to organize.")
        return

    # Initialize session state
    if "file_organization_folders" not in st.session_state:
        st.session_state.file_organization_folders = ["Uncategorized"]

    if "file_organization_assignments" not in st.session_state:
        st.session_state.file_organization_assignments = {}

    if "file_organization_current_index" not in st.session_state:
        st.session_state.file_organization_current_index = 0

    if "file_organization_show_all" not in st.session_state:
        st.session_state.file_organization_show_all = False

    folders = st.session_state.file_organization_folders
    assignments = st.session_state.file_organization_assignments

    st.markdown(
        "Assign each file to a folder. You can create new folders or use existing ones."
    )

    # Folder management section
    with st.expander("📁 Manage Folders", expanded=True):
        st.markdown("### Existing Folders")
        for i, folder in enumerate(folders):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_name = st.text_input(
                    f"Folder {i + 1}",
                    value=folder,
                    key=f"folder_edit_{i}",
                    label_visibility="collapsed",
                )
            with col2:
                if st.button(
                    "Delete",
                    key=f"folder_delete_{i}",
                    disabled=(folder == "Uncategorized"),
                ):
                    # Remove folder and reassign its files to Uncategorized
                    for filename in list(assignments.keys()):
                        if assignments[filename] == folder:
                            assignments[filename] = "Uncategorized"
                    folders.remove(folder)
                    st.rerun()

            if new_name != folder and new_name.strip():
                # Rename folder
                old_name = folder
                folders[i] = new_name.strip()
                # Update all assignments
                for filename in assignments:
                    if assignments[filename] == old_name:
                        assignments[filename] = new_name.strip()
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
        show_all = st.checkbox(
            "Show all files at once", value=st.session_state.file_organization_show_all
        )
        if show_all != st.session_state.file_organization_show_all:
            st.session_state.file_organization_show_all = show_all
            st.rerun()

    # Sort files by date (newest first)
    sorted_files = sorted(files_data, key=parse_file_date, reverse=True)

    if st.session_state.file_organization_show_all:
        # Show all files at once
        st.markdown(f"### Assign {len(sorted_files)} Files to Folders")

        for file in sorted_files:
            filename = file.get("filename", "")
            create_date = file.get("create_date", "")
            edit_date = file.get("edit_date", "")
            file_preview = file.get("file_contents_preview", "")

            # Format dates
            if create_date:
                create_display = (
                    create_date[:16] if len(create_date) > 10 else create_date[:10]
                )
            else:
                create_display = ""
            if edit_date:
                edit_display = edit_date[:16] if len(edit_date) > 10 else edit_date[:10]
            else:
                edit_display = ""

            # Determine file icon
            file_ext = filename.split(".")[-1].lower() if "." in filename else ""
            file_icon = "📄"
            if file_ext in ["jpg", "jpeg", "png", "gif", "bmp", "svg", "webp"]:
                file_icon = "🖼️"
            elif file_ext in ["pdf"]:
                file_icon = "📕"
            elif file_ext in ["doc", "docx"]:
                file_icon = "📘"
            elif file_ext in ["xls", "xlsx", "csv"]:
                file_icon = "📊"
            elif file_ext in ["txt", "md", "rtf"]:
                file_icon = "📝"
            elif file_ext in ["py", "js", "java", "cpp", "c", "ts", "html", "css"]:
                file_icon = "💻"
            elif file_ext in ["zip", "rar", "7z", "tar", "gz"]:
                file_icon = "📦"
            elif file_ext in ["mp3", "mp4", "avi", "wav", "flac"]:
                file_icon = "🎵"

            with st.expander(f"{file_icon} {filename}", expanded=False):
                # File details
                col1a, col1b = st.columns([2, 1])
                with col1a:
                    st.markdown(f"**Filename:** {filename}")
                    if create_display:
                        st.markdown(f"**Date Created:** {create_display}")
                    if edit_display:
                        st.markdown(f"**Date Modified:** {edit_display}")
                with col1b:
                    if file_ext:
                        st.markdown(f"**Type:** {file_ext.upper()}")

                if file_preview:
                    st.markdown("**Preview:**")
                    preview_display = (
                        file_preview[:500] + "..."
                        if len(file_preview) > 500
                        else file_preview
                    )
                    st.code(preview_display, language=None)

                # Folder assignment
                current_folder = assignments.get(filename, "Uncategorized")
                selected_folder = st.selectbox(
                    "Assign to folder:",
                    options=folders,
                    index=folders.index(current_folder)
                    if current_folder in folders
                    else 0,
                    key=f"folder_select_{filename}",
                    accept_new_options=True,
                    placeholder="Select or create a folder",
                )

                if selected_folder != current_folder:
                    assignments[filename] = selected_folder
                    # Add new folder if it doesn't exist
                    if selected_folder not in folders:
                        folders.append(selected_folder)
                    st.rerun()
    else:
        # Show one file at a time
        current_index = st.session_state.file_organization_current_index

        if current_index >= len(sorted_files):
            st.success("✅ You've organized all files!")
            st.markdown("### Summary")
            folder_counts = {}
            for filename, folder in assignments.items():
                folder_counts[folder] = folder_counts.get(folder, 0) + 1

            for folder in sorted(folders):
                count = folder_counts.get(folder, 0)
                st.markdown(f"- **{folder}**: {count} files")

            if st.button("Start Over", key="restart_organization"):
                st.session_state.file_organization_current_index = 0
                st.session_state.file_organization_assignments = {}
                st.rerun()
            return

        file = sorted_files[current_index]
        filename = file.get("filename", "")
        create_date = file.get("create_date", "")
        edit_date = file.get("edit_date", "")
        file_preview = file.get("file_contents_preview", "")

        st.markdown(f"### File {current_index + 1} of {len(sorted_files)}")

        # Format dates
        if create_date:
            create_display = (
                create_date[:16] if len(create_date) > 10 else create_date[:10]
            )
        else:
            create_display = ""
        if edit_date:
            edit_display = edit_date[:16] if len(edit_date) > 10 else edit_date[:10]
        else:
            edit_display = ""

        # Determine file icon
        file_ext = filename.split(".")[-1].lower() if "." in filename else ""
        file_icon = "📄"
        if file_ext in ["jpg", "jpeg", "png", "gif", "bmp", "svg", "webp"]:
            file_icon = "🖼️"
        elif file_ext in ["pdf"]:
            file_icon = "📕"
        elif file_ext in ["doc", "docx"]:
            file_icon = "📘"
        elif file_ext in ["xls", "xlsx", "csv"]:
            file_icon = "📊"
        elif file_ext in ["txt", "md", "rtf"]:
            file_icon = "📝"
        elif file_ext in ["py", "js", "java", "cpp", "c", "ts", "html", "css"]:
            file_icon = "💻"
        elif file_ext in ["zip", "rar", "7z", "tar", "gz"]:
            file_icon = "📦"
        elif file_ext in ["mp3", "mp4", "avi", "wav", "flac"]:
            file_icon = "🎵"

        # File details
        with st.container(border=True):
            col1a, col1b = st.columns([2, 1])
            with col1a:
                st.markdown(f"**{file_icon} Filename:** {filename}")
                if create_display:
                    st.markdown(f"**Date Created:** {create_display}")
                if edit_display:
                    st.markdown(f"**Date Modified:** {edit_display}")
            with col1b:
                if file_ext:
                    st.markdown(f"**Type:** {file_ext.upper()}")

            if file_preview:
                st.markdown("**Preview:**")
                preview_display = (
                    file_preview[:500] + "..."
                    if len(file_preview) > 500
                    else file_preview
                )
                st.code(preview_display, language=None)

        # Folder assignment
        current_folder = assignments.get(filename, "Uncategorized")
        selected_folder = st.selectbox(
            "Assign to folder:",
            options=folders,
            index=folders.index(current_folder) if current_folder in folders else 0,
            key="current_file_folder_select",
            accept_new_options=True,
            placeholder="Select or create a folder",
        )

        # Update assignment
        if selected_folder != current_folder:
            assignments[filename] = selected_folder
            # Add new folder if it doesn't exist
            if selected_folder not in folders:
                folders.append(selected_folder)

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("← Previous", disabled=(current_index == 0)):
                st.session_state.file_organization_current_index = current_index - 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary"):
                # Save current assignment
                assignments[filename] = selected_folder
                if selected_folder not in folders:
                    folders.append(selected_folder)
                st.session_state.file_organization_current_index = current_index + 1
                st.rerun()
        with col3:
            progress = (current_index + 1) / len(sorted_files)
            st.progress(
                progress,
                text=f"{current_index + 1} / {len(sorted_files)} files organized",
            )

    # Store assignments in a format that can be retrieved
    # The assignments dict maps filename -> folder_name
    # This can be used to construct y* later


def assignments_to_policy(assignments: Dict[str, str]) -> str:
    """
    Convert file assignments dict to policy format.

    Args:
        assignments: Dict mapping filename -> folder_name

    Returns:
        Policy string wrapped in <policy></policy> tags
    """
    import json

    moving_rules = []
    for filename, folder_name in assignments.items():
        # Extract base filename without extension for matching
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        moving_rules.append(
            {"conditions": {"name_contains": [base_name]}, "folder": folder_name}
        )
    user_policy = {"moving_rules": moving_rules, "naming_policy": {}}
    return f"<policy>{json.dumps(user_policy)}</policy>"


def render_user_policy(assignments: Dict[str, str], files_data: List[Dict]):
    """
    Render the user's organization policy (similar to how liked items are rendered).

    Args:
        assignments: Dict mapping filename -> folder_name
        files_data: List of file dictionaries
    """
    if not assignments:
        st.markdown("*No organization assignments were made during exploration.*")
        return

    st.markdown(f"You organized {len(assignments)} file(s) during exploration.")

    # Convert assignments to policy and render it
    user_policy_str = assignments_to_policy(assignments)

    # Use the spec's render_msg_fn to display the policy results
    # This will show the files organized by folder
    from data.file_organization.streamlit_render import render_file_policy_results

    render_file_policy_results(user_policy_str, files_data)
