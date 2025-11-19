from typing import List, Dict, Tuple, Optional
import streamlit as st
import json
import random
from datetime import datetime
from utils.misc import parse_for_answer_tags
from data.file_organization.reward import apply_policy
from data.file_organization.parser import parse_policy


def render_file_policy_results(msg: str, files_data: List[Dict]) -> None:
    """
    Render the file policy results grouped by folder, showing files within each folder.
    Displays in a Windows Explorer-like interface.

    Args:
        msg: The message containing the policy
        files_data: List of file dictionaries
    """
    # Extract policy and find its position in the message
    policy_str, start_end = parse_for_answer_tags(
        msg, keyword="policy", return_start_end=True, return_none_if_not_found=True
    )

    # Display message without the policy section
    if start_end is not None:
        # Message has policy tags - display everything except the policy
        before_policy = msg[: start_end[0]]
        after_policy = msg[start_end[1] :]
        message_without_policy = before_policy + after_policy
        if message_without_policy.strip():
            st.markdown(message_without_policy)
    else:
        # No policy tags found - display the full message
        st.markdown(msg)

    # Parse the policy from the message
    policy = parse_policy(msg, raise_errors=False)

    if policy is None:
        # If we can't parse the policy, we've already displayed the message
        return

    # Apply the policy to all files
    try:
        organized = apply_policy(files_data, policy)
    except Exception as e:
        st.error(f"Error applying policy: {str(e)}")
        return

    # Windows Explorer-like display
    with st.container(border=True, height=700):
        st.subheader("File Explorer")
        # Add CSS for Windows Explorer-like styling
        st.markdown("""
        <style>
        div[data-testid="column"]:first-child {
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
        }
        .folder-item {
            padding: 6px 10px;
            cursor: pointer;
            border-radius: 3px;
            margin: 2px 0;
            font-size: 14px;
        }
        .folder-item:hover {
            background-color: #e8f4f8;
        }
        .file-item {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
            margin: 0;
            font-size: 14px;
        }
        .file-item:hover {
            background-color: #f0f0f0;
        }
        .explorer-header {
            background-color: #f5f5f5;
            padding: 10px 12px;
            border-bottom: 2px solid #ddd;
            font-weight: 600;
            font-size: 14px;
            color: #333;
        }
        .file-grid {
            display: grid;
            grid-template-columns: 50px 2fr 140px 140px;
            gap: 12px;
            padding: 8px 12px;
            align-items: center;
            border-bottom: 1px solid #eee;
            box-sizing: border-box;
            width: 100%;
        }
        .file-grid:last-child {
            border-bottom: none;
        }
        .file-grid-header {
            font-weight: 600;
            background-color: #f5f5f5;
            padding: 10px 12px;
            border-bottom: 2px solid #ddd;
            color: #333;
            font-size: 13px;
            box-sizing: border-box;
            width: 100%;
        }
        .file-grid-container {
            overflow-x: auto;
            width: 100%;
            box-sizing: border-box;
        }
        .file-name {
            font-size: 14px;
            color: #333;
        }
        .file-date {
            font-size: 13px;
            color: #666;
        }
        .original-name {
            font-size: 12px;
            color: #888;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sort folders for consistent display
        sorted_folders = [f for f in sorted(organized.keys()) if len(organized[f]) > 0]
        
        if not sorted_folders:
            st.info("No files to display.")
            return
        
        # Create a two-column layout: sidebar for folders, main area for files
        col1, col2 = st.columns([1, 3], gap="small")
        
        # Sidebar: Folder tree view (like Windows Explorer navigation pane)
        with col1:            
            # Display folder tree
            for folder in sorted_folders:
                files_count = len(organized[folder])
                st.markdown(f'<div class="folder-item">📁 <strong>{folder}</strong> ({files_count})</div>', unsafe_allow_html=True)
        
        # Main area: File details view (like Windows Explorer details pane)
        with col2:
            # Create tabs for each folder if multiple folders exist
            if len(sorted_folders) > 1:
                folder_tabs = st.tabs([f"📁 {folder} ({len(organized[folder])})" for folder in sorted_folders])
                folder_tab_map = {folder: tab for folder, tab in zip(sorted_folders, folder_tabs)}
            else:
                # Single folder - no tabs needed
                folder_tab_map = {sorted_folders[0]: st.container()}
            
            # Display files for each folder
            for folder in sorted_folders:
                files = organized[folder]
                if not files:
                    continue
                
                # Sort files by edit_date (newest first), fallback to create_date if edit_date is missing
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
                
                # Sort files by date (newest first)
                sorted_files = sorted(files, key=parse_file_date, reverse=True)
                
                # Use tab if multiple folders, otherwise use container
                container = folder_tab_map[folder]
                
                with container:
                    # Wrap file grids in a container for proper overflow handling
                    st.markdown('<div class="file-grid-container">', unsafe_allow_html=True)
                    
                    # Header row (column headers)
                    st.markdown("""
                    <div class="file-grid file-grid-header">
                        <div></div>
                        <div>Name</div>
                        <div>Date Created</div>
                        <div>Date Modified</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # File rows
                    for file in sorted_files:
                        original_filename = file.get("original_filename", "")
                        new_filename = file.get("filename", "")
                        create_date = file.get("create_date", "")
                        edit_date = file.get("edit_date", "")
                        file_preview = file.get("file_contents_preview", "")
                        
                        # Format dates (show date and time if available)
                        if create_date:
                            create_display = create_date[:16] if len(create_date) > 10 else create_date[:10]
                        else:
                            create_display = ""
                        if edit_date:
                            edit_display = edit_date[:16] if len(edit_date) > 10 else edit_date[:10]
                        else:
                            edit_display = ""
                        
                        # Determine file icon based on extension
                        file_ext = new_filename.split('.')[-1].lower() if '.' in new_filename else ""
                        file_icon = "📄"
                        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp']:
                            file_icon = "🖼️"
                        elif file_ext in ['pdf']:
                            file_icon = "📕"
                        elif file_ext in ['doc', 'docx']:
                            file_icon = "📘"
                        elif file_ext in ['xls', 'xlsx', 'csv']:
                            file_icon = "📊"
                        elif file_ext in ['txt', 'md', 'rtf']:
                            file_icon = "📝"
                        elif file_ext in ['py', 'js', 'java', 'cpp', 'c', 'ts', 'html', 'css']:
                            file_icon = "💻"
                        elif file_ext in ['zip', 'rar', '7z', 'tar', 'gz']:
                            file_icon = "📦"
                        elif file_ext in ['mp3', 'mp4', 'avi', 'wav', 'flac']:
                            file_icon = "🎵"
                        
                        # Show new filename if different from original, otherwise show original
                        display_name = new_filename if new_filename != original_filename else original_filename
                        name_display = f'<span class="file-name">{display_name}</span>'
                        if new_filename != original_filename:
                            name_display += f'<br><span class="original-name">Original filename: {original_filename}</span>'
                        
                        # Add preview if available
                        if file_preview:
                            # Truncate preview for inline display
                            preview_display = file_preview[:200] + "..." if len(file_preview) > 200 else file_preview
                            # Escape HTML in preview to prevent rendering issues
                            preview_display = preview_display.replace("<", "&lt;").replace(">", "&gt;")
                            name_display += f'<br><span class="original-name" style="font-size: 11px; color: #888; font-style: normal; margin-top: 4px; display: block;">{preview_display}</span>'
                        
                        st.markdown(f"""
                        <div class="file-grid file-item">
                            <div style="font-size: 20px;">{file_icon}</div>
                            <div>{name_display}</div>
                            <div class="file-date">{create_display}</div>
                            <div class="file-date">{edit_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Close the file grid container
                    st.markdown('</div>', unsafe_allow_html=True)


def render_file_policy_results_txt(msg: str, files_data: List[Dict]) -> str:
    """
    Render the file policy results as JSON organized by folder, with files listed under each folder.

    Args:
        msg: The message containing the policy
        files_data: List of file dictionaries

    Returns:
        JSON string with folders as keys and arrays of file objects as values
    """
    # Parse the policy from the message
    policy = parse_policy(msg, raise_errors=False)

    if policy is None:
        # If we can't parse the policy, return the raw message
        return msg

    # Apply the policy to all files
    try:
        organized = apply_policy(files_data, policy)
    except Exception as e:
        # Return error message as JSON
        return (
            msg
            + "\n\n-----------Working directory state after running the policy:-----------\n\n"
            + json.dumps({"error": f"Error applying policy: {str(e)}"})
        )

    # Truncate each file_contents_preview to 500 characters
    for folder, files in organized.items():
        for file in files:
            preview = file.get("file_contents_preview", "")
            if preview is not None and len(preview) > 500:
                file["file_contents_preview"] = preview[:500] + "..."
            file["new_filename"] = file.pop("filename")

    return (
        msg
        + "\n\n-----------Working directory state after running the policy:-----------\n\n"
        + json.dumps(organized)
    )


def render_eval(
    *,
    final_prediction: str,
    y0: str,
    files_data: List[Dict],
    num_comparisons: int = 5,
    num_items_per_comparison: int = 5,
) -> Tuple[bool, Optional[Dict]]:
    """
    Render evaluation UI comparing final_prediction against y0.
    Shows side-by-side clustering comparisons and collects user preferences.
    
    Args:
        final_prediction: The assistant's policy prediction
        y0: The gold/reference policy
        files_data: List of file dictionaries
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
        pred_organized = apply_policy(files_data, pred_policy)
        y0_organized = apply_policy(files_data, y0_policy)
    except Exception as e:
        st.error(f"Error applying policies: {str(e)}")
        return False, None
    
    # Build filename to file mapping
    # Note: files_data contains raw files with "filename" as the original filename
    # After apply_policy, files will have both "original_filename" and "filename" (new name)
    filename_to_file = {f.get("filename", ""): f for f in files_data if f.get("filename", "")}
    all_filenames = list(filename_to_file.keys())
    
    # Run comparisons
    current_comparison = len(comparison_results)
    
    if current_comparison < num_comparisons:
        # Sample items for this comparison (stable across reruns)
        sample_key = f"comparison_{current_comparison}_sample"
        if sample_key not in st.session_state:
            # Sample unique items
            sample_size = min(num_items_per_comparison, len(all_filenames))
            st.session_state[sample_key] = random.sample(all_filenames, sample_size)
        sampled_filenames = st.session_state[sample_key]
        
        # Randomize which policy goes on which side (stable across reruns)
        side_key = f"comparison_{current_comparison}_side_assignment"
        if side_key not in st.session_state:
            # Randomly assign: True means pred on left, False means y0 on left
            st.session_state[side_key] = random.choice([True, False])
        pred_on_left = st.session_state[side_key]
        
        st.markdown(f"### Comparison {current_comparison + 1} of {num_comparisons}")
        st.markdown(f"Comparing how **{len(sampled_filenames)} files** are organized by each policy.")
        
        # Add CSS for Windows Explorer-like styling
        st.markdown("""
        <style>
        .file-item {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
            margin: 0;
            font-size: 14px;
        }
        .file-item:hover {
            background-color: #f0f0f0;
        }
        .file-grid {
            display: grid;
            grid-template-columns: 50px 2fr 140px 140px;
            gap: 12px;
            padding: 8px 12px;
            align-items: center;
            border-bottom: 1px solid #eee;
            box-sizing: border-box;
            width: 100%;
        }
        .file-grid:last-child {
            border-bottom: none;
        }
        .file-grid-header {
            font-weight: 600;
            background-color: #f5f5f5;
            padding: 10px 12px;
            border-bottom: 2px solid #ddd;
            color: #333;
            font-size: 13px;
            box-sizing: border-box;
            width: 100%;
        }
        .file-grid-container {
            overflow-x: auto;
            width: 100%;
            box-sizing: border-box;
        }
        .file-name {
            font-size: 14px;
            color: #333;
        }
        .file-date {
            font-size: 13px;
            color: #666;
        }
        .original-name {
            font-size: 12px;
            color: #888;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get clustering for sampled items under each policy
        pred_clusters = {}
        y0_clusters = {}
        
        for filename in sampled_filenames:
            # Find which folder each file goes to in each policy
            # After apply_policy, files have "original_filename" field
            for folder, files in pred_organized.items():
                if any(f.get("original_filename", "") == filename for f in files):
                    pred_clusters[filename] = folder
                    break
            else:
                pred_clusters[filename] = "Uncategorized"
            
            for folder, files in y0_organized.items():
                if any(f.get("original_filename", "") == filename for f in files):
                    y0_clusters[filename] = folder
                    break
            else:
                y0_clusters[filename] = "Uncategorized"
        
        # Assign clusters to left/right based on randomization
        left_clusters = pred_clusters if pred_on_left else y0_clusters
        right_clusters = y0_clusters if pred_on_left else pred_clusters
        
        # Helper function to parse file date for sorting
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
        
        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("#### **Policy A**")
                left_folders = {}
                for filename in sampled_filenames:
                    folder = left_clusters[filename]
                    if folder not in left_folders:
                        left_folders[folder] = []
                    left_folders[folder].append(filename)
                
                # Display each folder as a section with files in Windows Explorer style
                for folder in sorted(left_folders.keys()):
                    files_in_folder = left_folders[folder]
                    # Get full file objects from organized data and sort by date
                    folder_files = []
                    for filename in files_in_folder:
                        for f in pred_organized.get(left_clusters[filename], []):
                            if f.get("original_filename", "") == filename:
                                folder_files.append(f)
                                break
                    
                    sorted_files = sorted(folder_files, key=parse_file_date, reverse=True)
                    
                    st.markdown(f"**📁 {folder}** ({len(files_in_folder)} files)")
                    
                    # Windows Explorer-style grid
                    st.markdown('<div class="file-grid-container">', unsafe_allow_html=True)
                    
                    # Header row
                    st.markdown("""
                    <div class="file-grid file-grid-header">
                        <div></div>
                        <div>Name</div>
                        <div>Date Created</div>
                        <div>Date Modified</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # File rows
                    for file in sorted_files:
                        original_filename = file.get("original_filename", "")
                        new_filename = file.get("filename", "")
                        create_date = file.get("create_date", "")
                        edit_date = file.get("edit_date", "")
                        file_preview = file.get("file_contents_preview", "")
                        
                        # Format dates
                        if create_date:
                            create_display = create_date[:16] if len(create_date) > 10 else create_date[:10]
                        else:
                            create_display = ""
                        if edit_date:
                            edit_display = edit_date[:16] if len(edit_date) > 10 else edit_date[:10]
                        else:
                            edit_display = ""
                        
                        # Determine file icon
                        file_ext = new_filename.split('.')[-1].lower() if '.' in new_filename else ""
                        file_icon = "📄"
                        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp']:
                            file_icon = "🖼️"
                        elif file_ext in ['pdf']:
                            file_icon = "📕"
                        elif file_ext in ['doc', 'docx']:
                            file_icon = "📘"
                        elif file_ext in ['xls', 'xlsx', 'csv']:
                            file_icon = "📊"
                        elif file_ext in ['txt', 'md', 'rtf']:
                            file_icon = "📝"
                        elif file_ext in ['py', 'js', 'java', 'cpp', 'c', 'ts', 'html', 'css']:
                            file_icon = "💻"
                        elif file_ext in ['zip', 'rar', '7z', 'tar', 'gz']:
                            file_icon = "📦"
                        elif file_ext in ['mp3', 'mp4', 'avi', 'wav', 'flac']:
                            file_icon = "🎵"
                        
                        # Show new filename if different from original, otherwise show original
                        display_name = new_filename if new_filename != original_filename else original_filename
                        name_display = f'<span class="file-name">{display_name}</span>'
                        if new_filename != original_filename:
                            name_display += f'<br><span class="original-name">Original filename: {original_filename}</span>'
                        
                        # Add preview if available
                        if file_preview:
                            preview_display = file_preview[:200] + "..." if len(file_preview) > 200 else file_preview
                            preview_display = preview_display.replace("<", "&lt;").replace(">", "&gt;")
                            name_display += f'<br><span class="original-name" style="font-size: 11px; color: #888; font-style: normal; margin-top: 4px; display: block;">{preview_display}</span>'
                        
                        st.markdown(f"""
                        <div class="file-grid file-item">
                            <div style="font-size: 20px;">{file_icon}</div>
                            <div>{name_display}</div>
                            <div class="file-date">{create_display}</div>
                            <div class="file-date">{edit_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("#### **Policy B**")
                right_folders = {}
                for filename in sampled_filenames:
                    folder = right_clusters[filename]
                    if folder not in right_folders:
                        right_folders[folder] = []
                    right_folders[folder].append(filename)
                
                # Display each folder as a section with files in Windows Explorer style
                for folder in sorted(right_folders.keys()):
                    files_in_folder = right_folders[folder]
                    # Get full file objects from organized data and sort by date
                    folder_files = []
                    for filename in files_in_folder:
                        for f in y0_organized.get(right_clusters[filename], []):
                            if f.get("original_filename", "") == filename:
                                folder_files.append(f)
                                break
                    
                    sorted_files = sorted(folder_files, key=parse_file_date, reverse=True)
                    
                    st.markdown(f"**📁 {folder}** ({len(files_in_folder)} files)")
                    
                    # Windows Explorer-style grid
                    st.markdown('<div class="file-grid-container">', unsafe_allow_html=True)
                    
                    # Header row
                    st.markdown("""
                    <div class="file-grid file-grid-header">
                        <div></div>
                        <div>Name</div>
                        <div>Date Created</div>
                        <div>Date Modified</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # File rows
                    for file in sorted_files:
                        original_filename = file.get("original_filename", "")
                        new_filename = file.get("filename", "")
                        create_date = file.get("create_date", "")
                        edit_date = file.get("edit_date", "")
                        file_preview = file.get("file_contents_preview", "")
                        
                        # Format dates
                        if create_date:
                            create_display = create_date[:16] if len(create_date) > 10 else create_date[:10]
                        else:
                            create_display = ""
                        if edit_date:
                            edit_display = edit_date[:16] if len(edit_date) > 10 else edit_date[:10]
                        else:
                            edit_display = ""
                        
                        # Determine file icon
                        file_ext = new_filename.split('.')[-1].lower() if '.' in new_filename else ""
                        file_icon = "📄"
                        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp']:
                            file_icon = "🖼️"
                        elif file_ext in ['pdf']:
                            file_icon = "📕"
                        elif file_ext in ['doc', 'docx']:
                            file_icon = "📘"
                        elif file_ext in ['xls', 'xlsx', 'csv']:
                            file_icon = "📊"
                        elif file_ext in ['txt', 'md', 'rtf']:
                            file_icon = "📝"
                        elif file_ext in ['py', 'js', 'java', 'cpp', 'c', 'ts', 'html', 'css']:
                            file_icon = "💻"
                        elif file_ext in ['zip', 'rar', '7z', 'tar', 'gz']:
                            file_icon = "📦"
                        elif file_ext in ['mp3', 'mp4', 'avi', 'wav', 'flac']:
                            file_icon = "🎵"
                        
                        # Show new filename if different from original, otherwise show original
                        display_name = new_filename if new_filename != original_filename else original_filename
                        name_display = f'<span class="file-name">{display_name}</span>'
                        if new_filename != original_filename:
                            name_display += f'<br><span class="original-name">Original filename: {original_filename}</span>'
                        
                        # Add preview if available
                        if file_preview:
                            preview_display = file_preview[:200] + "..." if len(file_preview) > 200 else file_preview
                            preview_display = preview_display.replace("<", "&lt;").replace(">", "&gt;")
                            name_display += f'<br><span class="original-name" style="font-size: 11px; color: #888; font-style: normal; margin-top: 4px; display: block;">{preview_display}</span>'
                        
                        st.markdown(f"""
                        <div class="file-grid file-item">
                            <div style="font-size: 20px;">{file_icon}</div>
                            <div>{name_display}</div>
                            <div class="file-date">{create_display}</div>
                            <div class="file-date">{edit_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Collect user preference
        with st.form(key=f"comparison_form_{current_comparison}"):
            preference = st.radio(
                "Which policy organizes these files better?",
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
                    "sampled_filenames": sampled_filenames,
                    "preference": preference,
                    "actual_preference": actual_preference,  # Internal tracking
                    "pred_on_left": pred_on_left,  # Track assignment for this comparison
                    "pred_clusters": pred_clusters,
                    "y0_clusters": y0_clusters,
                })
                st.session_state.form_results["final_evaluation"]["comparison_results"] = comparison_results
                st.rerun()
        
        return False, None
    
    # All comparisons completed
    st.success(f"✅ Completed all {num_comparisons} comparisons!")
    
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
    
    st.markdown("### Summary")
    st.markdown(f"- **Policy A wins**: {sum(1 for r in comparison_results if r['preference'] == 'Policy A')}")
    st.markdown(f"- **Policy B wins**: {sum(1 for r in comparison_results if r['preference'] == 'Policy B')}")
    st.markdown(f"- **No preference**: {ties}")
    st.markdown(f"- **Score**: {score:.3f} (win rate for final prediction)")
    
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

