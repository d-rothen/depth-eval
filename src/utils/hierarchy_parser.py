"""
Hierarchy Parser Module

Provides utilities for parsing and manipulating hierarchically structured data
with nested children dictionaries and file entries.
"""

from typing import Any, Optional


def get_matches(input_hierarchies: list[dict]) -> list[dict]:
    """
    Extract paths from multiple hierarchical JSON objects.

    Args:
        input_hierarchies: List of hierarchical objects with the same structure
                          but potentially different file paths.

    Returns:
        List of match objects containing:
        - hierarchy: List of keys forming the path to this entry
        - id: The matched file id
        - paths: List of paths from each input hierarchy (None if not found)
    """
    if not input_hierarchies:
        return []

    # Collect all file entries with their hierarchy paths from all inputs
    all_entries: dict[tuple[tuple[str, ...], str], list[Optional[str]]] = {}

    for idx, hierarchy in enumerate(input_hierarchies):
        entries = _collect_file_entries(hierarchy)
        for hierarchy_path, file_id, path in entries:
            key = (tuple(hierarchy_path), file_id)
            if key not in all_entries:
                all_entries[key] = [None] * len(input_hierarchies)
            all_entries[key][idx] = path

    # Convert to output format
    results = []
    for (hierarchy_path, file_id), paths in all_entries.items():
        results.append({
            "hierarchy": list(hierarchy_path),
            "id": file_id,
            "paths": paths
        })

    return results


def _collect_file_entries(
    node: dict,
    current_path: Optional[list[str]] = None
) -> list[tuple[list[str], str, str]]:
    """
    Recursively collect all file entries from a hierarchy node.

    Returns:
        List of tuples: (hierarchy_path, file_id, file_path)
    """
    if current_path is None:
        current_path = []

    entries = []

    # Process files at this level
    files = node.get("files", [])
    for file_entry in files:
        if isinstance(file_entry, dict) and "path" in file_entry and "id" in file_entry:
            entries.append((current_path.copy(), file_entry["id"], file_entry["path"]))

    # Recurse into children
    children = node.get("children", {})
    if isinstance(children, dict):
        for child_name, child_node in children.items():
            if isinstance(child_node, dict):
                child_path = current_path + [child_name]
                entries.extend(_collect_file_entries(child_node, child_path))

    return entries


def create_map(input_hierarchy: dict, separator: str = "/") -> dict[str, str]:
    """
    Create a flat mapping from a hierarchical structure.

    Traverses the hierarchy and creates keys by joining the hierarchy path
    and file id with the separator, mapping to the corresponding file path.

    Args:
        input_hierarchy: The hierarchical object to traverse
        separator: The string used to join hierarchy levels and id (default: "/")

    Returns:
        Dict mapping "level1{sep}level2{sep}...{sep}id" -> path

    Example:
        Given hierarchy with path ["images", "cats"], id "cat_001", path "/data/cat.jpg"
        and separator "/", returns: {"images/cats/cat_001": "/data/cat.jpg"}
    """
    entries = _collect_file_entries(input_hierarchy)
    result = {}

    for hierarchy_path, file_id, path in entries:
        key_parts = hierarchy_path + [file_id]
        key = separator.join(key_parts)
        result[key] = path

    return result


def set_value(
    target: dict,
    hierarchy: list[str],
    file_id: str,
    value: Any
) -> dict:
    """
    Place a value object into the target hierarchy at the specified location.

    Creates intermediate hierarchy levels if they don't exist.
    Adds or updates the file entry with the given id.

    Args:
        target: The target object to mutate
        hierarchy: List of keys forming the path (e.g., ["level_1", "level_2"])
        file_id: The id of the file entry to set
        value: The value object to place (should contain at least 'path')

    Returns:
        The mutated target object
    """
    # Navigate/create the hierarchy path
    current = target

    for level in hierarchy:
        if "children" not in current:
            current["children"] = {}
        if level not in current["children"]:
            current["children"][level] = {}
        current = current["children"][level]

    # Ensure files array exists
    if "files" not in current:
        current["files"] = []

    # Find existing entry with this id or add new one
    for i, file_entry in enumerate(current["files"]):
        if isinstance(file_entry, dict) and file_entry.get("id") == file_id:
            current["files"][i] = value
            return target

    # No existing entry found, append new one
    current["files"].append(value)
    return target


def retrieve_value(
    source: dict,
    hierarchy: list[str],
    file_id: str
) -> Optional[Any]:
    """
    Retrieve the value at a specific hierarchy path and file id.

    Args:
        source: The source object to search
        hierarchy: List of keys forming the path
        file_id: The id of the file entry to retrieve

    Returns:
        The file entry object if found, None otherwise
    """
    current = source

    # Navigate the hierarchy
    for level in hierarchy:
        children = current.get("children", {})
        if not isinstance(children, dict) or level not in children:
            return None
        current = children[level]
        if not isinstance(current, dict):
            return None

    # Search for file with matching id
    files = current.get("files", [])
    for file_entry in files:
        if isinstance(file_entry, dict) and file_entry.get("id") == file_id:
            return file_entry

    return None


def retrieve_values(
    source: dict,
    hierarchy: list[str]
) -> Optional[dict]:
    """
    Retrieve the mapping (node) at a specific hierarchy level.

    Args:
        source: The source object to search
        hierarchy: List of keys forming the path

    Returns:
        The node at that hierarchy level, or None if path doesn't exist
    """
    current = source

    # Navigate the hierarchy
    for level in hierarchy:
        children = current.get("children", {})
        if not isinstance(children, dict) or level not in children:
            return None
        current = children[level]
        if not isinstance(current, dict):
            return None

    return current
