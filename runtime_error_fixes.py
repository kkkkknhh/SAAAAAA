"""
Runtime Error Fixes for Policy Analysis

This module contains fixes for three critical runtime errors:
1. 'bool' object is not iterable - Functions returning bool instead of list
2. 'str' object has no attribute 'text' - String passed where spacy object expected
3. can't multiply sequence by non-int of type 'float' - List multiplication by float

These fixes are applied defensively to prevent crashes in production.
"""

from typing import Any, List, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


def ensure_list_return(value: Any) -> List[Any]:
    """
    Ensure a value is a list, converting bool/None to empty list.
    
    Fixes: 'bool' object is not iterable
    
    Args:
        value: Value that should be a list
    
    Returns:
        Empty list if value is False/None/bool, otherwise the value as-is
    """
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, list):
        return value
    # If it's iterable but not a list, convert it
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def safe_text_extract(obj: Any) -> str:
    """
    Safely extract text from object that might be str or have .text attribute.
    
    Fixes: 'str' object has no attribute 'text'
    
    Args:
        obj: Object that is either str or has .text attribute (e.g., spacy Doc/Span)
    
    Returns:
        Extracted text string
    """
    # If it's already a string, return it
    if isinstance(obj, str):
        return obj
    
    # If it has a .text attribute, extract it
    if hasattr(obj, 'text'):
        text_value = getattr(obj, 'text')
        if isinstance(text_value, str):
            return text_value
        # If .text is callable (shouldn't be, but defensive)
        if callable(text_value):
            return str(obj)
    
    # Fallback: convert to string
    return str(obj)


def safe_weighted_multiply(items: Union[List[float], Any], weight: float) -> Union[List[float], Any]:
    """
    Safely multiply a list or array by a weight.
    
    Fixes: can't multiply sequence by non-int of type 'float'
    
    Args:
        items: List or array of numbers
        weight: Weight to multiply by
    
    Returns:
        New list/array with each element multiplied by weight
    """
    # If it's a numpy array, use numpy multiplication
    if HAS_NUMPY and np is not None and isinstance(items, np.ndarray):
        return items * weight
    
    # If it's a list, use list comprehension
    if isinstance(items, list):
        return [item * weight for item in items]
    
    # If it's something else iterable, convert and multiply
    try:
        return [item * weight for item in items]
    except (TypeError, ValueError):
        # If multiplication fails, return empty list
        return []


def safe_list_iteration(value: Any) -> List[Any]:
    """
    Ensure a value can be safely iterated over.
    
    Converts bool, None, or non-iterables to empty list.
    Handles the common error of trying to iterate over bool.
    
    Args:
        value: Value to iterate over
    
    Returns:
        Iterable list
    """
    # Reject booleans explicitly
    if isinstance(value, bool):
        return []
    
    # Handle None
    if value is None:
        return []
    
    # If it's already a list, return it
    if isinstance(value, list):
        return value
    
    # If it's a string, don't iterate over characters - return as single item
    if isinstance(value, str):
        return [value]
    
    # Try to convert to list
    try:
        return list(value)
    except (TypeError, ValueError):
        return []
