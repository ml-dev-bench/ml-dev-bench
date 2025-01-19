"""Utility functions for agents."""

from typing import Any, Dict, List


def create_message_content(text: str) -> List[Dict[str, Any]]:
    """Create a standardized message content structure.

    Args:
        text: The text content of the message.

    Returns:
        A list containing a single dictionary with the standardized message structure.
    """
    return [
        {
            'text': text,
            'type': 'text',
            'cache_control': {'type': 'ephemeral'},
        }
    ]
