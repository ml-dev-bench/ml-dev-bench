# ruff: noqa
# a test script to check if the labelbox connection is working
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file"""
    # Look for .env.runtime file in parent directories until found
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        env_file = current_dir / '.env.runtime'
        if env_file.exists():
            load_dotenv(env_file)
            return True
        current_dir = current_dir.parent
    return False


def ensure_labelbox_installed() -> bool:
    """
    Check if labelbox is installed, if not, install it using pip.

    Returns:
        bool: True if labelbox is available (either pre-installed or successfully installed),
              False if installation failed
    """

    print('Labelbox package not found. Attempting to install...')
    try:
        subprocess.check_call(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                'python-dotenv',
                'labelbox[data]',
            ]
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f'Failed to install required packages: {str(e)}')
        return False


def test_labelbox_connection(api_key: Optional[str] = None) -> Dict:
    """
    Test Labelbox API connection by creating a test project.

    Args:
        api_key: Labelbox API key. If None, tries to get from environment variable LABELBOX_API_KEY

    Returns:
        Dict containing connection test results and project details if successful
    """
    # First ensure required packages are installed
    if not ensure_labelbox_installed():
        return {'success': False, 'error': 'Failed to install required packages'}

    # Load environment variables
    load_env()

    # Now we can safely import labelbox
    import labelbox as lb

    try:
        # Get API key from environment if not provided
        api_key = api_key or os.getenv('LABELBOX_API_KEY')
        if not api_key:
            return {
                'success': False,
                'error': 'No Labelbox API key provided or found in environment variables',
            }

        # Initialize client
        client = lb.Client(api_key=api_key)

        # Create a test project
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_name = f'test_connection_{timestamp}'

        # Create project with required media_type parameter
        project = client.create_project(
            name=project_name, media_type=lb.MediaType.Image
        )

        # Test project creation
        if not project or not project.uid:
            return {'success': False, 'error': 'Failed to create test project'}

        # Clean up - delete test project
        project.delete()

        return {
            'success': True,
            'message': 'Successfully connected to Labelbox API',
            'test_project': {'name': project_name, 'created_at': timestamp},
        }

    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}


if __name__ == '__main__':
    # Test the connection and save results
    results = test_labelbox_connection()

    # Save results to file
    output_file = 'labelbox_connection_test.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if results['success']:
        print(f'✓ Labelbox connection test passed: {results["message"]}')
    else:
        print(f'✗ Labelbox connection test failed: {results["error"]}')
