import os


def get_poetry_python_path() -> str:
    """Get the path to the Poetry environment's Python interpreter."""
    import subprocess

    try:
        project_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))  # Go up two directories
        )
        # activate the poetry environment
        runtime_project = os.path.join(project_dir, 'dependencies')

        # Get the poetry env path
        result = subprocess.run(
            ['poetry', 'env', 'info', '--path'],
            capture_output=True,
            text=True,
            check=True,
            cwd=runtime_project,
        )

        env_path = result.stdout.strip()

        # Build path to Python interpreter based on OS
        python_path = os.path.join(env_path, 'bin')

        # convert relative path to absolute path
        if not os.path.isabs(python_path):
            python_path = os.path.join(runtime_project, python_path)

        if not os.path.exists(python_path):
            raise FileNotFoundError(f'Python interpreter not found at {python_path}')

        return python_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f'Poetry environment not found. Are you in a Poetry project? project path: {runtime_project} Error: {str(e)}'
        )
    except Exception as e:
        raise RuntimeError(f'Error getting Poetry Python path: {str(e)}')


def escape_value(value: str) -> str:
    # Escape special characters and wrap in quotes if contains spaces
    value = str(value).replace('"', '\\"')
    if ' ' in value or '(' in value or ')' in value:
        return f'"{value}"'
    return value
