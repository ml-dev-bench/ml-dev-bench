import os

from composio import action


@action(toolname='createoroverwritefile')
def create_or_overwrite_file(filepath: str, content: str) -> str:
    """
    Creates a new file with the given content or overwrites an existing file.

    :param filepath: The path where the file should be created/overwritten
    :param content: The content to write to the file
    :return status: A message confirming the file operation
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(os.path.dirname(filepath))
        # Write the content to the file
        with open(filepath, 'w') as f:
            f.write(content)
        print('File created successfully')
        return f'Successfully created/updated file at {filepath}'
    except Exception as e:
        return f'Error creating/updating file: {str(e)}'
