"""Task for implementing GRPO algorithm."""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ml_dev_bench.case import Case, CaseStatus
from ml_dev_bench.common.file_utils import copy_directory


class GRPOImplementationCase(Case):
    """Case for implementing GRPO algorithm."""

    def __init__(self):
        """Initialize the case."""
        super().__init__()
        self.workspace_dir = None

    def setup(self, workspace_dir: str) -> None:
        """Set up the workspace for the case.

        Args:
            workspace_dir: Directory to set up the workspace in.
        """
        self.workspace_dir = workspace_dir
        # Copy the setup workspace to the workspace directory
        copy_directory(
            os.path.join(os.path.dirname(__file__), "setup_workspace"),
            workspace_dir,
        )

    def get_status(self) -> CaseStatus:
        """Get the status of the case.

        Returns:
            The status of the case.
        """
        if not os.path.exists(os.path.join(self.workspace_dir, "grpo.py")):
            return CaseStatus.NOT_STARTED

        try:
            # Run the test file
            sys.path.insert(0, self.workspace_dir)
            import test_grpo

            test_grpo.run_tests()
            return CaseStatus.CORRECT
        except Exception as e:
            print(f"Error running tests: {e}")
            return CaseStatus.INCORRECT
        finally:
            sys.path.pop(0)

    def get_files_to_update(self) -> List[str]:
        """Get the list of files that need to be updated.

        Returns:
            List of files that need to be updated.
        """
        return ["grpo.py"]
