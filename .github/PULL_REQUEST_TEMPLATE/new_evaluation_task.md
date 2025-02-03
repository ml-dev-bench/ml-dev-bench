## New Evaluation Task: [Task Name]

### Task Overview
<!-- Brief description of what this task evaluates -->

### Implementation Checklist
- [ ] Created directory `ml_dev_bench/cases/[task_name]/` with:
  - [ ] `task.txt` with clear agent instructions
  - [ ] `[task_name]_eval.py` with validation logic
  - [ ] `config.yaml` (if needed)
  - [ ] `setup_workspace/` directory with:
    - [ ] Input files (if needed)
    - [ ] Test scripts (if needed)

### Validation Logic
<!-- Describe your success criteria and how it's implemented -->
```python
# Brief example of key validation checks
```

### Testing
<!-- Describe how you've tested this task -->
- [ ] Tested with expected success case
- [ ] Tested with expected failure cases
- [ ] Verified test scripts work as expected
- [ ] Verified validation logic catches common errors

### Task Structure
<!-- Fill in the actual content for your task -->
<details>
<summary>task.txt content</summary>

```
[Paste your task.txt content here]
```
</details>

<details>
<summary>Test script example (if applicable)</summary>

```python
[Paste relevant test script content]
```
</details>

<details>
<summary>Validation implementation highlights</summary>

```python
[Paste key sections of your validation logic]
```
</details>

### Reference Examples
I have referenced these example tasks:
- [ ] [hello_world](https://github.com/ml-dev-bench/ml-dev-bench/tree/main/ml_dev_bench/cases/hello_world) for basic task structure
- [ ] [nan_losses](https://github.com/ml-dev-bench/ml-dev-bench/tree/main/ml_dev_bench/cases/debugging/nan_losses) for tasks requiring setup files and test scripts

### Additional Notes
<!-- Any other relevant information -->
- Does this task require specific dependencies?
- Are there any special setup requirements?
- Expected runtime considerations?
