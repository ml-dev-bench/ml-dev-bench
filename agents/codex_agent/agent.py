"""Codex agent implementation for ML-Dev-Bench."""

import asyncio
import json
import os
import shlex
from typing import Any, Dict, Optional

from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry
from runtime.environments.shell_setup import get_poetry_python_path


@EvalRegistry.register_agent
class CodexAgent(BaseAgent):
    """Agent that runs the Codex CLI in non-interactive mode."""

    agent_id = 'codex_agent'
    description = 'Runs OpenAI Codex CLI against ML-Dev-Bench tasks'

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def uses_litellm(self) -> bool:
        # Codex CLI directly calls model providers; it does not use litellm here.
        return False

    @staticmethod
    def _extract_text_from_json_event(event: dict[str, Any]) -> Optional[str]:
        """Best-effort extraction of assistant text from a Codex JSON event."""
        for key in ('text', 'message', 'content', 'output'):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, dict):
                        text_val = item.get('text')
                        if isinstance(text_val, str) and text_val.strip():
                            parts.append(text_val.strip())
                if parts:
                    return '\n'.join(parts)
        return None

    @staticmethod
    def _is_retryable_failure(stdout: str, stderr: str, response: str | None) -> bool:
        """Return True for transient failures worth retrying."""
        haystack = '\n'.join([stdout or '', stderr or '', response or '']).lower()
        retry_markers = [
            'invalid_encrypted_content',
            'encrypted content',
            'could not be decrypted or parsed',
            'reconnecting...',
            'stream disconnected',
            '502',
            '503',
            '504',
            'connection reset',
            'connection closed',
            'timed out',
            'timeout',
        ]
        return any(marker in haystack for marker in retry_markers)

    @staticmethod
    def _build_runtime_preamble() -> str:
        """Instruction prefix to reduce environment mismatch failures."""
        return (
            "Environment instructions:\\n"
            "- Use the preconfigured ML-Dev-Bench Python runtime already available in this workspace shell.\\n"
            "- Do NOT create a new virtual environment.\\n"
            "- Before running task code, verify Python/tooling with commands like:\\n"
            "  - which python\\n"
            "  - python -c \"import torch; print(torch.__version__)\"\\n"
            "- Prefer `python` over hardcoded interpreter paths unless required by the task.\\n\\n"
        )

    async def run(self, task: str) -> Dict[str, Any]:
        workspace_dir = self.config.workspace_dir
        if workspace_dir is None:
            return {
                'success': False,
                'error': 'workspace_dir is required for CodexAgent',
            }

        model_name = self.config.model_name or self.config.config.get('model_name')
        if not model_name:
            return {
                'success': False,
                'error': 'model_name is required for CodexAgent',
            }

        model = model_name.split('/')[-1]
        timeout_sec = int(self.config.config.get('timeout_sec', 3600))
        reasoning_effort = self.config.config.get('reasoning_effort')

        if not os.getenv('OPENAI_API_KEY'):
            return {
                'success': False,
                'error': 'OPENAI_API_KEY is not set; Codex CLI cannot authenticate',
            }

        env = os.environ.copy()
        # Mirror runtime tool behavior by prioritizing the runtime Poetry Python bin.
        # This helps Codex shell commands resolve torch-enabled Python consistently.
        try:
            runtime_python_bin = get_poetry_python_path()
            env['PATH'] = f"{runtime_python_bin}:{env.get('PATH', os.environ.get('PATH', ''))}"
        except Exception:
            # Best effort only; keep existing PATH if runtime env discovery fails.
            pass
        env['CODEX_HOME'] = str(workspace_dir / '.codex')
        os.makedirs(env['CODEX_HOME'], exist_ok=True)

        # Match Harbor Codex auth flow: Codex reads OPENAI_API_KEY from CODEX_HOME/auth.json.
        auth_path = os.path.join(env['CODEX_HOME'], 'auth.json')
        with open(auth_path, 'w', encoding='utf-8') as f:
            json.dump({'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', '')}, f)

        full_task = self._build_runtime_preamble() + task
        escaped_task = shlex.quote(full_task)
        reasoning_flag = (
            f'-c model_reasoning_effort={shlex.quote(str(reasoning_effort))} '
            if reasoning_effort
            else ''
        )

        # `--dangerously-bypass-approvals-and-sandbox` keeps execution non-interactive.
        command = (
            'codex exec '
            '--dangerously-bypass-approvals-and-sandbox '
            '--skip-git-repo-check '
            f'--model {shlex.quote(model)} '
            '--json '
            f'{reasoning_flag}'
            '-- '
            f'{escaped_task}'
        )

        # Defaults are code-level so retry robustness works even without Hydra overrides.
        max_retries = int(self.config.config.get('max_retries', 4))
        base_backoff_sec = float(self.config.config.get('retry_backoff_sec', 2.0))
        max_attempts = max(1, max_retries + 1)
        last_result: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            attempt_env = env.copy()
            # New CODEX_HOME per attempt helps avoid state/session corruption carrying over.
            attempt_env['CODEX_HOME'] = str(workspace_dir / f'.codex-attempt-{attempt}')
            os.makedirs(attempt_env['CODEX_HOME'], exist_ok=True)
            auth_path = os.path.join(attempt_env['CODEX_HOME'], 'auth.json')
            with open(auth_path, 'w', encoding='utf-8') as f:
                json.dump({'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', '')}, f)

            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(workspace_dir),
                    env=attempt_env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                return {
                    'success': False,
                    'error': (
                        'codex CLI not found. Install it first and ensure `codex` is on PATH. '
                        'See https://github.com/openai/codex for install instructions.'
                    ),
                }

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {
                    'success': False,
                    'error': f'Codex command timed out after {timeout_sec} seconds',
                    'attempt': attempt,
                    'max_attempts': max_attempts,
                }

            stdout = stdout_bytes.decode(errors='replace')
            stderr = stderr_bytes.decode(errors='replace')

            parsed_events: list[dict[str, Any]] = []
            final_response: Optional[str] = None

            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        parsed_events.append(event)
                        maybe_text = self._extract_text_from_json_event(event)
                        if maybe_text:
                            final_response = maybe_text
                except json.JSONDecodeError:
                    # Keep going: some Codex output may not be JSON.
                    pass

            if not final_response:
                final_response = stdout.strip() or stderr.strip()

            attempt_result = {
                'success': proc.returncode == 0,
                'response': final_response,
                'return_code': proc.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'events': parsed_events,
                'command': command,
                'attempt': attempt,
                'max_attempts': max_attempts,
                'codex_home': attempt_env['CODEX_HOME'],
            }

            if attempt_result['success']:
                return attempt_result

            retryable = self._is_retryable_failure(stdout, stderr, final_response)
            attempt_result['retryable'] = retryable
            last_result = attempt_result

            if not retryable or attempt >= max_attempts:
                return attempt_result

            backoff = base_backoff_sec * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

        return last_result
