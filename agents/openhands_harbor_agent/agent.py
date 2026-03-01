"""Harbor-style OpenHands parity agent.

This agent runs OpenHands in an isolated virtual environment to avoid
dependency conflicts with the main ml-dev-bench Poetry environment.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_agent
class OpenHandsHarborParityAgent(BaseAgent):
    """Run OpenHands headless with Harbor-like environment settings."""

    agent_id = 'openhands_harbor_parity_agent'
    description = 'Runs OpenHands in Harbor-like local-runtime mode for parity checks'

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        repo_root = Path(__file__).resolve().parents[2]
        self._default_venv_dir = repo_root / '.venv-openhands-harbor-parity'

    def uses_litellm(self) -> bool:
        return False

    def _resolve_api_key(self, env: Dict[str, str], model_name: str) -> str | None:
        if env.get('LLM_API_KEY'):
            return env['LLM_API_KEY']

        lowered = model_name.lower()
        candidates: list[str] = []
        if lowered.startswith('openai/') or '/openai/' in lowered:
            candidates.extend(['OPENAI_API_KEY'])
        elif lowered.startswith('anthropic/') or '/anthropic/' in lowered:
            candidates.extend(['ANTHROPIC_API_KEY'])

        candidates.extend(
            ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'AZURE_API_KEY']
        )

        for candidate in candidates:
            value = env.get(candidate)
            if value:
                return value

        return None

    def _ensure_openhands_venv(self, venv_dir: Path, openhands_version: str) -> Path:
        """Create/update an isolated OpenHands venv and return its python binary."""
        venv_python = venv_dir / 'bin' / 'python'
        version_marker = venv_dir / '.openhands-version'
        expected_marker = f'openhands-ai=={openhands_version}'

        needs_install = True
        if venv_python.exists() and version_marker.exists():
            if version_marker.read_text(encoding='utf-8').strip() == expected_marker:
                needs_install = False

        if not venv_python.exists():
            subprocess.run(
                [sys.executable, '-m', 'venv', str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            needs_install = True

        if needs_install:
            subprocess.run(
                [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    str(venv_python),
                    '-m',
                    'pip',
                    'install',
                    f'openhands-ai=={openhands_version}',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            version_marker.write_text(expected_marker, encoding='utf-8')

        return venv_python

    async def run(self, task: str) -> Dict[str, Any]:
        workspace_dir = self.config.workspace_dir
        if workspace_dir is None:
            return {'success': False, 'error': 'workspace_dir is required'}

        model_name = self.config.model_name or self.config.config.get('model_name')
        if not model_name:
            return {'success': False, 'error': 'model_name is required'}

        openhands_version = str(self.config.config.get('openhands_version', '1.4.0'))
        timeout_sec = int(self.config.config.get('timeout_sec', 3600))
        venv_dir = Path(
            self.config.config.get('openhands_venv_dir', str(self._default_venv_dir))
        )
        logs_dir = workspace_dir / '.openhands-harbor-logs'
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            venv_python = await asyncio.to_thread(
                self._ensure_openhands_venv, venv_dir, openhands_version
            )
        except subprocess.CalledProcessError as exc:
            return {
                'success': False,
                'error': f'Failed to prepare OpenHands venv: {exc.stderr or exc.stdout}',
            }
        except Exception as exc:  # pragma: no cover - defensive guard
            return {'success': False, 'error': f'Failed to prepare OpenHands venv: {exc}'}

        env = os.environ.copy()
        llm_api_key = self._resolve_api_key(env, model_name)
        if not llm_api_key:
            return {
                'success': False,
                'error': (
                    'No API key found. Set LLM_API_KEY or provider key env vars '
                    '(OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY).'
                ),
            }

        env['LLM_API_KEY'] = llm_api_key
        env['LLM_MODEL'] = model_name
        env['AGENT_ENABLE_PROMPT_EXTENSIONS'] = 'false'
        env['AGENT_ENABLE_BROWSING'] = 'false'
        env['ENABLE_BROWSER'] = 'false'
        env['SANDBOX_ENABLE_AUTO_LINT'] = 'true'
        env['SKIP_DEPENDENCY_CHECK'] = '1'
        env['RUN_AS_OPENHANDS'] = 'false'
        env['RUNTIME'] = 'local'
        env['SAVE_TRAJECTORY_PATH'] = str(logs_dir / 'openhands.trajectory.json')
        env['FILE_STORE'] = 'local'
        env['FILE_STORE_PATH'] = str(logs_dir)
        env['LLM_LOG_COMPLETIONS'] = 'true'
        env['LLM_LOG_COMPLETIONS_FOLDER'] = str(logs_dir / 'completions')
        env['SANDBOX_VOLUMES'] = f'{workspace_dir}:/workspace:rw'

        if self.config.config.get('api_base'):
            env['LLM_BASE_URL'] = str(self.config.config.get('api_base'))
        elif env.get('OPENAI_API_BASE'):
            env['LLM_BASE_URL'] = env['OPENAI_API_BASE']

        if self.config.config.get('reasoning_effort'):
            env['LLM_REASONING_EFFORT'] = str(
                self.config.config.get('reasoning_effort')
            )

        cmd = [str(venv_python), '-m', 'openhands.core.main', f'--task={task}']

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workspace_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                'success': False,
                'error': f'OpenHands timed out after {timeout_sec} seconds',
                'command': ' '.join(cmd),
            }

        stdout = stdout_bytes.decode(errors='replace')
        stderr = stderr_bytes.decode(errors='replace')
        (logs_dir / 'openhands.stdout.txt').write_text(stdout, encoding='utf-8')
        (logs_dir / 'openhands.stderr.txt').write_text(stderr, encoding='utf-8')

        return {
            'success': proc.returncode == 0,
            'return_code': proc.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'command': ' '.join(cmd),
            'logs_dir': str(logs_dir),
            'openhands_version': openhands_version,
        }

