"""
PydanticAI Model implementation for GitHub Copilot SDK.

This module provides a drop-in replacement Model implementation that uses
GitHub Copilot SDK as the backend, allowing seamless integration with PydanticAI.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    RetryPromptPart,
    ModelResponseStreamEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

try:
    from copilot import CopilotClient, PermissionHandler
    from copilot.types import SessionConfig
except ImportError as e:
    raise ImportError(
        "Please install github-copilot-sdk to use CopilotModel: "
        "pip install github-copilot-sdk"
    ) from e


class CopilotStreamedResponse(StreamedResponse):
    """Implementation of StreamedResponse for CopilotModel."""

    def __init__(
        self,
        stream: AsyncIterator[ModelResponseStreamEvent],
        model_name: str,
        model_request_parameters: ModelRequestParameters,
    ):
        super().__init__(model_request_parameters)
        self._stream = stream
        self._model_name = model_name
        self._timestamp = datetime.now(timezone.utc)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "copilot"

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._stream:
            yield event


class CopilotModelSettings(ModelSettings, total=False):
    """Settings for Copilot model requests."""
    copilot_model: str
    copilot_provider: dict[str, Any]
    copilot_working_directory: str
    copilot_available_tools: list[str]
    copilot_excluded_tools: list[str]
    copilot_timeout: float


def _find_copilot_cli_path(start_dir: Path | None = None) -> str | None:
    """Find the `copilot` CLI executable in common locations."""
    env_path = os.environ.get("COPILOT_CLI_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p)

    names = ["copilot"]
    if os.name == "nt":
        names = ["copilot.exe", "copilot.cmd", "copilot.bat", "copilot"]

    start = start_dir or Path.cwd()

    for base in [start, *start.parents]:
        for name in names:
            p = base / "node_modules" / ".bin" / name
            if p.exists():
                return str(p)
        for name in names:
            p = base / "node_modules" / ".pnpm" / "node_modules" / ".bin" / name
            if p.exists():
                return str(p)

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    return None


@dataclass
class CopilotModel(Model):
    """PydanticAI Model implementation using GitHub Copilot SDK."""

    cli_path: str | None = None
    model_name_value: str = "gpt-4"
    working_directory: str | None = None
    timeout: float = 300.0

    _client: Optional[Any] = None
    _session: Optional[Any] = None
    _session_lock: Optional[asyncio.Lock] = None

    @property
    def model_name(self) -> str:
        return self.model_name_value

    @property
    def system(self) -> str:
        return "copilot"

    def __post_init__(self):
        if self.cli_path is None:
            self.cli_path = _find_copilot_cli_path()
            if self.cli_path is None:
                raise RuntimeError(
                    "Copilot CLI not found. Please install it or provide `cli_path=` explicitly."
                )
        self._session_lock = asyncio.Lock()

    async def _ensure_client(self):
        if self._client is None:
            client_opts = {"cli_path": self.cli_path} if self.cli_path else {}
            self._client = CopilotClient(client_opts)
            await self._client.start()

    async def _ensure_session(
        self,
        system_prompt: str | None = None,
        model_settings: ModelSettings | None = None,
        force_new: bool = False,
    ):
        async with self._session_lock:
            if force_new and self._session:
                try:
                    await self._session.destroy()
                except Exception:
                    pass
                self._session = None

            if self._session is None:
                await self._ensure_client()
                session_config: SessionConfig = {
                    "on_permission_request": PermissionHandler.approve_all,
                }
                
                model = self.model_name_value
                if model_settings and "copilot_model" in model_settings:
                    model = model_settings["copilot_model"]
                session_config["model"] = model

                if model_settings and "copilot_provider" in model_settings:
                    session_config["provider"] = model_settings["copilot_provider"]

                if system_prompt:
                    session_config["system_message"] = system_prompt

                working_dir = self.working_directory
                if model_settings and "copilot_working_directory" in model_settings:
                    working_dir = model_settings["copilot_working_directory"]
                if working_dir:
                    session_config["working_directory"] = working_dir

                if model_settings:
                    if "copilot_available_tools" in model_settings:
                        session_config["available_tools"] = model_settings[
                            "copilot_available_tools"
                        ]
                    if "copilot_excluded_tools" in model_settings:
                        session_config["excluded_tools"] = model_settings[
                            "copilot_excluded_tools"
                        ]

                self._session = await self._client.create_session(session_config)

    def _convert_messages_to_prompt(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, str]:
        """Convert standard PydanticAI messages to a prompt string."""
        system_prompt = None
        user_parts = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt = part.content
                    elif isinstance(part, UserPromptPart):
                        user_parts.append(part.content)
                    elif isinstance(part, ToolReturnPart):
                        user_parts.append(
                            f"Tool '{part.tool_name}' returned: {part.content}"
                        )
                    elif isinstance(part, RetryPromptPart):
                        # Construct a clear retry message
                        error_msg = part.content
                        details = ""
                        if isinstance(error_msg, list):
                            # Handle list of Pydantic validation errors
                            try:
                                details = "\n".join(
                                    f"- {e.get('loc', 'root')}: {e.get('msg', '')}"
                                    for e in error_msg
                                    if isinstance(e, dict)
                                )
                            except Exception:
                                details = str(error_msg)
                        else:
                            details = str(error_msg)
                        
                        user_parts.append(
                            f"\nERROR: The previous response output was invalid or failed schema validation.\n"
                            f"Validation Failures:\n{details}\n\n"
                            f"CRITICAL: You MUST fix these errors in your next response. "
                            f"Return ONLY valid JSON that matches the schema."
                        )

            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        user_parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        user_parts.append(
                            f"Assistant called tool '{part.tool_name}' with args: {part.args}"
                        )

        user_prompt = "\n".join(user_parts) if user_parts else ""
        return system_prompt, user_prompt

    def _build_schema_instruction(
        self, model_request_parameters: ModelRequestParameters
    ) -> str:
        """Builds instruction to force JSON output for structured result agents."""
        if not model_request_parameters.output_tools:
            return ""

        output_tool = model_request_parameters.output_tools[0]
        schema = output_tool.parameters_json_schema
        schema_json = json.dumps(schema, indent=2)
        
        return (
            f"\n\n*** SYSTEM INSTRUCTION ***\n"
            f"You MUST call the tool '{output_tool.name}' to provide the final answer.\n"
            f"This tool requires strict JSON arguments matching the schema below.\n"
            f"Output a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Do not include explanation text outside the JSON if possible.\n"
            f"You may use markdown blocks (```json ... ```)."
        )

    def _extract_json_string(self, content: str) -> str | None:
        """Robustly extract the first valid JSON object using stack-based matching."""
        content = content.replace("```json", "").replace("```", "").strip()
        
        start = content.find('{')
        if start == -1:
            return None
            
        depth = 0
        in_string = False
        escape = False
        
        # Scan from the first brace
        for i, char in enumerate(content[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        # Found the matching closing brace
                        return content[start : i + 1]
        
        return None

    def _extract_structured_output(self, content: str) -> dict | None:
        """Extract structured JSON output from agent response."""
        content = content.strip()
        
        # 1. Try stack-based extraction (Most robust for mixed text)
        json_str = self._extract_json_string(content)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 2. Try regex as fallback
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Last resort: try entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        return None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        if model_request_parameters.output_tools:
            user_prompt += self._build_schema_instruction(model_request_parameters)

        await self._ensure_session(system_prompt, model_settings, force_new=True)

        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        response = await self._session.send_and_wait(
            {"prompt": user_prompt}, timeout=timeout
        )

        if not response:
            raise RuntimeError("No response from Copilot agent")

        content = response.data.content
        parts: list[ModelResponsePart] = []

        if model_request_parameters.output_tools:
            structured_output = self._extract_structured_output(content)
            if structured_output:
                output_tool = model_request_parameters.output_tools[0]
                parts.append(
                    ToolCallPart(
                        tool_name=output_tool.name,
                        args=structured_output,
                        tool_call_id=str(uuid.uuid4()),
                    )
                )
            else:
                parts.append(TextPart(content=content))
        else:
            parts.append(TextPart(content=content))

        usage_info = RequestUsage()
        return ModelResponse(parts=parts, usage=usage_info)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        if model_request_parameters.output_tools:
            user_prompt += self._build_schema_instruction(model_request_parameters)

        await self._ensure_session(system_prompt, model_settings, force_new=True)

        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        event_queue = asyncio.Queue()

        def on_event(event):
            event_queue.put_nowait(event)

        unsubscribe = self._session.on(on_event)

        try:
            await self._session.send({"prompt": user_prompt})

            async def stream_generator():
                full_content = ""
                
                # Start with an empty text part so we can send deltas. This is vital for streaming.
                yield PartStartEvent(index=0, part=TextPart(content=""))

                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=timeout
                        )

                        if event.type.value == "assistant.message":
                            chunk_content = event.data.content
                            full_content += chunk_content
                            
                            # Update the text part with delta
                            yield PartDeltaEvent(
                                index=0, 
                                delta=TextPartDelta(content_delta=chunk_content)
                            )

                        elif event.type.value == "session.idle":
                            break

                        elif event.type.value == "session.error":
                            # We might log this instead of crashing if partial content is meaningful
                            raise RuntimeError(f"Session error: {event.data.message}")

                    except asyncio.TimeoutError:
                        break

                # At end of stream, try to parse tool call if output_tools are present
                if model_request_parameters.output_tools and full_content:
                    structured_output = self._extract_structured_output(full_content)
                    if structured_output:
                        output_tool = model_request_parameters.output_tools[0]
                        # Yield a NEW part for the tool call
                        yield PartStartEvent(
                            index=1,
                            part=ToolCallPart(
                                tool_name=output_tool.name,
                                args=structured_output,
                                tool_call_id=str(uuid.uuid4()),
                            )
                        )

            usage_info = RequestUsage()
            response = CopilotStreamedResponse(
                stream=stream_generator(),
                model_name=self.model_name_value,
                model_request_parameters=model_request_parameters,
            )
            response._usage = usage_info
            
            yield response

        finally:
            unsubscribe()

    async def cleanup(self):
        if self._session:
            try:
                await self._session.destroy()
            except Exception:
                pass
            self._session = None

        if self._client:
            try:
                await self._client.stop()
            except Exception:
                pass
            self._client = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        await self.cleanup()