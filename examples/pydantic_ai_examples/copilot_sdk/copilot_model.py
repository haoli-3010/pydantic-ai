"""
PydanticAI Model implementation for GitHub Copilot SDK.

This module provides a drop-in replacement Model implementation that uses
GitHub Copilot SDK as the backend, allowing seamless integration with PydanticAI.

Example:
    >>> from pydantic_ai import Agent
    >>> from copilot_model import CopilotModel
    >>>
    >>> model = CopilotModel(cli_path="/path/to/copilot")
    >>> agent = Agent(model, result_type=MySchema)
    >>> result = await agent.run("Your prompt")
"""

from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

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


class CopilotModelSettings(ModelSettings, total=False):
    """Settings for Copilot model requests."""

    copilot_model: str
    """The model to use (e.g., 'gpt-4', 'claude-3-5-sonnet')"""

    copilot_provider: dict[str, Any]
    """Provider configuration for BYOK (Bring Your Own Key)"""

    copilot_working_directory: str
    """Working directory for file operations"""

    copilot_available_tools: list[str]
    """List of copilot CLI tools to enable"""

    copilot_excluded_tools: list[str]
    """List of copilot CLI tools to disable"""

    copilot_timeout: float
    """Timeout in seconds for agent responses"""


@dataclass
class CopilotModel(Model):
    """PydanticAI Model implementation using GitHub Copilot SDK.

    This allows using Copilot SDK as a drop-in replacement for other models in PydanticAI.

    Example:
        >>> from pydantic_ai import Agent
        >>> from copilot_model import CopilotModel
        >>>
        >>> model = CopilotModel(cli_path="/path/to/copilot", model_name="gpt-4")
        >>> agent = Agent(model, result_type=WeatherResponse)
        >>> result = await agent.run("What's the weather?")
    """

    cli_path: str | None = None
    """Path to copilot CLI binary. If None, will try to find it in PATH."""

    model_name_value: str = "gpt-4"
    """Default model to use"""

    working_directory: str | None = None
    """Default working directory for file operations"""

    timeout: float = 300.0
    """Default timeout for requests"""

    _client: Optional[Any] = None  # CopilotClient
    _session: Optional[Any] = None  # CopilotSession
    _session_lock: Optional[asyncio.Lock] = None

    @property
    def model_name(self) -> str:
        """The model name."""
        return self.model_name_value

    @property
    def system(self) -> str:
        """The model provider system name."""
        return "copilot"

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.cli_path is None:
            self.cli_path = shutil.which("copilot")
            if self.cli_path is None:
                raise RuntimeError(
                    "Copilot CLI not found in PATH. Please install it or provide cli_path."
                )
        self._session_lock = asyncio.Lock()

    async def _ensure_client(self):
        """Ensure Copilot client is started."""
        if self._client is None:
            client_opts = {"cli_path": self.cli_path} if self.cli_path else {}
            self._client = CopilotClient(client_opts)
            await self._client.start()

    async def _ensure_session(
        self,
        system_prompt: str | None = None,
        model_settings: ModelSettings | None = None,
    ):
        """Ensure a Copilot session exists."""
        async with self._session_lock:
            if self._session is None:
                await self._ensure_client()

                session_config: SessionConfig = {
                    "on_permission_request": PermissionHandler.approve_all,
                }

                # Get model from settings or use default
                model = self.model_name_value
                if model_settings and "copilot_model" in model_settings:
                    model = model_settings["copilot_model"]
                session_config["model"] = model

                # Add provider if specified (BYOK)
                if model_settings and "copilot_provider" in model_settings:
                    session_config["provider"] = model_settings["copilot_provider"]

                # Add system message if provided
                if system_prompt:
                    session_config["system_message"] = system_prompt

                # Add working directory
                working_dir = self.working_directory
                if model_settings and "copilot_working_directory" in model_settings:
                    working_dir = model_settings["copilot_working_directory"]
                if working_dir:
                    session_config["working_directory"] = working_dir

                # Add tool filtering
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
        """Convert PydanticAI messages to Copilot SDK format.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
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
                        # Include tool results in the prompt
                        user_parts.append(
                            f"Tool {part.tool_name} returned: {part.content}"
                        )
            elif isinstance(msg, ModelResponse):
                # Include previous assistant responses
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        user_parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        user_parts.append(
                            f"Assistant called tool {part.tool_name} with args: {part.args}"
                        )

        user_prompt = "\n".join(user_parts) if user_parts else ""
        return system_prompt, user_prompt

    def _build_schema_instruction(
        self, model_request_parameters: ModelRequestParameters
    ) -> str:
        """Build instruction for structured output if needed."""
        if not model_request_parameters.output_tools:
            return ""

        # Get the output tool schema
        output_tool = model_request_parameters.output_tools[0]
        schema = output_tool.parameters_json_schema

        schema_json = json.dumps(schema, indent=2)
        return (
            f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Respond ONLY with the JSON object, no additional text."
        )

    def _extract_structured_output(self, content: str) -> dict | None:
        """Extract structured JSON output from agent response."""
        import re

        # Look for JSON code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse the entire content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Look for JSON object anywhere in the text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the Copilot model."""
        # Convert messages
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        # Add schema instruction if structured output requested
        if model_request_parameters.output_tools:
            user_prompt += self._build_schema_instruction(model_request_parameters)

        # Ensure session
        await self._ensure_session(system_prompt, model_settings)

        # Get timeout
        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        # Send to Copilot SDK and wait for response
        response = await self._session.send_and_wait(
            {"prompt": user_prompt}, timeout=timeout
        )

        if not response:
            raise RuntimeError("No response from Copilot agent")

        content = response.data.content

        # Build response parts
        parts: list[ModelResponsePart] = []

        # Check if we need to extract structured output
        if model_request_parameters.output_tools:
            structured_output = self._extract_structured_output(content)
            if structured_output:
                # Create a tool call for the structured output
                output_tool = model_request_parameters.output_tools[0]
                parts.append(
                    ToolCallPart(
                        tool_name=output_tool.name,
                        args=structured_output,
                    )
                )
            else:
                # Fallback to text if extraction failed
                parts.append(TextPart(content=content))
        else:
            # Regular text response
            parts.append(TextPart(content=content))

        # Create usage info (Copilot SDK doesn't provide token counts)
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
        """Make a streaming request to the Copilot model."""
        # Convert messages
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        # Add schema instruction if structured output requested
        if model_request_parameters.output_tools:
            user_prompt += self._build_schema_instruction(model_request_parameters)

        # Ensure session
        await self._ensure_session(system_prompt, model_settings)

        # Get timeout
        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        # Set up event handler for streaming
        event_queue = asyncio.Queue()

        def on_event(event):
            event_queue.put_nowait(event)

        unsubscribe = self._session.on(on_event)

        try:
            # Send message
            await self._session.send({"prompt": user_prompt})

            # Create streamed response
            async def stream_generator():
                full_content = ""
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=timeout
                        )

                        # Handle different event types
                        if event.type.value == "assistant.message":
                            chunk_content = event.data.content
                            full_content = chunk_content
                            # Yield text chunk
                            yield TextPart(content=chunk_content)

                        elif event.type.value == "session.idle":
                            # Session finished
                            break

                        elif event.type.value == "session.error":
                            raise RuntimeError(f"Session error: {event.data.message}")

                    except asyncio.TimeoutError:
                        raise RuntimeError(f"Timeout after {timeout}s")

                # Handle structured output at the end
                if model_request_parameters.output_tools and full_content:
                    structured_output = self._extract_structured_output(full_content)
                    if structured_output:
                        output_tool = model_request_parameters.output_tools[0]
                        yield ToolCallPart(
                            tool_name=output_tool.name,
                            args=structured_output,
                        )

            # Create usage info
            usage_info = RequestUsage()

            # Yield the streamed response
            yield StreamedResponse(
                stream=stream_generator(),
                usage=usage_info,
            )

        finally:
            unsubscribe()

    async def cleanup(self):
        """Clean up resources."""
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
        """Context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        """Context manager exit."""
        await self.cleanup()
