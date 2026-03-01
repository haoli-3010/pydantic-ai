"""
PydanticAI + Copilot SDK Bridge

This module provides integration between PydanticAI's type-safe agent interface
and GitHub Copilot SDK's production agent runtime.

Architecture:
    PydanticAI Agent (validation) → CopilotAgentBridge → Copilot SDK Session (execution)

Example:
    >>> from pydantic import BaseModel
    >>> from pydantic_ai import Agent
    >>>
    >>> class WeatherResponse(BaseModel):
    ...     temperature: float
    ...     conditions: str
    >>>
    >>> agent = Agent(
    ...     CopilotAgentBridge(),
    ...     result_type=WeatherResponse,
    ...     system_prompt="You are a weather assistant"
    ... )
    >>>
    >>> result = await agent.run("What's the weather in NYC?")
    >>> print(f"Temperature: {result.data.temperature}°F")
"""

import asyncio
import json
from typing import Any, Optional, AsyncIterator, Literal
from dataclasses import dataclass

from pydantic import BaseModel
from copilot import CopilotClient, PermissionHandler
from copilot.types import SessionConfig, MessageOptions


@dataclass
class CopilotAgentConfig:
    """Configuration for Copilot agent bridge."""

    model: Optional[str] = None
    """Model to use (e.g., 'gpt-4', 'claude-3-5-sonnet')"""

    provider: Optional[dict] = None
    """Provider configuration for BYOK (Bring Your Own Key)"""

    cli_path: Optional[str] = None
    """Path to copilot CLI binary"""

    working_directory: Optional[str] = None
    """Working directory for file operations"""

    available_tools: Optional[list[str]] = None
    """List of copilot CLI tools to enable"""

    excluded_tools: Optional[list[str]] = None
    """List of copilot CLI tools to disable"""

    timeout: float = 300.0
    """Timeout in seconds for agent responses"""


class CopilotAgentBridge:
    """
    Bridge between PydanticAI and Copilot SDK.

    This class allows PydanticAI agents to use GitHub Copilot's production
    agent runtime while maintaining PydanticAI's type safety and validation.

    The bridge handles:
    - Converting PydanticAI messages to Copilot SDK format
    - Managing Copilot SDK sessions
    - Streaming responses back to PydanticAI
    - Tool execution coordination
    - Structured output extraction

    Example:
        >>> bridge = CopilotAgentBridge(CopilotAgentConfig(
        ...     model="gpt-4",
        ...     working_directory="/path/to/project"
        ... ))
        >>>
        >>> agent = Agent(bridge, result_type=MySchema)
        >>> result = await agent.run("Analyze this codebase")
    """

    def __init__(self, config: Optional[CopilotAgentConfig] = None):
        """
        Initialize the Copilot agent bridge.

        Args:
            config: Configuration for the Copilot agent. If None, uses defaults.
        """
        self.config = config or CopilotAgentConfig()
        self._client: Optional[CopilotClient] = None
        self._session: Optional[Any] = None  # CopilotSession
        self._session_lock = asyncio.Lock()

    async def __aenter__(self):
        """Context manager entry - starts Copilot client."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        """Context manager exit - cleans up resources."""
        await self.cleanup()

    async def _ensure_client(self):
        """Ensure Copilot client is started."""
        if self._client is None:
            client_opts = {}
            if self.config.cli_path:
                client_opts["cli_path"] = self.config.cli_path

            self._client = CopilotClient(client_opts)
            await self._client.start()

    async def _ensure_session(self, system_prompt: Optional[str] = None):
        """Ensure a Copilot session exists."""
        async with self._session_lock:
            if self._session is None:
                await self._ensure_client()

                session_config: SessionConfig = {
                    "on_permission_request": PermissionHandler.approve_all,
                }

                # Add model if specified
                if self.config.model:
                    session_config["model"] = self.config.model

                # Add provider if specified (BYOK)
                if self.config.provider:
                    session_config["provider"] = self.config.provider

                # Add system message if provided
                if system_prompt:
                    session_config["system_message"] = system_prompt

                # Add working directory if specified
                if self.config.working_directory:
                    session_config["working_directory"] = self.config.working_directory

                # Add tool filtering
                if self.config.available_tools is not None:
                    session_config["available_tools"] = self.config.available_tools
                if self.config.excluded_tools:
                    session_config["excluded_tools"] = self.config.excluded_tools

                self._session = await self._client.create_session(session_config)

    async def run_agent(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        result_schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Run the agent with PydanticAI-style messages.

        This is the main integration point. It:
        1. Converts PydanticAI messages to Copilot format
        2. Executes via Copilot SDK
        3. Extracts structured output if schema provided
        4. Returns in PydanticAI-compatible format

        Args:
            messages: List of messages in PydanticAI format
            system_prompt: System prompt for the agent
            result_schema: JSON schema for structured output

        Returns:
            Response dict with 'content' and optional 'structured_output'
        """
        await self._ensure_session(system_prompt)

        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)

        # Add schema instruction if structured output requested
        if result_schema:
            schema_instruction = self._build_schema_instruction(result_schema)
            prompt = f"{prompt}\n\n{schema_instruction}"

        # Send to Copilot SDK and wait for response
        response = await self._session.send_and_wait(
            {"prompt": prompt},
            timeout=self.config.timeout
        )

        if not response:
            raise RuntimeError("No response from Copilot agent")

        content = response.data.content

        # Extract structured output if schema provided
        structured_output = None
        if result_schema:
            structured_output = self._extract_structured_output(content, result_schema)

        return {
            "content": content,
            "structured_output": structured_output,
        }

    async def run_agent_streaming(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        result_schema: Optional[dict] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run the agent with streaming responses.

        Yields chunks as they arrive from Copilot SDK.

        Args:
            messages: List of messages in PydanticAI format
            system_prompt: System prompt for the agent
            result_schema: JSON schema for structured output

        Yields:
            Response chunks with 'type' and 'content'
        """
        await self._ensure_session(system_prompt)

        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)

        # Add schema instruction if structured output requested
        if result_schema:
            schema_instruction = self._build_schema_instruction(result_schema)
            prompt = f"{prompt}\n\n{schema_instruction}"

        # Set up event handler for streaming
        chunks = []
        event_queue = asyncio.Queue()

        def on_event(event):
            event_queue.put_nowait(event)

        unsubscribe = self._session.on(on_event)

        try:
            # Send message
            await self._session.send({"prompt": prompt})

            # Stream events
            full_content = ""
            while True:
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=self.config.timeout
                    )

                    # Handle different event types
                    if event.type.value == "assistant.message":
                        chunk_content = event.data.content
                        full_content = chunk_content
                        yield {
                            "type": "content",
                            "content": chunk_content,
                        }

                    elif event.type.value == "assistant.reasoning":
                        yield {
                            "type": "reasoning",
                            "content": event.data.content,
                        }

                    elif event.type.value == "tool.execution_start":
                        yield {
                            "type": "tool_start",
                            "tool_name": event.data.tool_name,
                        }

                    elif event.type.value == "tool.execution_end":
                        yield {
                            "type": "tool_end",
                            "tool_name": event.data.tool_name,
                        }

                    elif event.type.value == "session.idle":
                        # Session finished
                        break

                    elif event.type.value == "session.error":
                        raise RuntimeError(f"Session error: {event.data.message}")

                except asyncio.TimeoutError:
                    raise RuntimeError(f"Timeout after {self.config.timeout}s")

            # Extract structured output if requested
            if result_schema and full_content:
                structured_output = self._extract_structured_output(
                    full_content,
                    result_schema
                )
                if structured_output:
                    yield {
                        "type": "structured_output",
                        "content": structured_output,
                    }

        finally:
            unsubscribe()

    def _convert_messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """
        Convert PydanticAI message format to Copilot SDK prompt.

        PydanticAI uses OpenAI-style messages with roles (system, user, assistant).
        Copilot SDK manages conversation history internally, so we extract
        the latest user message as the prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Prompt string for Copilot SDK
        """
        # Find the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multi-part content (text + images)
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if part.get("type") == "text"
                    ]
                    return "\n".join(text_parts)

        return ""

    def _build_schema_instruction(self, schema: dict) -> str:
        """
        Build instruction for structured output.

        Args:
            schema: JSON schema for the expected output

        Returns:
            Instruction string to append to prompt
        """
        schema_json = json.dumps(schema, indent=2)
        return (
            f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Respond ONLY with the JSON object, no additional text."
        )

    def _extract_structured_output(
        self,
        content: str,
        schema: dict
    ) -> Optional[dict]:
        """
        Extract structured JSON output from agent response.

        Args:
            content: Raw response content from agent
            schema: Expected JSON schema

        Returns:
            Parsed JSON dict, or None if extraction fails
        """
        # Try to find JSON in the response
        import re

        # Look for JSON code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
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
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

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


# PydanticAI Integration Wrapper
class CopilotPydanticAgent:
    """
    High-level wrapper that provides PydanticAI-like interface using Copilot SDK.

    This class mimics PydanticAI's Agent API but uses Copilot SDK for execution.

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class CodeAnalysis(BaseModel):
        ...     language: str
        ...     complexity: int
        ...     suggestions: list[str]
        >>>
        >>> agent = CopilotPydanticAgent(
        ...     result_type=CodeAnalysis,
        ...     system_prompt="You are a code analysis expert",
        ...     config=CopilotAgentConfig(model="gpt-4")
        ... )
        >>>
        >>> async with agent:
        ...     result = await agent.run("Analyze this Python code: ...")
        ...     print(result.data.complexity)
    """

    def __init__(
        self,
        result_type: type[BaseModel],
        system_prompt: Optional[str] = None,
        config: Optional[CopilotAgentConfig] = None,
    ):
        """
        Initialize a Copilot-backed PydanticAI-style agent.

        Args:
            result_type: Pydantic model class for structured output
            system_prompt: System prompt for the agent
            config: Copilot agent configuration
        """
        self.result_type = result_type
        self.system_prompt = system_prompt
        self.bridge = CopilotAgentBridge(config)
        self._conversation_history: list[dict] = []

    async def __aenter__(self):
        """Context manager entry."""
        await self.bridge.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Context manager exit."""
        await self.bridge.__aexit__(*args)

    async def run(self, user_prompt: str) -> "AgentResult":
        """
        Run the agent with a user prompt.

        Args:
            user_prompt: The user's input message

        Returns:
            AgentResult with validated data
        """
        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Get JSON schema from Pydantic model
        schema = self.result_type.model_json_schema()

        # Run agent
        response = await self.bridge.run_agent(
            messages=self._conversation_history,
            system_prompt=self.system_prompt,
            result_schema=schema,
        )

        # Add assistant response to history
        self._conversation_history.append({
            "role": "assistant",
            "content": response["content"],
        })

        # Validate and parse structured output
        if response["structured_output"]:
            validated_data = self.result_type(**response["structured_output"])
        else:
            # Fallback: try to parse from content
            try:
                validated_data = self.result_type.model_validate_json(
                    response["content"]
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to extract structured output: {e}\n"
                    f"Response: {response['content']}"
                )

        return AgentResult(
            data=validated_data,
            raw_content=response["content"],
        )

    async def run_streaming(self, user_prompt: str) -> AsyncIterator["StreamChunk"]:
        """
        Run the agent with streaming responses.

        Args:
            user_prompt: The user's input message

        Yields:
            StreamChunk objects with incremental updates
        """
        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Get JSON schema from Pydantic model
        schema = self.result_type.model_json_schema()

        # Stream agent responses
        full_content = ""
        async for chunk in self.bridge.run_agent_streaming(
            messages=self._conversation_history,
            system_prompt=self.system_prompt,
            result_schema=schema,
        ):
            if chunk["type"] == "content":
                full_content = chunk["content"]

            yield StreamChunk(
                type=chunk["type"],
                content=chunk.get("content"),
                tool_name=chunk.get("tool_name"),
            )

        # Add final response to history
        if full_content:
            self._conversation_history.append({
                "role": "assistant",
                "content": full_content,
            })


@dataclass
class AgentResult:
    """Result from agent execution."""
    data: BaseModel
    """Validated Pydantic model instance"""

    raw_content: str
    """Raw text response from agent"""


@dataclass
class StreamChunk:
    """Streaming chunk from agent."""
    type: Literal["content", "reasoning", "tool_start", "tool_end", "structured_output"]
    """Type of chunk"""

    content: Optional[Any] = None
    """Chunk content (varies by type)"""

    tool_name: Optional[str] = None
    """Tool name (for tool_start/tool_end types)"""
