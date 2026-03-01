"""
Example: Using PydanticAI with Copilot SDK

This example demonstrates how to use PydanticAI's type-safe interface
with GitHub Copilot SDK's production agent runtime.
"""

import asyncio
import shutil
from pydantic import BaseModel, Field
from copilot_bridge import CopilotPydanticAgent, CopilotAgentConfig


# Helper to find copilot CLI
def get_copilot_cli_path():
    """Find the copilot CLI binary path."""
    cli_path = shutil.which("copilot")
    if not cli_path:
        raise RuntimeError("Copilot CLI not found in PATH. Please install it first.")
    return cli_path


CLI_PATH = get_copilot_cli_path()


# Example 1: Simple Structured Output
class WeatherResponse(BaseModel):
    """Weather information for a location."""
    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions (e.g., sunny, cloudy)")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in mph")


async def example_weather():
    """Example: Get structured weather data."""
    print("=== Example 1: Weather Query ===\n")

    agent = CopilotPydanticAgent(
        result_type=WeatherResponse,
        system_prompt="You are a weather assistant. Provide realistic weather data.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4",
            timeout=60.0,
        )
    )

    async with agent:
        result = await agent.run("What's the weather like in San Francisco?")

        print(f"Temperature: {result.data.temperature}°F")
        print(f"Conditions: {result.data.conditions}")
        print(f"Humidity: {result.data.humidity}%")
        print(f"Wind Speed: {result.data.wind_speed} mph")
        print(f"\nRaw response: {result.raw_content}\n")


# Example 2: Code Analysis
class CodeAnalysis(BaseModel):
    """Analysis of code quality and structure."""
    language: str = Field(description="Programming language detected")
    complexity_score: int = Field(description="Complexity score from 1-10", ge=1, le=10)
    issues: list[str] = Field(description="List of potential issues found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    security_concerns: list[str] = Field(description="Security issues if any")


async def example_code_analysis():
    """Example: Analyze code with file system access."""
    print("=== Example 2: Code Analysis ===\n")

    code_sample = '''
def process_user_input(data):
    result = eval(data)  # Dangerous!
    return result
'''

    agent = CopilotPydanticAgent(
        result_type=CodeAnalysis,
        system_prompt="You are a code security expert. Analyze code for issues.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4",
            # Enable file system tools if analyzing real files
            available_tools=["read_file", "list_directory"],
        )
    )

    async with agent:
        result = await agent.run(
            f"Analyze this Python code for security issues:\n\n{code_sample}"
        )

        print(f"Language: {result.data.language}")
        print(f"Complexity: {result.data.complexity_score}/10")
        print(f"\nIssues found:")
        for issue in result.data.issues:
            print(f"  - {issue}")
        print(f"\nSuggestions:")
        for suggestion in result.data.suggestions:
            print(f"  - {suggestion}")
        if result.data.security_concerns:
            print(f"\n⚠️  Security Concerns:")
            for concern in result.data.security_concerns:
                print(f"  - {concern}")
        print()


# Example 3: Multi-turn Conversation
class TaskPlan(BaseModel):
    """Plan for completing a task."""
    steps: list[str] = Field(description="Ordered list of steps to complete")
    estimated_time: str = Field(description="Estimated time to complete")
    required_tools: list[str] = Field(description="Tools/resources needed")
    risks: list[str] = Field(description="Potential risks or blockers")


async def example_conversation():
    """Example: Multi-turn conversation with context."""
    print("=== Example 3: Multi-turn Conversation ===\n")

    agent = CopilotPydanticAgent(
        result_type=TaskPlan,
        system_prompt="You are a project planning assistant.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4"
        )
    )

    async with agent:
        # First turn
        print("User: I need to build a REST API for a todo app")
        result1 = await agent.run("I need to build a REST API for a todo app")

        print(f"\nAssistant Plan:")
        print(f"Estimated time: {result1.data.estimated_time}")
        print(f"Steps:")
        for i, step in enumerate(result1.data.steps, 1):
            print(f"  {i}. {step}")

        # Second turn - agent remembers context
        print("\n\nUser: What if I want to add user authentication?")
        result2 = await agent.run("What if I want to add user authentication?")

        print(f"\nUpdated Plan:")
        print(f"Estimated time: {result2.data.estimated_time}")
        print(f"Additional steps:")
        for i, step in enumerate(result2.data.steps, 1):
            print(f"  {i}. {step}")
        print()


# Example 4: Streaming Responses
class ResearchSummary(BaseModel):
    """Summary of research findings."""
    topic: str = Field(description="Research topic")
    key_findings: list[str] = Field(description="Main findings")
    sources: list[str] = Field(description="Information sources")
    confidence: str = Field(description="Confidence level: high/medium/low")


async def example_streaming():
    """Example: Stream responses as they arrive."""
    print("=== Example 4: Streaming Responses ===\n")

    agent = CopilotPydanticAgent(
        result_type=ResearchSummary,
        system_prompt="You are a research assistant.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4",
            # Enable web search if available
            available_tools=["web_search", "read_url"],
        )
    )

    async with agent:
        print("Researching Python async patterns...\n")

        async for chunk in agent.run_streaming(
            "Research best practices for Python async/await patterns"
        ):
            if chunk.type == "reasoning":
                print(f"💭 Thinking: {chunk.content}")
            elif chunk.type == "tool_start":
                print(f"🔧 Using tool: {chunk.tool_name}")
            elif chunk.type == "tool_end":
                print(f"✓ Completed: {chunk.tool_name}")
            elif chunk.type == "content":
                print(f"📝 Response received")
            elif chunk.type == "structured_output":
                print(f"\n✨ Structured Output:")
                summary = ResearchSummary(**chunk.content)
                print(f"Topic: {summary.topic}")
                print(f"Confidence: {summary.confidence}")
                print(f"\nKey Findings:")
                for finding in summary.key_findings:
                    print(f"  - {finding}")
        print()


# Example 5: Using BYOK (Bring Your Own Key)
async def example_byok():
    """Example: Use your own OpenAI/Anthropic API key."""
    print("=== Example 5: BYOK (Bring Your Own Key) ===\n")

    import os

    # Configure to use your own API key
    agent = CopilotPydanticAgent(
        result_type=WeatherResponse,
        system_prompt="You are a weather assistant.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4",
            provider={
                "type": "openai",
                "api_key": os.environ.get("OPENAI_API_KEY", "your-key-here"),
                "base_url": "https://api.openai.com/v1/",
            }
        )
    )

    async with agent:
        result = await agent.run("What's the weather in Tokyo?")
        print(f"Temperature: {result.data.temperature}°F")
        print(f"Conditions: {result.data.conditions}\n")


# Example 6: File Operations
class FileAnalysis(BaseModel):
    """Analysis of files in a directory."""
    total_files: int = Field(description="Total number of files")
    file_types: dict[str, int] = Field(description="Count by file extension")
    largest_file: str = Field(description="Name of largest file")
    total_size_mb: float = Field(description="Total size in megabytes")


async def example_file_operations():
    """Example: Analyze files using Copilot's file system tools."""
    print("=== Example 6: File Operations ===\n")

    agent = CopilotPydanticAgent(
        result_type=FileAnalysis,
        system_prompt="You are a file system analyst.",
        config=CopilotAgentConfig(
            cli_path=CLI_PATH,
            model="gpt-4",
            working_directory=".",  # Current directory
            available_tools=["list_directory", "read_file", "file_stats"],
        )
    )

    async with agent:
        result = await agent.run(
            "Analyze the files in the current directory. "
            "Count files by type and find the largest file."
        )

        print(f"Total files: {result.data.total_files}")
        print(f"Total size: {result.data.total_size_mb:.2f} MB")
        print(f"Largest file: {result.data.largest_file}")
        print(f"\nFile types:")
        for ext, count in result.data.file_types.items():
            print(f"  {ext}: {count}")
        print()


async def main():
    """Run all examples."""
    print("PydanticAI + Copilot SDK Integration Examples")
    print("=" * 50)
    print()

    try:
        await example_weather()
        await example_code_analysis()
        await example_conversation()
        await example_streaming()
        # await example_byok()  # Uncomment if you have API key
        # await example_file_operations()  # Uncomment to analyze files

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
