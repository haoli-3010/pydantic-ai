"""
Test CopilotModel as a drop-in replacement for PydanticAI models.

This demonstrates using CopilotModel with the standard PydanticAI Agent API.
"""

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from copilot_model import CopilotModel


class WeatherResponse(BaseModel):
    """Weather information for a location."""

    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage", ge=0, le=100)


async def test_drop_in_replacement():
    """Test using CopilotModel as a drop-in replacement."""
    print("Testing CopilotModel as drop-in replacement for PydanticAI")
    print("=" * 60)

    # Create the model - this is the only Copilot-specific line!
    model = CopilotModel(model_name_value="gpt-4")

    # Everything else is standard PydanticAI
    agent = Agent(
        model,
        output_type=WeatherResponse,
        system_prompt="You are a weather assistant. Provide realistic weather data.",
    )

    print("\n📤 Asking: What's the weather in Seattle?")

    async with model:
        result = await agent.run("What's the weather in Seattle?")

        print(f"\n✅ Success!")
        print(f"Temperature: {result.output.temperature}°F")
        print(f"Conditions: {result.output.conditions}")
        print(f"Humidity: {result.output.humidity}%")


async def test_with_tools():
    """Test CopilotModel with PydanticAI tools."""
    print("\n\nTesting CopilotModel with PydanticAI tools")
    print("=" * 60)

    model = CopilotModel(model_name_value="gpt-4")

    agent = Agent(
        model,
        system_prompt="You are a helpful calculator assistant.",
    )

    @agent.tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @agent.tool
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    print("\n📤 Asking: What is (5 + 3) * 2?")

    async with model:
        result = await agent.run("What is (5 + 3) * 2?")
        print(f"\n✅ Result: {result.output}")


async def test_streaming():
    """Test streaming with CopilotModel."""
    print("\n\nTesting CopilotModel with streaming")
    print("=" * 60)

    model = CopilotModel(model_name_value="gpt-4")

    agent = Agent(
        model,
        system_prompt="You are a helpful assistant.",
    )

    print("\n📤 Asking: Tell me a short story about a robot")

    async with model:
        async with agent.run_stream("Tell me a short story about a robot") as response:
            print("\n📝 Streaming response:")
            async for chunk in response.stream_text():
                print(chunk, end="", flush=True)
            print("\n")


async def main():
    """Run all tests."""
    try:
        await test_drop_in_replacement()
        # await test_with_tools()  # Uncomment when tool support is added
        # await test_streaming()  # Uncomment to test streaming

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
