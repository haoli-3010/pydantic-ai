"""
Quick test of the Copilot SDK bridge with PydanticAI
"""

import asyncio
from pydantic import BaseModel, Field
import sys
sys.path.insert(0, 'examples')

from pydantic_ai_examples.copilot_sdk.copilot_bridge import (
    CopilotPydanticAgent,
    CopilotAgentConfig,
)


class SimpleResponse(BaseModel):
    """A simple response for testing."""
    message: str = Field(description="A simple message")
    number: int = Field(description="A number between 1 and 10", ge=1, le=10)


async def test_basic():
    """Test basic agent functionality."""
    print("Testing Copilot SDK Bridge with PydanticAI...")
    print("=" * 60)
    
    config = CopilotAgentConfig(
        model="gpt-4",
        timeout=30.0,
        cli_path="/home/hfeng1/.nvm/versions/node/v24.8.0/bin/copilot",
    )
    
    agent = CopilotPydanticAgent(
        result_type=SimpleResponse,
        system_prompt="You are a helpful assistant. Always respond with valid JSON.",
        config=config,
    )
    
    try:
        print("\n🔧 Starting Copilot client...")
        async with agent:
            print("✓ Client started")
            print("\n📤 Sending: Tell me hello and give me the number 7")
            result = await agent.run("Tell me hello and give me the number 7")
            
            print(f"\n✅ Success!")
            print(f"Message: {result.data.message}")
            print(f"Number: {result.data.number}")
            print(f"\nRaw response: {result.raw_content[:200]}...")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_basic())
