# PydanticAI + Copilot SDK Integration

This integration combines the best of both worlds:
- **PydanticAI**: Type-safe agents with validation and structured outputs
- **Copilot SDK**: Production-tested agent runtime with file ops, git, planning

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Application                                       │
│  - Define Pydantic models for structured outputs        │
│  - Use PydanticAI-style API                            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  CopilotPydanticAgent (Bridge Layer)                    │
│  - Converts PydanticAI messages → Copilot format       │
│  - Extracts structured outputs                          │
│  - Validates with Pydantic models                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Copilot SDK Session                                    │
│  - Agent orchestration                                  │
│  - Tool execution (file ops, git, web, etc.)           │
│  - Conversation management                              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  GitHub Copilot CLI                                     │
│  - LLM calls (GPT-4, Claude, etc.)                     │
│  - Planning and reasoning                               │
└─────────────────────────────────────────────────────────┘
```

## Key Benefits

### From PydanticAI:
✅ **Type Safety**: Runtime validation of inputs and outputs
✅ **Structured Outputs**: Guaranteed JSON schema compliance
✅ **Developer Experience**: Clean, Pythonic API
✅ **Validation**: Automatic data validation with Pydantic

### From Copilot SDK:
✅ **Production Agent Runtime**: Battle-tested orchestration
✅ **Rich Tool Set**: File system, git, web search, etc.
✅ **Planning**: Multi-step reasoning and execution
✅ **BYOK Support**: Use your own API keys
✅ **Session Management**: Conversation history and context

## Installation

```bash
# Install Copilot SDK
pip install github-copilot-sdk

# Install PydanticAI
pip install pydantic-ai

# Install Copilot CLI
# Follow: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli
```

## Quick Start

### Basic Usage

```python
from pydantic import BaseModel, Field
from pydantic_ai_bridge import CopilotPydanticAgent, CopilotAgentConfig

# Define your output schema
class WeatherResponse(BaseModel):
    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")

# Create agent
agent = CopilotPydanticAgent(
    result_type=WeatherResponse,
    system_prompt="You are a weather assistant",
    config=CopilotAgentConfig(model="gpt-4")
)

# Use it
async with agent:
    result = await agent.run("What's the weather in NYC?")
    print(f"Temperature: {result.data.temperature}°F")
    print(f"Conditions: {result.data.conditions}")
```

### With Streaming

```python
async with agent:
    async for chunk in agent.run_streaming("Analyze this code..."):
        if chunk.type == "reasoning":
            print(f"Thinking: {chunk.content}")
        elif chunk.type == "tool_start":
            print(f"Using tool: {chunk.tool_name}")
        elif chunk.type == "structured_output":
            validated_data = WeatherResponse(**chunk.content)
            print(f"Result: {validated_data}")
```

### Multi-turn Conversations

```python
agent = CopilotPydanticAgent(
    result_type=TaskPlan,
    system_prompt="You are a planning assistant"
)

async with agent:
    # First message
    result1 = await agent.run("Plan a REST API project")

    # Second message - agent remembers context
    result2 = await agent.run("Add authentication to the plan")
    # Agent updates the plan based on previous conversation
```

## Configuration Options

### CopilotAgentConfig

```python
config = CopilotAgentConfig(
    model="gpt-4",                    # Model to use
    cli_path="/path/to/copilot",      # Custom CLI path
    working_directory="/path/to/dir", # Working directory for file ops
    available_tools=["read_file"],    # Whitelist tools
    excluded_tools=["git_commit"],    # Blacklist tools
    timeout=300.0,                    # Timeout in seconds
    provider={                        # BYOK configuration
        "type": "openai",
        "api_key": "your-key",
        "base_url": "https://api.openai.com/v1/",
    }
)
```

### Available Tools

When `available_tools` is not specified, all Copilot CLI tools are enabled:

**File System:**
- `read_file` - Read file contents
- `write_file` - Write to files
- `list_directory` - List directory contents
- `file_stats` - Get file metadata
- `search_files` - Search for files

**Git Operations:**
- `git_status` - Check git status
- `git_diff` - View changes
- `git_commit` - Commit changes
- `git_log` - View history

**Web:**
- `web_search` - Search the web
- `read_url` - Fetch URL content

**Code:**
- `run_command` - Execute shell commands
- `code_search` - Search code patterns

## Advanced Examples

### Example 1: Code Analysis with File Access

```python
class CodeAnalysis(BaseModel):
    language: str
    complexity_score: int = Field(ge=1, le=10)
    issues: list[str]
    suggestions: list[str]

agent = CopilotPydanticAgent(
    result_type=CodeAnalysis,
    system_prompt="You are a code reviewer",
    config=CopilotAgentConfig(
        working_directory="./src",
        available_tools=["read_file", "list_directory", "code_search"]
    )
)

async with agent:
    result = await agent.run(
        "Analyze all Python files in the current directory for code quality"
    )
    print(f"Complexity: {result.data.complexity_score}/10")
    for issue in result.data.issues:
        print(f"Issue: {issue}")
```

### Example 2: Git Workflow Automation

```python
class GitAnalysis(BaseModel):
    uncommitted_files: list[str]
    branch: str
    last_commit: str
    suggested_commit_message: str

agent = CopilotPydanticAgent(
    result_type=GitAnalysis,
    system_prompt="You are a git assistant",
    config=CopilotAgentConfig(
        working_directory=".",
        available_tools=["git_status", "git_diff", "git_log"]
    )
)

async with agent:
    result = await agent.run(
        "Analyze the current git state and suggest a commit message"
    )
    print(f"Branch: {result.data.branch}")
    print(f"Suggested commit: {result.data.suggested_commit_message}")
```

### Example 3: Research with Web Access

```python
class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence: str = Field(pattern="^(high|medium|low)$")

agent = CopilotPydanticAgent(
    result_type=ResearchSummary,
    system_prompt="You are a research assistant",
    config=CopilotAgentConfig(
        available_tools=["web_search", "read_url"]
    )
)

async with agent:
    result = await agent.run(
        "Research the latest developments in Python async programming"
    )
    print(f"Confidence: {result.data.confidence}")
    for finding in result.data.key_findings:
        print(f"- {finding}")
```

## Comparison with Pure PydanticAI

| Feature | Pure PydanticAI | PydanticAI + Copilot SDK |
|---------|----------------|--------------------------|
| Type Safety | ✅ | ✅ |
| Structured Outputs | ✅ | ✅ |
| File Operations | ❌ | ✅ (via Copilot tools) |
| Git Operations | ❌ | ✅ (via Copilot tools) |
| Web Search | ❌ | ✅ (via Copilot tools) |
| Multi-step Planning | ⚠️ Manual | ✅ Automatic |
| Production Runtime | ❌ | ✅ (GitHub's engine) |
| BYOK | ✅ | ✅ |
| Conversation History | Manual | ✅ Automatic |

## Comparison with Pure Copilot SDK

| Feature | Pure Copilot SDK | PydanticAI + Copilot SDK |
|---------|-----------------|--------------------------|
| Type Safety | ❌ | ✅ |
| Structured Outputs | ⚠️ Manual parsing | ✅ Automatic validation |
| File Operations | ✅ | ✅ |
| Git Operations | ✅ | ✅ |
| Agent Orchestration | ✅ | ✅ |
| Python Type Hints | ❌ | ✅ |
| Runtime Validation | ❌ | ✅ |

## Error Handling

```python
from pydantic import ValidationError

try:
    async with agent:
        result = await agent.run("Your prompt")
except ValidationError as e:
    print(f"Output validation failed: {e}")
except RuntimeError as e:
    print(f"Agent execution failed: {e}")
except asyncio.TimeoutError:
    print("Agent timed out")
```

## Best Practices

### 1. Define Clear Schemas

```python
class GoodSchema(BaseModel):
    """Clear, well-documented schema."""
    temperature: float = Field(
        description="Temperature in Fahrenheit",
        ge=-100,
        le=150
    )
    conditions: str = Field(
        description="Weather conditions (sunny, cloudy, rainy, etc.)"
    )
```

### 2. Use Appropriate Timeouts

```python
# Short timeout for simple queries
config = CopilotAgentConfig(timeout=30.0)

# Longer timeout for complex operations
config = CopilotAgentConfig(timeout=300.0)
```

### 3. Limit Tool Access

```python
# Only enable necessary tools
config = CopilotAgentConfig(
    available_tools=["read_file", "list_directory"]
)

# Or exclude dangerous tools
config = CopilotAgentConfig(
    excluded_tools=["write_file", "run_command"]
)
```

### 4. Handle Streaming Appropriately

```python
async for chunk in agent.run_streaming(prompt):
    if chunk.type == "tool_start":
        # Show progress to user
        print(f"Working: {chunk.tool_name}...")
    elif chunk.type == "structured_output":
        # Process final result
        process_result(chunk.content)
```

## Limitations

1. **Structured Output Extraction**: Relies on LLM following JSON format instructions. May fail with complex schemas or less capable models.

2. **Conversation Context**: Copilot SDK manages conversation internally. You can't directly manipulate message history.

3. **Tool Execution**: Tools run via Copilot CLI. Custom PydanticAI tools need to be implemented as Copilot CLI tools.

4. **Streaming Granularity**: Streaming is at the event level, not token level.

## Troubleshooting

### "No response from Copilot agent"

- Check Copilot CLI is installed and in PATH
- Verify GitHub authentication: `copilot auth status`
- Increase timeout in config

### "Failed to extract structured output"

- Simplify your Pydantic schema
- Add more explicit field descriptions
- Use a more capable model (gpt-4 vs gpt-3.5)
- Check raw_content in result for debugging

### "Tool not available"

- Check tool name spelling
- Verify tool is in `available_tools` list
- Some tools require specific permissions

## Contributing

Contributions welcome! Areas for improvement:

- Better structured output extraction
- Support for PydanticAI's dependency injection
- Custom tool integration
- Improved streaming granularity
- Error recovery strategies

## License

MIT - Same as Copilot SDK and PydanticAI
