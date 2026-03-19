"""
Simple MCP Agent with vLLM Backend.
Please refer to https://github.com/Snowflake-Labs/agent-world-model for more details.
"""
import asyncio
import contextlib
import io
import json
import logging
import re
import os
import time
from dataclasses import dataclass
from textwrap import dedent

from loguru import logger
from openai import AsyncOpenAI

from awm.tools import tools_robust_json_loads, isolated_mcp_env

for _name in ["mcp_agent", "mcp", "httpx", "httpcore", "anyio"]:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
    logging.getLogger(_name).propagate = False
    logging.getLogger(_name).handlers = []

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings

logger.disable("mcp_agent")
logger.disable("mcp")


@dataclass
class Config:
    # Task description for the agent
    task: str
    # MCP server URL (streamable HTTP transport)
    mcp_url: str
    # vLLM OpenAI-compatible API URL
    vllm_url: str = "http://localhost:8001/v1"
    # Model name (served_model_name in vLLM)
    model: str = "Snowflake/Arctic-AWM-4B"
    # Max agent loop iterations
    max_iterations: int = 30
    # Generation temperature
    temperature: float = 0.6
    # Max completion tokens
    max_tokens: int = 2048
    # Timeout per LLM request in seconds
    request_timeout: float = 60.0
    # Whether to enable model thinking mode when supported by the backend
    enable_thinking: bool = True
    # Verbose logging
    verbose: bool = True


def get_system_prompt() -> str:
    tools_str = dedent("""\
        1. list_tools
            - Description: List all available MCP tools for the current environment to help you finish the user task.
            - Arguments: None
            - Output: A list of MCP environment-specific tools and their descriptions

        2. call_tool
            - Description: Call a MCP environment-specific tool
            - Arguments:
                - tool_name: str, required, the tool name in the list_tools output
                - arguments: str, required, the arguments for calling <tool_name>. You must pass a valid JSON string without any markdown fences or additional commentary. This JSON str will be parsed by the tool and executed. You can pass an empty JSON str if no arguments are required by <tool_name>.
            - Output: The result of the <tool_name> tool call""")

    return dedent(f"""\
        # MCP Toolss
        
        You are at a MCP environment. You need to call MCP tools to assist with the user query. At each step, you can only call one function. You have already logged in, and your user id is 1 if required for the MCP tool.

        You are provided with TWO functions within <tools></tools> XML tags:
        <tools>
        {tools_str}
        </tools>

        You should always call list_tools function first to get the available tools, and should only call it once. You should always directly output the answer or summary at the final step instead of calling any function. 

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {{"name": <function-name>, "arguments": <args-json-object>}}
        </tool_call>
        
        Example Function Call #1:
        <tool_call>
        {{"name": "list_tools", "arguments": null}}
        </tool_call>
        
        Example Function Call #2:
        <tool_call>
        {{"name": "call_tool", "arguments": {{"tool_name": "get_weather", "arguments": "{{"city": "Beijing"}}"}}}}
        </tool_call>""")


def parse_tool_calls(content: str) -> list[dict]:
    tool_calls = []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        data = tools_robust_json_loads(match.strip())
        if not data:
            logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
            continue

        # Handle list wrapping: [{"name": ..., "arguments": ...}]
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                continue

        if not isinstance(data, dict):
            continue

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        if name.startswith("mcp_tool_"):
            arguments = {
                "tool_name": name,
                "arguments": arguments if arguments else {},
            }
            name = "call_tool"

        tool_calls.append({
            "id": f"call_{int(time.time() * 1000)}_{i}",
            "name": name,
            "arguments": arguments,
        })

    return tool_calls


def parse_call_tool_arguments(arguments: dict | str | None) -> tuple[str, dict]:
    if isinstance(arguments, str):
        arguments = tools_robust_json_loads(arguments)
    if not isinstance(arguments, dict):
        return "", {}

    tool_name = arguments.get("tool_name", "")
    inner_args = arguments.get("arguments", {})

    if tool_name.startswith("mcp_tool_"):
        tool_name = tool_name[len("mcp_tool_"):]

    if isinstance(inner_args, str):
        parsed = tools_robust_json_loads(inner_args) if inner_args.strip() else {}
        if isinstance(parsed, dict):
            inner_args = parsed
        else:
            inner_args = {}

    if not isinstance(inner_args, dict):
        inner_args = {}

    return tool_name, inner_args


def format_tools_for_response(tools: list[dict]) -> str:

    def format_input_schema(schema: dict, indent_level: int = 6, parent_required: list | None = None) -> str:
        if not schema:
            return ""

        result = ""
        indent_str = " " * indent_level
        properties = schema.get('properties', {})
        required_fields = parent_required if parent_required is not None else schema.get('required', [])

        for prop_name, prop_info in properties.items():
            is_required = prop_name in required_fields
            required_str = " (required)" if is_required else " (optional)"
            prop_type = prop_info.get('type', 'unknown')
            description = prop_info.get('description', '')
            default = prop_info.get('default')
            enum_values = prop_info.get('enum')
            nested_properties = prop_info.get('properties', {})
            nested_required = prop_info.get('required', [])

            result += f"{indent_str}- {prop_name}: {prop_type}{required_str}\n"
            if description:
                result += f"{indent_str}  Description: {description}\n"
            if default is not None:
                result += f"{indent_str}  Default: {default}\n"
            if enum_values:
                result += f"{indent_str}  Allowed values: {enum_values}\n"

            if prop_type == "object" and nested_properties:
                result += f"{indent_str}  Properties:\n"
                nested_schema = {"properties": nested_properties, "required": nested_required}
                result += format_input_schema(nested_schema, indent_level + 4, nested_required)

        return result

    # filter out list_tools meta-tool
    actual_tools = [t for t in tools if t.get('name') != 'list_tools']

    docs_text = f"Available MCP Tools ({len(actual_tools)} tools):\n"
    docs_text += "=" * 80 + "\n\n"

    for i, tool in enumerate(actual_tools, 1):
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", tool.get("input_schema", {}))

        # add mcp_tool_ prefix
        mcp_name = f"mcp_tool_{name}" if not name.startswith("mcp_tool_") else name

        # parse description for multi-line
        desc_lines = description.split('\n')
        first_line = desc_lines[0].strip() if desc_lines else "No description"
        more_desc = '\n'.join(line.strip() for line in desc_lines[1:]).strip() if len(desc_lines) > 1 else ""

        docs_text += f"{i}. {mcp_name}\n"
        docs_text += f"   Description: {first_line}\n"
        if more_desc:
            for line in more_desc.split('\n'):
                if line.strip():
                    docs_text += f"   {line}\n"

        if input_schema and input_schema.get('properties'):
            docs_text += f"   Parameters:\n"
            docs_text += format_input_schema(input_schema)
        else:
            docs_text += f"   Parameters: None\n"

        docs_text += "\n"

    return docs_text.strip()


MCP_SERVER_NAME = "mcp_server"


class MCPToolExecutor:
    """MCP tool executor with persistent connection support.

    Use as async context manager to keep connection alive across calls::

        async with MCPToolExecutor(url) as mcp:
            tools = await mcp.list_tools()
            result = await mcp.call_tool("foo", {"bar": 1})

    Without ``async with`` it falls back to reconnecting per call.
    """

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict] = []
        self._connected = False
        self._app_ctx = None
        self._stderr_redirect = None

        with isolated_mcp_env():
            settings = Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(
                    type="none",
                    transports=["none"],
                    progress_display=False,
                    level="error",
                ),
                mcp=MCPSettings(
                    servers={
                        MCP_SERVER_NAME: MCPServerSettings(
                            transport='streamable_http',
                            url=self.mcp_url,
                        ),
                    }
                ),
            )
            self._app = MCPApp(name="demo_agent", settings=settings)
            self._agent = Agent(name="executor", server_names=[MCP_SERVER_NAME])

    async def __aenter__(self):
        self._stderr_redirect = contextlib.redirect_stderr(io.StringIO())
        self._stderr_redirect.__enter__()
        self._app_ctx = self._app.run()
        await self._app_ctx.__aenter__()
        await self._agent.__aenter__()
        self._connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._connected = False
        try:
            await self._agent.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass
        try:
            await self._app_ctx.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass
        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(None, None, None)
            self._stderr_redirect = None

    async def list_tools(self) -> list[dict]:
        if self._connected:
            return await self._list_tools_inner()
        with contextlib.redirect_stderr(io.StringIO()):
            async with self._app.run():
                async with self._agent:
                    return await self._list_tools_inner()

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self._connected:
            return await self._call_tool_inner(tool_name, arguments)
        with contextlib.redirect_stderr(io.StringIO()):
            async with self._app.run():
                async with self._agent:
                    return await self._call_tool_inner(tool_name, arguments)

    async def _list_tools_inner(self) -> list[dict]:
        result = await asyncio.wait_for(
            self._agent.list_tools(), timeout=self.timeout
        )
        self._tools = []
        for t in result.tools:
            self._tools.append({
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            })
        return self._tools

    async def _call_tool_inner(self, tool_name: str, arguments: dict) -> str:
        result = await asyncio.wait_for(
            self._agent.call_tool(tool_name, arguments),
            timeout=self.timeout,
        )
        parts = []
        for c in result.content:
            if hasattr(c, 'text'):
                parts.append(c.text)
            else:
                parts.append(str(c))
        text = "\n".join(parts)
        if result.isError:
            return f"Error: {text}"
        return text

async def generate_response(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    config: Config,
) -> tuple[str, list[dict]]:
    api_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role in ("system", "user"):
            api_messages.append({"role": role, "content": content})
        elif role == "assistant":
            api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            api_messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", "call_unknown"),
                "content": content,
            })

    extra_body = {
        "add_generation_prompt": True,
        "min_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": bool(getattr(config, "enable_thinking", True))},
        "top_k": getattr(config, "top_k", 20),
        "min_p": getattr(config, "min_p", 0.0),
    }

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model_name,
            messages=api_messages,
            max_completion_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=getattr(config, "top_p", 0.95),
            extra_body=extra_body,
        ),
        timeout=max(1.0, float(getattr(config, "request_timeout", 60.0))),
    )

    content = response.choices[0].message.content or ""
    tool_calls = parse_tool_calls(content)

    return content, tool_calls


async def run_agent(config: Config):
    logger.info("=" * 80)
    logger.info("MCP Agent")
    logger.info("=" * 80)
    logger.info(f"Task: {config.task}")
    logger.info(f"MCP Server: {config.mcp_url}")
    logger.info(f"vLLM Server: {config.vllm_url}")
    logger.info(f"Model: {config.model}")
    logger.info("=" * 80)

    mcp = MCPToolExecutor(config.mcp_url)
    vllm_client = AsyncOpenAI(
        api_key=os.environ.get('OPENAI_API_KEY') or "EMPTY",
        base_url=config.vllm_url,
        timeout=config.request_timeout,
    )
    try:
        logger.info("Connecting to MCP server and fetching tools...")
        tools = await mcp.list_tools()
        logger.info(f"Loaded {len(tools)} tools from MCP server")
        tools_response_text = format_tools_for_response(tools)

        messages: list[dict] = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": config.task},
        ]

        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{config.max_iterations} ---")

            try:
                content, tool_calls = await generate_response(
                    vllm_client, config.model, messages, config,
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM request timed out after {config.request_timeout}s")
                break

            if config.verbose:
                preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"Assistant ({len(content)} chars): {preview}")
                logger.info(f"Tool calls: {len(tool_calls)}")

            messages.append({"role": "assistant", "content": content})

            # no tool calls -> task complete
            if not tool_calls:
                logger.info("No tool calls detected - task complete.")
                logger.info(f"\n{'=' * 40} Final Answer {'=' * 40}")
                logger.info(content)
                logger.info("=" * 80)
                break

            # process first tool call only (one per turn)
            tc = tool_calls[0]
            name = tc["name"]
            arguments = tc["arguments"]
            tool_call_id = tc["id"]

            if len(tool_calls) > 1:
                skipped = [t["name"] for t in tool_calls[1:]]
                logger.warning(f"Multiple tool calls detected. Only executing first: {name}. Skipped: {skipped}")

            # execute tool call
            if name == "list_tools":
                logger.info("Executing: list_tools")
                response_text = tools_response_text

            elif name == "call_tool":
                tool_name, tool_args = parse_call_tool_arguments(arguments)
                logger.info(f"Executing: call_tool({tool_name}, {json.dumps(tool_args, ensure_ascii=False)[:200]})")
                try:
                    response_text = await mcp.call_tool(tool_name, tool_args)
                except asyncio.TimeoutError:
                    response_text = f"Error: Tool call timed out after {mcp.timeout}s"
                    logger.error(response_text)
                except Exception as e:
                    response_text = f"Error: {e}"
                    logger.error(f"Tool call failed: {e}")
            else:
                response_text = f"Error: Unknown tool '{name}'. Only 'list_tools' and 'call_tool' are available."

            if config.verbose:
                preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logger.info(f"Tool response ({len(response_text)} chars): {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": response_text,
            })
        else:
            logger.warning("Max iterations reached without completion.")

        logger.info("=" * 80)
        logger.info(f"Agent execution complete. Total iterations: {iteration}")
        logger.info("=" * 80)
    finally:
        close = getattr(vllm_client, "close", None)
        if close is not None:
            await close()


def run(config: Config):
    asyncio.run(run_agent(config))


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)
