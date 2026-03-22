from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama


# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are an expert weather forecaster who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get weather for a location
- get_user_location: use this to get the user's location

If the user asks for weather and location isn't clear, call get_user_location."""

# Runtime context for memory
@dataclass
class Context:
    user_id: str


# --- Tools ---
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Get user location based on user_id."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# --- MODEL: OLLAMA (LOCAL) ---
model = ChatOllama(
    model="gemma3",     # or llama3.2 / qwen2.5 / mistral
    temperature=0.2,
    max_tokens=512
)


# --- Structured Response Schema ---
@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None


# --- Memory ---
checkpointer = InMemorySaver()


# --- Agent ---
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# --- Run agent ---
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])

