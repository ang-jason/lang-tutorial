from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama


# ========= 1. System prompt =========
SYSTEM_PROMPT = """
You are an expert weather forecaster who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific city
- get_user_location: use this to get the user's location

If the user asks you for the weather and the location is not clear, first call get_user_location.
Keep your answers short, friendly, and a bit funny.
"""


# ========= 2. Context schema (for runtime) =========
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# ========= 3. Tools =========
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    # This is just a fake example.
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user location based on user ID."""
    user_id = runtime.context.user_id
    # Simple demo logic
    return "Florida" if user_id == "1" else "San Francisco"


# ========= 4. Model: Ollama (local) =========
# IMPORTANT: model name must match `ollama list`
# e.g. "phi3:mini" or "llama3.2:1b"
model = ChatOllama(
    model="phi3:mini",   # <- change to "llama3.2:1b" if you pulled that instead
    temperature=0.2,
)


# ========= 5. Memory (LangGraph) =========
checkpointer = InMemorySaver()


# ========= 6. Create the agent =========
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,
)


# ========= 7. Run the agent =========
if __name__ == "__main__":
    # thread_id keeps conversation memory between calls
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what is the weather outside?",
                }
            ]
        },
        config=config,
        context=Context(user_id="1"),
    )

    # LangGraph agents return a dict with "messages"
    last_message = response["messages"][-1]
    print("Assistant:", last_message["content"])

