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

If the user asks for the weather and the location is not clear,
you must first call get_user_location.

Keep replies short, helpful, and a bit funny.
"""


# ========= 2. Context schema =========
@dataclass
class Context:
    user_id: str


# ========= 3. Tools =========
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user location based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "San Francisco"


# ========= 4. Model: Gemma 3 (local Ollama) =========
# Make sure model name matches `ollama list`
model = ChatOllama(
    model="gemma3:latest",   # <-- or "gemma3:2b" depending on what you pulled
    temperature=0.2,
)


# ========= 5. Memory =========
checkpointer = InMemorySaver()


# ========= 6. Build the agent =========
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,
)


# ========= 7. Run the agent =========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "what is the weather outside?"}
            ]
        },
        config=config,
        context=Context(user_id="1"),
    )

    last_msg = response["messages"][-1]
    print("\nAssistant:", last_msg["content"])

