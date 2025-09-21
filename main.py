import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()

# Get Groq API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in .env file")

# Set up AsyncOpenAI external client with Groq base URL
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize the OpenAIChatCompletionsModel with the configure external client
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="llama-3.3-70b-versatile"
)

# Configure the execution environment for the agent
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define the agent
agent = Agent(
    name="Student Agent",
    instructions="You are a helpful student assistant. You can answer academic questions, give study tips, and summarize short text."
)

# Sends a welcome message to the user
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])
    await cl.Message("Welcome! I'm your student assistant.\n\n"
        "Please answer the following prompts one by one:\n"
        "Ask an academic question.\n"
        "Ask for a study tip (mention topic).\n"
        "Provide a short passage to summarize.\n\n"
        "Go ahead! Start by typing your academic question below.".send()

# Handle incoming messages in Chainlit
@cl.on_message
async def handle_message(message: cl.Message):
    try:
      chat_history = cl.user_session.get("chat_history", []) 
      chat_history.append({"role": "user", "content": message.content})

      result = await Runner.run(
            agent,
            input=chat_history,
            run_config=config     
        )
      
      chat_history.append({"role": "assistant", "content": result.final_output})
      cl.user_session.set("chat_history", chat_history)
      
      await cl.Message(content=result.final_output).send() 
      
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
