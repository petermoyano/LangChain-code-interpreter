from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, create_csv_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool, Tool

load_dotenv()


def main():
    print("Start...")
    # Combines a LLM and a tool to create a python agent
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4",),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Comines a LLM and a tool to create a csv agent
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4",),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # csv_agent.run("How many columns does the csv file have?")
    # csv_agent.run(
    #     "Which writter wrote the most episodes? How many did he/she write?")
    grand_agent = initialize_agent(tools=[
        Tool(
            name="PythonAgent",
            func=python_agent_executor.run,
            description="""
                 Useful when you need to transform natural language into python code and execute it returning the result of the execution. DO NOT SEND PYTHON CODE TO THIS TOOL.
            """
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.run,
            description="""
                Useful when you need to answer questions about a csv file. Takes and input and returns an answer
                after running pandas calculations.
            """
        )],
        llm=ChatOpenAI(temperature=0, model="gpt-4",),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    # grand_agent.run(
    #     "Generate and save in the current working directory 2 QRcodes that point to https://www.linkedin.com/in/pedro-moyano. You have qrcode package installed already. Use it.")


if __name__ == "__main__":
    main()
