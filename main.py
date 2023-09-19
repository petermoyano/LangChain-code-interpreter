from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool
from langchain.agents import AgentType


def main():
    print("Start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4",),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # python_agent_executor.run(
    #     "generate and save in the current working directory 2 QRcodes that point to https://www.linkedin.com/in/pedro-moyano, you have already installed the qrcode library, you can use it.")

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4",),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # csv_agent.run("How many columns does the csv file have?")
    csv_agent.run(
        "Which writter wrote the most episodes? How many did he/she write?")


if __name__ == "__main__":
    main()
