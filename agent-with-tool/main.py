from typing import Union
from dotenv import load_dotenv
from langchain_core.tools import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List
from langchain.agents.format_scratchpad.log import format_log_to_str
from callbacks import AgentCallbackHandler

load_dotenv()

# ツールを定義
@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"get_text_length called with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non-alphabetic characters just in case

    return len(text)

# ツールを選択
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("hello world")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # ツールを選択させるプロンプト
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation"], # Thought以下のテキストは出力する必要無いので、Observationで生成を終了させる
        callbacks=[AgentCallbackHandler()],
    )

    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    # AgentFinishが返却されるまでループ
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of DOG ?",
                "intermediate_steps": intermediate_steps,
            }
        )
        print(agent_step)

        # AgentActionが返却されている際は、ツールを選択し、ループを継続
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)
