import os
from typing import TypedDict, Annotated, List, Any, Union
import operator
from pathlib import Path
from loguru import logger

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END

from variable_importance.utils.code_executer import OutputCapturingExecutor, ExecutionResult
from variable_importance.utils.output_manager import OutputManager


def cp_data_to_folder(input_path: str, output_path: str) -> None:
    import shutil
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    shutil.copy(input_path, output_path)

class Code(BaseModel):
    code: str = Field(..., description="Python code to be executed")
    task: str = Field(..., description="Description of the task to be performed")

class State(TypedDict):
    code: str
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    input_data_path: str
    stage_name: str
    code_output: ExecutionResult
    summary: str
    graph_summaries: Annotated[List[str], operator.add]

class GraphState(TypedDict):
    graph_summary: str
    graph_summaries: Annotated[List[str], operator.add]

class ExecutionDeps(TypedDict):
    executor: OutputCapturingExecutor
    output_manager: OutputManager

def determine_if_code_needed(state: State, runtime: Runtime[ExecutionDeps]):
    if state.get("input_data_path") is None:
        return False
    user_query = state["messages"]
    class CodeNeeded(BaseModel):
        code_needed: bool = Field(..., description="Whether code is needed to answer the user's query or can be answered directly.")
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_llm = llm.with_structured_output(CodeNeeded)
    result = structured_llm.invoke([
        SystemMessage(
            content=f"You are a helpful data science agent that determines whether code is needed to answer the user's query."),
        *user_query,
    ])
    return result.code_needed

def write_code_for_task(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434")

    structured_code_llm = code_llm.with_structured_output(Code)

    input_data_path = Path(state["input_data_path"])
    user_query = state["messages"]
    logger.info(f"Input data path for code generation: {input_data_path}")
    code = structured_code_llm.invoke([
        SystemMessage(
            content=f"""
            You are a helpful data science agent that writes Python code to perform data analysis tasks. 
            Write Python code to load data from `{input_data_path.name}` and solve the users query. 
            Any plots should be saved to files in the current working directory. 
            Use pandas for data manipulation and any necessary libraries for analysis.
            Always insure that the code you write loads the data from `{input_data_path.name}`.
            """),
        *user_query,
    ])
    
    return {'code': code.code}

def move_data_to_execution_folder(state: State, runtime: Runtime[ExecutionDeps]) -> None:
    stage_name = state["stage_name"]
    path = runtime.context["output_manager"].get_stage_dir(stage_name)
    execution_path = path / "execution"/ Path(state['input_data_path']).name
    cp_data_to_folder(state["input_data_path"], execution_path)
    return {}

def interpret_graph(state: GraphState, runtime: Runtime[ExecutionDeps]) -> str:
    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url="http://100.91.155.118:11434/"
    )
    summary = llm.invoke([
        SystemMessage(
            content="""
            You are a helpful data science agent that summarizes the results of code execution. 
            Provide a detailed summary of the code execution results and the statistical findings. 
            Justify your summary with the output and errors from the code execution."""),
        HumanMessage(
            content=f"The graph summary is: {state['graph_summary']}")
    ])
    return {'graph_summary': summary.content}

async def execute_code(state: State, runtime: Runtime[ExecutionDeps]) -> Any:
    executor = runtime.context["executor"]
    output_manager = runtime.context["output_manager"]
    stage_name = state["stage_name"]
    code = state["code"]

    result = await executor.execute_with_output_manager(
        code=code,
        stage_name=stage_name,
        output_manager=output_manager,
        code_filename="code_with_data.py"
    )

    return {'code_output': result}

def code_fix(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434")
    structured_code_llm = code_llm.with_structured_output(Code)
    code = state["code"]
    error = state["code_output"].error
    output = state['code_output'].stdout
    fixed_code = structured_code_llm.invoke([
        SystemMessage(
            content="You are a helpful data science agent that writes Python code. The following code has an error. Please fix the code to resolve the error."),
        AIMessage(content=code),
        HumanMessage(content=f"The error message is: {error}, and current output is {output}"),
    ])
    return {'code': fixed_code.code}

def check_execution_success(state: State, runtime: Runtime[ExecutionDeps]) -> bool:
    if state["code_output"].success:
        return "SUCCESS"
    else:
        return "FAILURE"

def summarize_results(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    code_output = state.get("code_output", None)
    stdout = ""
    error = ""
    if not code_output is None:
        stdout = code_output.stdout
        error = code_output.error

    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url="http://100.91.155.118:11434/"
    )
    summary = llm.invoke([
        SystemMessage(
            content="""
            You are a helpful data science agent that uses the results of code execution to answer a user's query. 
            Provide a detailed summary of the code execution results and answer the user's query. 
            Justify your summary and answer with the output and errors from the code execution.
            Do not mention code in your summary unless absolutely necessary. Focus on the results and insights of the analysis.
            """),
        *state["messages"],
        HumanMessage(
            content=f"The code was executed with the following output:\n\n{stdout}\n\nAnd the following error (if any):\n\n{error}")
    ])
    return {'summary': summary.content}

code_graph = StateGraph(State, ExecutionDeps)

code_graph.add_node("write_code", write_code_for_task)
code_graph.add_node("move_data", move_data_to_execution_folder)
code_graph.add_node("execute_code", execute_code)
code_graph.add_node("fix_code", code_fix)
code_graph.add_node("summarize_results", summarize_results)

code_graph.add_conditional_edges(START, determine_if_code_needed, {True: "write_code", False: "summarize_results"})


code_graph.add_edge("write_code", "move_data")
code_graph.add_edge("move_data", "execute_code")
code_graph.add_conditional_edges(
    "execute_code",
    check_execution_success,
    {"SUCCESS": "summarize_results", "FAILURE": "fix_code"}
)
code_graph.add_edge("fix_code", "execute_code")

code_graph.add_edge("summarize_results", END)


code_workflow = code_graph.compile()

