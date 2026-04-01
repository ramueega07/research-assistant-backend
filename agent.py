from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from rag_chain import retriever, llm, vector_store
from tools.serp_tool import serp_tool
from memory import memory
from langchain_core.tools import Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from tools.serp_tool import search_web
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
import os

#  RAG TOOL
def document_search(query: str) -> str:
    """
    Searches the internal knowledge base. 
    The agent should provide a query that includes keywords related to the 
    topic or the specific document name if known.
    """
    print("Agentic DocumentSearch CALLED for query: {}".format(query))

    # 1. get all the namespaces (shelves) in the Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    available_namespaces = list(stats['namespaces'].keys())

    # 2. Ask the LLM to select the best shelf (Agentic Selection)
    # We use a mini-prompt here to let the LLM decide which shelf is best
    shelf_selection_prompt = f"""
    Given the user's query: "{query}"
    And the following available document 'shelves' in our database: {available_namespaces}
    
    Which shelf is most likely to contain the answer? 
    Return ONLY the name of the shelf. If none match well, return 'default'.
    """
    
    # We call the LLM directly to make the "choice"
    selected_shelf = llm.invoke(shelf_selection_prompt).content.strip()
    
    # Clean up the response in case the LLM adds extra text
    if selected_shelf not in available_namespaces:
        selected_shelf = "default"

    print(f"🤖 LLM decided to use shelf: '{selected_shelf}'")

    # 3. Perform the actual RAG search on the selected shelf
    # Using the vector_store directly to ensure the namespace is applied
    docs = vector_store.as_retriever(
        search_kwargs={"k": 5, "namespace": selected_shelf}
    ).invoke(query)

    if not docs:
        return "I checked the '{selected_shelf}' shelf, but found no specific details for '{query}'."

    # Format context with human-friendly source references
    context_blocks = []
    for d in docs:
        source = os.path.basename(d.metadata.get('source', 'Unknown'))
        page = d.metadata.get('page', 'N/A')
        content = d.page_content.replace("\n", " ")
        # Simple, parseable format
        context_blocks.append(f"SOURCE:{source}|||PAGE:{page}|||CONTENT:{content}")

    return "\n---\n".join(context_blocks)

rag_tool = StructuredTool.from_function(
    name="DocumentSearch",
    func=document_search,
    description="Use this to search internal documents. Provide a descriptive search query."
)


def general_chat(input_text: str) -> str:
    """Useful for greetings, small talk, or general conversation that doesn't require searching."""
    return input_text # The agent will use this output to formulate its Final Answer

chat_tool = StructuredTool.from_function(
    name="GeneralChat",
    func=general_chat,
    description="Use this for greetings (hi, hello, how are you?) or non-research questions."
)

# ReAct agents REQUIRE {tools} and {tool_names} to be present in the string
template = """Answer the following questions as best you can. 
You are a professional research assistant with access to a library of 'shelves' (namespaces).
You have access to the following tools:{tools}

AGENTIC RULES:
1. Use 'GeneralChat' for greetings and small talk.
2. Use 'DocumentSearch' for any technical or internal document queries.
3. Use 'WebSearch' for general knowledge or when documents lack info.
4. **CRITICAL**: Do NOT include any citations, filenames, page numbers, or "Sources" lists in your Final Answer. Just provide the summarized information directly.
Use the following format:

Question: the input question you must answer
Thought: Do I need a tool for this? 
Action: the action to take, should be one of [{tool_names}] 
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer:[Your clean, summarized response in bullet points. No citations allowed.]

Begin!

History:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


'''
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant.
RULES:
1. You MUST answer using tools only.
2. First, try using DocumentSearch to answer the question.
3. If DocumentSearch does NOT return useful information, then use WebSearch.
4. NEVER answer from your own knowledge.

IMPORTANT:
- Always call one of the tools before answering.
- Do NOT skip tool usage.
- Use the tool outputs to construct the final answer.

When using documents, include citations:
(Document Name, Page Number)"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
'''

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    tools = [serp_tool, rag_tool,chat_tool],
    llm = llm,
    prompt=prompt
)

#agent = create_tool_calling_agent(llm,[serp_tool, rag_tool],prompt)

agent_executor = AgentExecutor(agent=agent, tools=[serp_tool, rag_tool,chat_tool], verbose=True, memory=memory, handle_parsing_errors=True,max_iterations=3,return_intermediate_steps=True,output_key="output")