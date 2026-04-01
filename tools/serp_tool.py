from config import SERPAPI_API_KEY
from serpapi import GoogleSearch
from langchain_core.tools import Tool, StructuredTool
def search_web(query: str) -> str:
    print("🌐 WebSearch CALLED")
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY
    }

    results = GoogleSearch(params).get_dict()

    # Get the top link to use for the button
    first_link = results.get("organic_results", [{}])[0].get("link", "#")
    
    snippets = [r.get("snippet", "") for r in results.get("organic_results", [])]
    
    # We prefix the first link so main.py can find it easily
    return f"TOP_LINK: {first_link}\n" + "\n".join(snippets)

serp_tool = StructuredTool.from_function(
    name = "WebSearch",
    func = search_web,
    description = "Search the web when documents are insufficient"
)