from urllib import response
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil,os

from ingestion import load_and_split_pdf, store_documents
from agent import agent_executor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/view-pdf", StaticFiles(directory=UPLOAD_DIR), name="view-pdf")

@app.post("/upload/")
async def upload_file(files: list[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed at once.")
    
    processed_shelves = []
    shelf_name = "default"
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the uploaded PDF
        docs = load_and_split_pdf(file_path)

        # If result is a string, it's an error message (e.g., page limit)
        if isinstance(docs, str):
            raise HTTPException(status_code=400, detail=docs)

        # Create a 'shelf' name from the filename (e.g., "research_paper")
        shelf_name = file.filename.replace(".pdf", "").replace(" ", "_").lower()

        store_documents(docs,namespace_name=shelf_name)
        processed_shelves.append(shelf_name)

    return {"message": f"Processed {len(processed_shelves)} documents successfully.", "shelves": processed_shelves}

# @app.post("/query/")
# async def query(q: str):
#     response = agent_executor.invoke({"input": q})
#     structured_sources = []
#     seen_sources = set() # 👈 To prevent duplicates

#     steps = response.get("intermediate_steps", [])
#     for action, observation in steps:
#         if hasattr(action, 'tool') and action.tool == "DocumentSearch":
#             chunks = str(observation).split("\n---\n")
#             for chunk in chunks:
#                 if "|||" in chunk and chunk not in seen_sources:
#                     try:
#                         seen_sources.add(chunk) # 👈 Mark as seen
#                         parts = chunk.split("|||")
#                         structured_sources.append({
#                             "filename": parts[0].replace("SOURCE:", "").strip(),
#                             "page": parts[1].replace("PAGE:", "").strip(),
#                             "content": parts[2].replace("CONTENT:", "").strip()
#                         })
#                     except: continue

#     return {"answer": response["output"], "sources": structured_sources}

@app.post("/query/")
async def query(q: str):
    response = agent_executor.invoke({"input": q})
    structured_sources = []
    seen_chunks = set()

    steps = response.get("intermediate_steps", [])
    
    if steps:
        # Get the very last tool that was used to get the answer
        last_action, last_observation = steps[-1]
        
        # CASE 1: DocumentSearch was the winner
        if last_action.tool == "DocumentSearch":
            chunks = str(last_observation).split("\n---\n")
            for chunk in chunks:
                if "|||" in chunk and chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    parts = chunk.split("|||")
                    structured_sources.append({
                        "type": "pdf",
                        "filename": parts[0].replace("SOURCE:", "").strip(),
                        "page": parts[1].replace("PAGE:", "").strip(),
                        "content": parts[2].replace("CONTENT:", "").strip()
                    })

        # CASE 2: WebSearch was the winner
        elif last_action.tool == "WebSearch":
            # If you updated serp_tool to return links (recommended), 
            # you would parse them here. Otherwise, we send a generic web type.
            structured_sources.append({
                "type": "web",
                "filename": "Web Source",
                "link": "https://www.google.com/search?q=" + q.replace(" ", "+") 
            })

        # CASE 3: GeneralChat (No sources added)
        elif last_action.tool == "GeneralChat":
            structured_sources = []

    return {"answer": response["output"], "sources": structured_sources}

