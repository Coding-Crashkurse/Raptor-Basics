## RAPTOR + Chroma + A2A Demo

This repo contains two coordinated pieces:

1. **`raptor_pipeline.ipynb`** – a notebook that demonstrates a RAPTOR/adRAP-style workflow over the sample text `themen_text.txt`. It splits the document, embeds chunks, clusters and summarizes them, persists embeddings in a local Chroma vector store, and finally writes the *root* summary to `artifacts/root_summary.txt`.
2. **`dummy_server.py`** – a minimalist A2A-compatible FastAPI server that exposes the “Hybrid Pizzeria Agent.” When the server starts it loads `artifacts/root_summary.txt` and uses the content as the `AgentCard.description`, so the agent card always reflects the latest RAPTOR root summary.

### Workflow

1. **Install dependencies**
   ```bash
   uv sync
   # activate the virtual environment if needed
   .\.venv\Scripts\activate
   ```
2. **Run the notebook**
   - Execute all cells in `raptor_pipeline.ipynb`.
   - The sections “Persist chunk embeddings in Chroma” and “Persist Root Summary for External Agents” create `./chroma_langchain_db/` and `artifacts/root_summary.txt`.  
   - Use the incremental-update helper cells to add new documents without rebuilding everything.
3. **Start the A2A server**
   ```bash
   uvicorn dummy_server:app --host 0.0.0.0 --port 8002 --reload
   ```
   The AgentCard served at `http://localhost:8002/` will include the most recent root summary from the notebook.

### Key Concepts

- **RAPTOR/adRAP** – Recursive summarization and clustering to create a multi-level representation of the dataset. adRAP updates only affected clusters when new data arrives.
- **Chroma Vector Store** – Stores chunk embeddings persistently so incremental updates do not require recomputing historical vectors.
- **A2A Agent Card Integration** – By persisting `root_summary.txt`, we keep the remote agent’s description synchronized with the notebook’s latest root node.

Feel free to modify the notebook text or plug in your own documents. Just rerun the root-summary cell and restart the server so the AgentCard stays current.
