"""
kagents/example_rag_agent.py
----------------------------
Full RAG (Retrieval-Augmented Generation) agent example using kagents.

This file matches the usage pattern requested by the user:
  1. Load a HuggingFace dataset as source documents
  2. Split + embed with ChromaDB
  3. Define a RetrieverTool(Tool)
  4. Create and run a CodeAgent
  5. Get the final answer

Install dependencies before running (in Kaggle notebook):
    !pip install -q datasets chromadb sentence-transformers langchain transformers tqdm

Run this file inside a @kbench.task if used in the Kaggle Benchmarks context.
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import os
from tqdm import tqdm

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer

from kagents import CodeAgent, Document, Tool, ToolInput

# ---------------------------------------------------------------------------
# 1. Load source documents from HuggingFace Hub
# ---------------------------------------------------------------------------
knowledge_base = load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(
        page_content=doc["text"],
        metadata={"source": doc["source"].split("/")[1]},
    )
    for doc in knowledge_base.select(range(5))
]

# ---------------------------------------------------------------------------
# 2. Split documents (deduplicated)
# ---------------------------------------------------------------------------
print("Splitting documents...")

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    # source_docs are kagents.Document; text_splitter expects objects with .page_content
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

print(f"  → {len(docs_processed)} unique chunks produced.")

# ---------------------------------------------------------------------------
# 3. Embed + store in ChromaDB
# ---------------------------------------------------------------------------
db_path = "./chroma_db"

print("Preparing vector store...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# check if Chroma DB already exists
if os.path.exists(db_path) and os.listdir(db_path):
    print("Loading existing ChromaDB...")
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
else:
    print("Embedding documents... (may take a few minutes)")
    vector_store = Chroma.from_documents(
        docs_processed,
        embedding=embeddings,
        persist_directory=db_path,
    )
    vector_store.persist()

print("  → Vector store ready.")

# ---------------------------------------------------------------------------
# 4. Define RetrieverTool
# ---------------------------------------------------------------------------
class RetrieverTool(Tool):
    """
    Semantic-search retriever backed by a Chroma vector store.
    Finds the k most relevant documentation chunks for a query.
    """

    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be "
        "most relevant to answer your query."
    )
    inputs = {
        "query": ToolInput(
            type="string",
            description=(
                "The query to perform. This should be semantically close to your target "
                "documents. Use the affirmative form rather than a question."
            ),
            required=True,
        ),
        "topk": ToolInput(
            type="integer",
            description="The number of documents to retrieve.",
            required=True,
        ),
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str, topk: int) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vector_store.similarity_search(query, k=topk)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {i} =====\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )


# ---------------------------------------------------------------------------
# 5. Build the agent and run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever_tool = RetrieverTool(vector_store)

    agent = CodeAgent(
        tools=[retriever_tool],
        model=kbench.llm,
    max_steps=25,
    verbosity_level=2,
    stream_outputs=True,
    additional_instructions="Your answer only based on the knowledge base, if there is no useful information, response I dont know."
)

    agent_output = agent.run("How can I push a model to the Hub?")

    print("\nFinal output:")
    print(agent_output)
