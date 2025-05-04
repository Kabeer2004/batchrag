# BatchRAG CLI

**Chat with entire folders of your local files using local LLMs.**

BatchRAG is a command-line tool designed to bridge the gap between having numerous files scattered across your local machine and the desire to interact with their collective knowledge using Retrieval-Augmented Generation (RAG).

Unlike tools that require uploading files or only handle single documents, BatchRAG lets you create persistent, searchable vector databases (we call them "BatchRAGs") directly from the contents of local folders. You can then chat with these indexed folders using local Large Language Models (LLMs) powered by Ollama.

## The Problem

You have folders full of documents, notes, PDFs, code, and maybe even multimedia files on your PC. Finding specific information requires manually opening files or using basic text search, which often misses context or doesn't work across different file types. Existing RAG tools often involve uploading data to the cloud or are limited to single-file interactions.

## The Solution

BatchRAG provides a simple CLI interface to:

1.  **Index Folders:** Point BatchRAG at a folder. It uses `unstructured` to extract text from various file types (e.g., `.txt`, `.md`, `.pdf`, `.docx`, `.html`, and more depending on dependencies).
2.  **Chunk & Embed:** The extracted text is intelligently chunked and then embedded into vectors using a local Ollama embedding model (like `nomic-embed-text`).
3.  **Store Locally:** These embeddings and the corresponding text chunks are stored locally in a persistent ChromaDB vector database.
4.  **Chat:** Use the `chat` command to interact with your indexed folder. Your questions are embedded, relevant chunks are retrieved from the local database, and a local Ollama LLM (like `llama3`) generates an answer based _only_ on the retrieved context, citing the source files.
5.  **Manage:** Create multiple BatchRAGs for different folders and manage them (list, delete) easily.

**All data processing, storage, and LLM inference happen locally on your machine.**

## Features (Current MVP)

- **Create BatchRAGs:** Index the content of all supported files within a specified folder (recursively).
  - Uses `unstructured` for broad file type support (text-based initially, PDF included).
  - Uses `Ollama` for generating text embeddings (configurable model).
  - Uses `ChromaDB` for local, persistent vector storage.
  - Employs semantic text splitting (paragraphs, sentences) for better chunking.
- **List BatchRAGs:** View all the BatchRAGs you have created, including their source folder and configuration.
- **Delete BatchRAGs:** Remove a specific BatchRAG and its associated vector store data.
- **Chat with BatchRAGs:**
  - Start an interactive chat session with any created BatchRAG.
  - Uses `Ollama` for chat generation (configurable model).
  - Performs local RAG: embeds query, retrieves relevant text chunks from ChromaDB.
  - Context-aware responses based _only_ on the documents in that specific BatchRAG.
  - **Source Citation:** Provides citations `[Source: filename.ext]` indicating which file(s) the answer was based on.
- **Purely Local:** No data leaves your machine. Relies on your local Ollama installation.
- **CLI Interface:** Simple and straightforward command-line usage via `typer`.

## Installation & Setup

**Prerequisites:**

1.  **Python:** Python 3.8 or higher recommended.
2.  **Ollama:** You **must** have Ollama installed and running locally. Download from [https://ollama.ai/](https://ollama.ai/).
3.  **Ollama Models:** Pull the models you intend to use. At a minimum, you'll likely need an embedding model and a chat model:
    ```bash
    ollama pull nomic-embed-text # Default embedding model
    ollama pull llama3.2:3b           # Default chat model (or choose another like mistral, llama3.1 etc.)
    ```
    _(Ensure the Ollama application/server is running in the background before using BatchRAG CLI)_

**Setup Steps:**

1.  **Clone or Download:** Get the project files:

    ```bash
    # If using git
    git clone https://github.com/Kabeer2004/batchrag
    cd batchrag-cli

    # Or download the .py files and requirements.txt into a folder
    # cd your-batchrag-folder
    ```

2.  **Create Virtual Environment:** (Recommended)

    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:** Install the required Python packages, including specific versions to avoid compatibility issues:

    ```bash
    pip install -r requirements.txt
    ```

    _(This installs typer, requests, chromadb, unstructured, rich, numpy, pdfminer.six etc.)_

4.  **Download NLTK Data:** `unstructured` requires NLTK data for text processing. Run the following commands in your activated virtual environment:
    ```bash
    python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng
    ```
    _(Alternatively, open a python interpreter, `import nltk`, then run `nltk.download('punkt')` and `nltk.download('punkt_tab')`)_

## Usage

Make sure your virtual environment is activated (`source venv/bin/activate` or `.\venv\Scripts\activate`) and Ollama is running.

All commands are run via `python batchrag_cli.py`.

**1. Create a BatchRAG:**

Index the contents of a folder.

```bash
python batchrag_cli.py create /path/to/your/documents --name my-docs-batch
```

- `/path/to/your/documents`: **Required.** The folder containing files to index.
- `--name my-docs-batch`: **Required.** A unique name for this BatchRAG.
- `--embed-model nomic-embed-text`: (Optional) Specify the Ollama _embedding_ model.
- `--chunk-size 1000`: (Optional) Target character size for text chunks.
- `--chunk-overlap 150`: (Optional) Character overlap between chunks.

**2. List Existing BatchRAGs:**

See which BatchRAGs you have created.

```bash
python batchrag_cli.py list
```

**3. Chat with a BatchRAG:**

Start an interactive session.

```bash
python batchrag_cli.py chat my-docs-batch
```

- `my-docs-batch`: **Required.** The name of the BatchRAG you want to chat with.
- `--llm llama3`: (Optional) Specify the Ollama _chat_ model to use.
- `--k 4`: (Optional) Number of relevant chunks to retrieve for context.
- `--system "Custom prompt"`: (Optional) Set a custom system prompt for the LLM.

Type your questions, and type `exit` or `quit` to end the session.

**4. Delete a BatchRAG:**

Remove a BatchRAG and its associated data.

```bash
python batchrag_cli.py delete my-docs-batch
```

- `my-docs-batch`: **Required.** The name of the BatchRAG to delete.
- `--force`: (Optional) Skip the confirmation prompt.

**5. Get Help:**

See all available commands and options.

```bash
python batchrag_cli.py --help
# Get help for a specific command
python batchrag_cli.py create --help
python batchrag_cli.py chat --help
```

## Future Roadmap

This is just the MVP. Potential future enhancements include:

- **GUI Application:** Develop a user-friendly graphical interface (likely using Tkinter, PyQt, or maybe a web framework like Streamlit/FastAPI run locally).
- **File Explorer Integration:** Add a right-click context menu option in Windows Explorer / macOS Finder / Linux File Managers to quickly create a BatchRAG from a folder ("BatchRAG This!").
- **Add/Update Functionality:** Allow adding new files to an existing BatchRAG or detecting changes in the source folder to update the index.
- **Enhanced File Support:**
  - Image processing (OCR) via Tesseract integration with `unstructured`.
  - Audio/Video transcription via Whisper integration (potentially via Ollama multi-modal or separate tools).
- **Advanced Citations:** Pinpoint the exact text segment or page number (for PDFs) or timestamp (for audio/video) related to the answer.
- **API Model Support:** Integrate with cloud LLM APIs (OpenAI, Anthropic, Gemini) as an alternative/addition to local Ollama.
- **Improved Splitting/Chunking:** Explore more advanced chunking strategies (e.g., based on headings, code structures).
- **Metadata Filtering:** Allow filtering retrieved documents based on metadata (e.g., file type, date).
- **Performance Optimizations:** Use multiprocessing/batching for faster embedding and indexing.
- **Configuration File:** Allow setting defaults (Ollama URL, models, paths) in a config file.

## Contributing

Contributions are welcome! If you have suggestions, find bugs, or want to contribute code, please feel free to open an issue or submit a pull request on the project repository (if applicable).

## License

MIT License
