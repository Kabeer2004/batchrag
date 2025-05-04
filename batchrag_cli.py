import typer
import os
import json
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Use direct clients/libraries
import chromadb
from unstructured.partition.auto import partition as unstructured_partition # Use unstructured directly
import requests # For Ollama checks

# Import our helper modules
import ollama_client
from text_splitter import recursive_text_splitter

# For pretty printing and user feedback
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# --- Configuration ---
APP_NAME = "batchrag-cli"
APP_DIR = Path(typer.get_app_dir(APP_NAME, force_posix=True))
VECTORSTORE_DIR = APP_DIR / "vectorstores"
MANIFEST_FILE = APP_DIR / "manifest.json"

# Ensure base directories exist
APP_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# --- State Management --- (Identical to previous version)

def load_manifest() -> Dict[str, Any]:
    """Loads the manifest file containing info about created BatchRAGs."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[bold red]Error:[/bold red] Manifest file {MANIFEST_FILE} is corrupted. Starting fresh.")
            return {}
    return {}

def save_manifest(manifest: Dict[str, Any]):
    """Saves the manifest file."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

# --- Core Logic ---

def create_batch_rag(
    folder_path: Path,
    name: str,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """Creates a new BatchRAG by indexing files in a folder without LangChain."""
    console = Console()
    manifest = load_manifest()

    if name in manifest:
        console.print(f"[bold red]Error:[/bold red] BatchRAG named '{name}' already exists.")
        raise typer.Exit(code=1)

    if not folder_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Folder not found: {folder_path}")
        raise typer.Exit(code=1)

    # Use collection name directly for ChromaDB path management within its directory
    # ChromaDB handles persistence within this path
    chroma_persist_path = str(VECTORSTORE_DIR / name) # Directory for this specific batch

    if Path(chroma_persist_path).exists():
         # Chroma's PersistentClient will load existing data if path exists.
         # We might want to explicitly delete it if we intend 'create' to always be fresh.
         # For now, let's warn and proceed (Chroma handles loading/appending).
         # A better approach for a strict 'create' might be to delete first:
         # shutil.rmtree(chroma_persist_path)
         console.print(f"[bold yellow]Warning:[/bold yellow] Data directory {chroma_persist_path} already exists. ChromaDB might load existing data.")


    console.print(f"Creating BatchRAG '{name}' from folder: {folder_path}")
    console.print(f"Using embedding model: {embedding_model}")

    # --- Check Ollama Connectivity ---
    console.print("Checking Ollama connection and embedding model...")
    try:
        if not ollama_client.check_model_availability(embedding_model):
             console.print(f"\n[bold red]Error:[/bold red] Ollama embedding model '{embedding_model}' not found or Ollama is not running.")
             console.print("Please ensure Ollama is running and run: [code]ollama pull {embedding_model}[/code]")
             raise typer.Exit(code=1)
        # Try a quick test embed
        ollama_client.get_embedding("test", embedding_model)
        console.print(f"[green]Successfully connected to Ollama embedding model '{embedding_model}'.[/green]")
    except (ConnectionError, RuntimeError, requests.exceptions.RequestException) as e:
        console.print(f"\n[bold red]Error connecting to Ollama:[/bold red]\n{e}")
        raise typer.Exit(code=1)


    # --- File Loading & Processing ---
    all_files = [p for p in folder_path.rglob("*.*") if p.is_file()]
    docs_data = [] # Store tuples of (text_content, metadata)
    processed_files_count = 0
    failed_files = []

    console.print(f"Found {len(all_files)} potential files.")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        TimeElapsedColumn(),
        transient=False, # Keep progress visible after completion
    ) as progress:
        task_load = progress.add_task("Loading files...", total=len(all_files))

        for file_path in all_files:
            relative_path = str(file_path.relative_to(folder_path))
            progress.update(task_load, description=f"Loading: {file_path.name}")
            try:
                # Use unstructured to partition the file; returns a list of "elements"
                elements = unstructured_partition(filename=str(file_path), strategy="auto")
                # Concatenate text from all elements for this file
                # You could also choose to chunk per element later if needed
                file_text = "\n".join([el.text for el in elements if hasattr(el, 'text')]).strip()

                if file_text: # Only process if text was extracted
                    docs_data.append({
                        "text": file_text,
                        "metadata": {"source": relative_path}
                    })
                    processed_files_count += 1
                else:
                     failed_files.append(f"{relative_path} (no text extracted)")

            except Exception as e:
                console.print(f"\n[yellow]Warning:[/yellow] Failed to process file {relative_path}: {e}", style="dim")
                failed_files.append(f"{relative_path} ({type(e).__name__})")
            finally:
                progress.update(task_load, advance=1)
        progress.update(task_load, description=f"File loading complete.")


    if not docs_data:
        console.print("[bold red]Error:[/bold red] No text could be extracted from any files. Aborting.")
        raise typer.Exit(code=1)

    console.print(f"Successfully extracted text from {processed_files_count} files. {len(failed_files)} files failed or yielded no text.")
    if failed_files:
         console.print("[yellow]Files failed/skipped:[/yellow]\n - " + "\n - ".join(failed_files))

    # --- Text Splitting ---
    all_chunks_texts = []
    all_chunks_metadatas = []
    console.print(f"Splitting text using recursive strategy (chunk size: {chunk_size}, overlap: {chunk_overlap})...")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
         task_split = progress.add_task("Splitting text into chunks...", total=len(docs_data))
         total_chunks_generated = 0
         for doc in docs_data:
             chunks = recursive_text_splitter(doc["text"], chunk_size, chunk_overlap)
             for i, chunk_text in enumerate(chunks):
                 all_chunks_texts.append(chunk_text)
                 # Create metadata for each chunk, including a unique part index
                 chunk_metadata = doc["metadata"].copy()
                 chunk_metadata["part"] = i # Add part index to metadata
                 all_chunks_metadatas.append(chunk_metadata)
             progress.update(task_split, advance=1)
             total_chunks_generated += len(chunks) # Keep track of actual chunks

    # Use the actual number of generated chunks
    num_chunks = total_chunks_generated # Update num_chunks based on actual count
    console.print(f"Split content into {num_chunks} chunks.")
    if num_chunks == 0:
        console.print("[bold red]Error:[/bold red] No text chunks were generated after splitting. Aborting.")
        raise typer.Exit(code=1)


    # --- Embedding Chunks ---
    all_embeddings = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task_embed = progress.add_task(f"Generating embeddings ({embedding_model})...", total=num_chunks)
        # Consider batching calls to ollama_client.get_embedding if Ollama API supports it
        # For now, embed one by one
        for i, text_chunk in enumerate(all_chunks_texts):
            try:
                embedding = ollama_client.get_embedding(text_chunk, model=embedding_model)
                all_embeddings.append(embedding)
            except (ConnectionError, RuntimeError) as e:
                 console.print(f"\n[bold red]Error embedding chunk {i} from {all_chunks_metadatas[i]['source']}:[/bold red]\n{e}")
                 console.print("Aborting embedding process.")
                 raise typer.Exit(code=1)
            except Exception as e:
                 console.print(f"\n[bold red]Unexpected error embedding chunk {i} from {all_chunks_metadatas[i]['source']}:[/bold red]\n{e}")
                 console.print("Aborting embedding process.")
                 raise typer.Exit(code=1)

            progress.update(task_embed, advance=1)
        progress.update(task_embed, description="Embedding generation complete.")


    # --- Initialize ChromaDB Client and Add Data ---
    console.print(f"Initializing vector store at: {chroma_persist_path}")
    try:
        # Initialize persistent client - creates directory if it doesn't exist
        chroma_client = chromadb.PersistentClient(path=chroma_persist_path)

        # Create or get collection. Naming convention: use the batchrag name.
        # No embedding function needed here, as we provide embeddings directly.
        collection = chroma_client.get_or_create_collection(
            name=name,
            # Optionally add metadata about the collection itself
            # metadata={"hnsw:space": "cosine"} # Example: specify distance metric if needed
        )

        # Add embedded chunks to the collection
        # Generate unique IDs for each chunk
        chunk_ids = [str(uuid.uuid4()) for _ in all_chunks_texts]

        # Add in batches to avoid potential issues with very large single requests
        batch_size = 512 # Adjust as needed
        num_batches = (num_chunks + batch_size - 1) // batch_size

        console.print(f"Adding {num_chunks} embeddings to ChromaDB collection '{name}'...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total} batches"), TimeElapsedColumn()) as progress:
            task_add = progress.add_task("Indexing batches...", total=num_batches)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_chunks)

                progress.update(task_add, description=f"Indexing batch {i+1}/{num_batches}")

                collection.add(
                    embeddings=all_embeddings[start_idx:end_idx],
                    documents=all_chunks_texts[start_idx:end_idx],
                    metadatas=all_chunks_metadatas[start_idx:end_idx],
                    ids=chunk_ids[start_idx:end_idx]
                )
                progress.update(task_add, advance=1)
            progress.update(task_add, description="ChromaDB indexing complete.")


    except Exception as e:
        console.print(f"\n[bold red]Error interacting with ChromaDB:[/bold red] {e}")
        # Attempt cleanup of potentially partial/corrupted ChromaDB directory
        if Path(chroma_persist_path).exists():
             console.print(f"Attempting to clean up directory: {chroma_persist_path}")
             shutil.rmtree(chroma_persist_path, ignore_errors=True)
        raise typer.Exit(code=1)


    # --- Update and save manifest ---
    manifest[name] = {
        "name": name,
        "original_folder": str(folder_path.resolve()),
        "vectorstore_path": chroma_persist_path, # Store the directory path
        "collection_name": name, # Chroma collection name (same as batch name)
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "processed_files_count": processed_files_count,
        "num_chunks": num_chunks,
        "failed_files": failed_files,
    }
    save_manifest(manifest)

    console.print(Panel(f"[bold green]Success![/bold green] BatchRAG '{name}' created.\nIndexed {num_chunks} chunks from {processed_files_count} files.",
                        title="Creation Complete", border_style="green"))


# --- CLI Commands ---

app = typer.Typer(help="BatchRAG CLI (No LangChain): Create and chat with RAG models for local folders.")
console = Console()


@app.command()
def create(
    folder_path: Path = typer.Argument(..., help="Path to the folder containing files to index.", exists=True, file_okay=False, resolve_path=True),
    name: str = typer.Option(..., "--name", "-n", help="Unique name for this BatchRAG."),
    embedding_model: str = typer.Option("nomic-embed-text", "--embed-model", "-e", help="Ollama embedding model to use."),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-c", help="Target chunk size for splitting text."),
    chunk_overlap: int = typer.Option(150, "--chunk-overlap", "-o", help="Overlap size between chunks."),
):
    """
    Creates a new BatchRAG from a specified folder (No LangChain).
    """
    create_batch_rag(folder_path, name, embedding_model, chunk_size, chunk_overlap)


@app.command(name="list")
def list_batches():
    """
    Lists all created BatchRAGs.
    """
    manifest = load_manifest()
    if not manifest:
        console.print("No BatchRAGs found.")
        return

    console.print(Panel("[bold]Available BatchRAGs:[/bold]", expand=False))
    for name, data in manifest.items():
        console.print(f"- [cyan]{name}[/cyan]")
        console.print(f"  Folder: {data.get('original_folder', 'N/A')}")
        console.print(f"  Chroma Path: {data.get('vectorstore_path', 'N/A')}")
        console.print(f"  Embedding Model: {data.get('embedding_model', 'N/A')}")
        console.print(f"  Files Processed: {data.get('processed_files_count', 'N/A')} ({data.get('num_chunks', 'N/A')} chunks)")
        if data.get("failed_files"):
             console.print(f"  [yellow]Files Failed/Skipped:[/yellow] {len(data['failed_files'])}")
    console.print(f"\nManage data at: {APP_DIR}")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Name of the BatchRAG to delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
):
    """
    Deletes a specific BatchRAG, including its vector store directory.
    """
    manifest = load_manifest()
    if name not in manifest:
        console.print(f"[bold red]Error:[/bold red] BatchRAG '{name}' not found.")
        raise typer.Exit(code=1)

    vectorstore_path = Path(manifest[name].get("vectorstore_path", ""))
    collection_name = manifest[name].get("collection_name", name) # Get collection name

    if not force:
        console.print(f"This will delete the BatchRAG '{name}', its manifest entry, and the entire directory: {vectorstore_path}")
        confirmation = typer.confirm("Are you sure?")
        if not confirmation:
            console.print("Deletion aborted.")
            raise typer.Exit()

    # Delete the entire vector store directory managed by this app
    deleted_dir = False
    if vectorstore_path.exists() and vectorstore_path.is_dir() and VECTORSTORE_DIR in vectorstore_path.parents:
        try:
            shutil.rmtree(vectorstore_path)
            console.print(f"Deleted vector store directory: {vectorstore_path}")
            deleted_dir = True
        except OSError as e:
            console.print(f"[bold red]Error deleting directory {vectorstore_path}:[/bold red] {e}")
            console.print("Please remove it manually.")
            # Proceed to remove from manifest even if directory deletion fails

    # Optional: Try deleting the collection via ChromaDB client if the directory wasn't deleted
    # This might be useful if the directory couldn't be removed but the DB connection works
    if not deleted_dir and vectorstore_path.exists(): # Only try if directory deletion failed but path exists
        try:
            chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR / name)) # Point to the specific dir
            chroma_client.delete_collection(name=collection_name)
            console.print(f"Deleted ChromaDB collection '{collection_name}' via client.")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not delete Chroma collection '{collection_name}' via client (directory removal might have failed): {e}")


    # Remove from manifest
    del manifest[name]
    save_manifest(manifest)
    console.print(f"[bold green]Success:[/bold green] BatchRAG '{name}' removed from manifest.")


@app.command()
def chat(
    name: str = typer.Argument(..., help="Name of the BatchRAG to chat with."),
    llm_model: str = typer.Option("llama3.2:3b", "--llm", "-m", help="Ollama model to use for chatting."),
    k_results: int = typer.Option(4, "--k", help="Number of relevant document chunks to retrieve."),
    # Update default system prompt for citation
    system_prompt: str = typer.Option(
        "You are a helpful assistant. Answer the user's question based *only* on the provided context. "
        "After your answer, cite the relevant source file(s) used from the list provided in the format [Source: filename.ext].",
        "--system", "-s", help="System prompt for the LLM."
    ),
):
    """
    Starts an interactive chat session with a specified BatchRAG (No LangChain).
    Includes source citation based on retrieved documents.
    """
    manifest = load_manifest()
    if name not in manifest:
        console.print(f"[bold red]Error:[/bold red] BatchRAG '{name}' not found.")
        raise typer.Exit(code=1)

    rag_data = manifest[name]
    vectorstore_path = Path(rag_data["vectorstore_path"])
    collection_name = rag_data["collection_name"]
    embedding_model_name = rag_data["embedding_model"]

    if not vectorstore_path.exists() or not vectorstore_path.is_dir():
         console.print(f"[bold red]Error:[/bold red] Vector store path not found or invalid for '{name}': {vectorstore_path}")
         console.print("The data may have been moved or deleted. Try recreating the BatchRAG.")
         raise typer.Exit(code=1)

    console.print(f"Loading BatchRAG '{name}'...")
    console.print(f"Chroma Path: {vectorstore_path}")
    console.print(f"Collection: {collection_name}")
    console.print(f"Embedding Model: {embedding_model_name}")
    console.print(f"Chat LLM: {llm_model}")

    # --- Check Ollama Connectivity for LLM ---
    console.print("Checking Ollama connection and chat model...")
    try:
        if not ollama_client.check_model_availability(llm_model):
             console.print(f"\n[bold red]Error:[/bold red] Ollama chat model '{llm_model}' not found or Ollama is not running.")
             console.print("Please ensure Ollama is running and run: [code]ollama pull {llm_model}[/code]")
             raise typer.Exit(code=1)
        console.print(f"[green]Successfully connected to Ollama chat model '{llm_model}'.[/green]")
    except (ConnectionError, RuntimeError, requests.exceptions.RequestException) as e:
        console.print(f"\n[bold red]Error connecting to Ollama:[/bold red]\n{e}")
        raise typer.Exit(code=1)


    # --- Initialize ChromaDB Client and Get Collection ---
    try:
        chroma_client = chromadb.PersistentClient(path=str(vectorstore_path))
        collection = chroma_client.get_collection(name=collection_name)
        console.print(f"Retriever ready, will fetch {k_results} chunks from collection '{collection_name}'.")
    except Exception as e:
        console.print(f"\n[bold red]Error loading vector store from {vectorstore_path}:[/bold red]\n{e}")
        console.print("The vector store might be corrupted or incompatible. Try recreating the BatchRAG.")
        raise typer.Exit(code=1)


    # --- RAG and Chat Loop ---
    console.print(Panel(f"[bold]Chatting with BatchRAG '{name}'[/bold]\nType 'exit' or 'quit' to end.", title="Chat Session", border_style="blue"))

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ")
            if question.lower() in ["exit", "quit"]:
                break
            if not question.strip():
                continue

            # 1. Embed the query (no changes here)
            try:
                with console.status("[dim]Generating query embedding...", spinner="dots"):
                    query_embedding = ollama_client.get_embedding(question, model=embedding_model_name)
            except (ConnectionError, RuntimeError) as e:
                 console.print(f"\n[bold red]Error embedding query:[/bold red] {e}")
                 continue

            # 2. Retrieve relevant chunks from ChromaDB (no changes here)
            try:
                with console.status("[dim]Retrieving relevant documents...", spinner="dots"):
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k_results,
                        include=['documents', 'metadatas'] # Ensure metadatas are included
                    )
            except Exception as e:
                console.print(f"\n[bold red]Error querying ChromaDB:[/bold red] {e}")
                continue

            retrieved_documents = results.get('documents', [[]])[0]
            retrieved_metadatas = results.get('metadatas', [[]])[0]

            if not retrieved_documents:
                console.print("[yellow]AI:[/yellow] I couldn't find any relevant information in the indexed documents to answer your question.")
                continue

            # --- Start Citation Modification ---

            # 3. Extract unique source filenames and format context/prompt
            unique_sources = sorted(list(set(
                meta['source'] for meta in retrieved_metadatas if meta and 'source' in meta
            ))) # Added check if meta is not None

            # Format the context by joining the retrieved document chunks
            context_str = "\n\n---\n\n".join(retrieved_documents)

            # Format the list of source files to be presented to the LLM
            if unique_sources:
                 sources_list_str = "\n - ".join(unique_sources)
                 sources_prompt_part = f"Source Files Available:\n - {sources_list_str}\n\n"
            else:
                 sources_prompt_part = "Source Files Available: None\n\n"


            # Construct the final prompt for the LLM
            # Explicitly instruct the LLM within the main prompt as well (reinforces system prompt)
            full_prompt = f"""{sources_prompt_part}Context:
            {context_str}

            ---
            Question: {question}

            Answer the question based *only* on the provided context. Cite the relevant source file(s) from the 'Source Files Available' list (e.g., "[Source: file.txt]") after your answer."""

            # Optional: Display retrieved sources to user (for debugging/info)
            console.print(f"[dim]Retrieved sources: {', '.join(unique_sources) or 'None'}[/dim]")

            # --- End Citation Modification ---


            # 4. Call Ollama LLM for generation (streaming)
            console.print(f"[bold green]AI:[/bold green] ", end="")
            full_response = ""
            line_buffer = "" # Buffer to hold text before printing
            try:
                with console.status("[dim]Generating response...", spinner="dots"):
                    response_generator = ollama_client.generate_completion(
                        prompt=full_prompt,
                        model=llm_model,
                        system_message=system_prompt,
                        stream=True
                    )
                    for chunk in response_generator:
                        # Add the raw chunk to the full response history
                        full_response += chunk

                        # Check if the chunk contains a newline character
                        if "\n" in chunk:
                            # Split the chunk by newlines
                            parts = chunk.split("\n")

                            # Print the buffer + the part before the first newline
                            print(line_buffer + parts[0], end="", flush=True)

                            # Print any intermediate full lines from the chunk
                            for part in parts[1:-1]:
                                print() # Go to the next line in the output
                                print(part, end="", flush=True) # Print the intermediate part

                            # If there was content after the last newline in the chunk,
                            # start a new line in the output and store that content in the buffer.
                            # Otherwise, clear the buffer.
                            if len(parts) > 1: # Ensure there was at least one newline
                                print() # Start a new line for the potentially buffered content
                                line_buffer = parts[-1] # Store the last part (after the last \n)
                            else:
                                # This happens if the chunk itself *is* just "\n" or ends with "\n"
                                line_buffer = "" # Nothing to buffer

                        else:
                            # No newline in this chunk, just add it to the buffer
                            line_buffer += chunk

                    # After the loop finishes, print any remaining text in the buffer
                    if line_buffer:
                        print(line_buffer, end="", flush=True)

                 # Ensure the final output ends with a newline
                print()
                
            except (ConnectionError, RuntimeError) as e:
                 print("\n[bold red]Error generating response from Ollama:[/bold red]", e)
            except Exception as e:
                 print(f"\n[bold red]Unexpected error during generation:[/bold red]", e)


        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"\n[bold red]An unexpected error occurred in the chat loop:[/bold red] {e}")

    console.print("\nChat session ended.")


if __name__ == "__main__":
    app()