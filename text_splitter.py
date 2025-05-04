import re
from typing import List

# A basic regex to split sentences. It's not perfect (e.g., struggles with abbreviations like Mr. Mrs.),
# but handles basic cases like periods, question marks, and exclamation points followed by space/newline.
# Handles cases like "sentence. Next", "sentence! Next", "sentence? Next"
# Need lookbehind (?<=[.!?]) to include the punctuation and \s+ to match following whitespace/newlines.
# Added negative lookbehind (?<!\w\.\w.) and (?<![A-Z][a-z]\.) to avoid splitting on things like e.g., i.e. or Mr.
# This regex can be further improved based on observed edge cases.
SENTENCE_SPLITTER_REGEX = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'

def split_by_sentences(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    if not text:
        return []
    sentences = re.split(SENTENCE_SPLITTER_REGEX, text)
    # Filter out any empty strings that might result from multiple separators together
    return [s.strip() for s in sentences if s and not s.isspace()]

def recursive_text_splitter(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Splits text recursively, prioritizing paragraphs, then sentences.

    Args:
        text: The input text.
        chunk_size: The target maximum size of each chunk (in characters).
        chunk_overlap: The target overlap between consecutive chunks (in characters).

    Returns:
        A list of text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size.")

    if not text or text.isspace():
        return []

    final_chunks = []
    # Start by splitting into paragraphs
    paragraphs = text.split('\n\n')
    all_sentences = []
    for para in paragraphs:
        if para and not para.isspace():
            all_sentences.extend(split_by_sentences(para))

    if not all_sentences: # Handle cases where splitting results in nothing
        # If text is not empty but splitting failed, return as single chunk if small enough
        if len(text) <= chunk_size:
             return [text]
        else: # Fallback to simple chunking if recursive fails and text is large
             print("Warning: Sentence splitting yielded no results, falling back to simple chunking for a large text block.")
             return simple_chunker(text, chunk_size, chunk_overlap)


    current_chunk_sentences = []
    current_chunk_len = 0

    for i, sentence in enumerate(all_sentences):
        sentence_len = len(sentence)

        # Check if adding the next sentence exceeds the chunk size
        # Add 1 for potential space separator between sentences
        if current_chunk_len > 0 and (current_chunk_len + sentence_len + 1) > chunk_size:
            # Finalize the current chunk
            chunk_text = " ".join(current_chunk_sentences)
            final_chunks.append(chunk_text)

            # Start new chunk with overlap
            # Find sentences from the end of the finalized chunk for overlap
            overlap_text = ""
            accumulated_len = 0
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                sent = current_chunk_sentences[j]
                sent_len_with_space = len(sent) + (1 if accumulated_len > 0 else 0)
                if accumulated_len + sent_len_with_space <= chunk_overlap:
                    overlap_text = sent + (" " if accumulated_len > 0 else "") + overlap_text
                    accumulated_len += sent_len_with_space
                else:
                    break # Stop when overlap size is reached

            # Start the new chunk with the overlap text and the current sentence
            current_chunk_sentences = ([overlap_text] if overlap_text else []) + [sentence]
            current_chunk_len = len(" ".join(current_chunk_sentences))

        else:
            # Add sentence to the current chunk
            current_chunk_sentences.append(sentence)
            # Add 1 for potential space separator
            current_chunk_len += sentence_len + (1 if current_chunk_len > 0 else 0)

    # Add the last remaining chunk
    if current_chunk_sentences:
        final_chunks.append(" ".join(current_chunk_sentences))

    # Filter out any potential empty chunks again just in case
    return [c for c in final_chunks if c and not c.isspace()]


# Keep the simple chunker as a potential fallback if needed
def simple_chunker(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into chunks of a target size with overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start forward by chunk_size minus overlap
        start += chunk_size - chunk_overlap
        # Ensure we don't get stuck in an infinite loop if overlap is large and text is short
        if start >= len(text) - chunk_overlap and start != 0:
            break

    # Add the very last part if it wasn't captured
    if start < len(text):
        final_chunk = text[start:]
        if final_chunk and not final_chunk.isspace():
            chunks.append(final_chunk)


    return [c for c in chunks if c and not c.isspace()] # Remove empty/whitespace chunks