import streamlit as st
import json
import os

# File paths
CHUNKS_FILE = "Results/chunks_for_labeling.jsonl"
LABELED_FILE = "Results/labeled_chunks.jsonl"

# Load chunks
if not os.path.exists(CHUNKS_FILE):
    st.error(f"Chunk file not found: {CHUNKS_FILE}")
    st.stop()

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Track progress
if os.path.exists(LABELED_FILE):
    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        labeled_chunks = [json.loads(line) for line in f]
    labeled_ids = {chunk["chunk_id"] for chunk in labeled_chunks}
else:
    labeled_chunks = []
    labeled_ids = set()

# Find next unlabeled chunk
remaining_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in labeled_ids]

st.title("PDF Chunk Labeling Tool")
st.write(f"Total Chunks: {len(chunks)} | Labeled: {len(labeled_ids)} | Remaining: {len(remaining_chunks)}")

if not remaining_chunks:
    st.success("✅ All chunks labeled! You can close this now.")
    st.stop()

chunk = remaining_chunks[0]

# Show chunk text
st.subheader(f"Chunk ID: {chunk['chunk_id']}")
st.text_area("Text Chunk:", chunk["text"], height=300)

# Label buttons
col1, col2 = st.columns(2)

if col1.button("✅ Content"):
    chunk["label"] = 1
    with open(LABELED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    st.experimental_rerun()

if col2.button("❌ Non-content"):
    chunk["label"] = 0
    with open(LABELED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    st.experimental_rerun()     


