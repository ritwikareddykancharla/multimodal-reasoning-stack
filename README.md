# multimodal-reasoning-stack

Build a full-stack enterprise reasoning engine with the following architecture:

**Backend (FastAPI + Python):**
- Create an async file ingestion pipeline at `/api/ingest` that accepts:
  - PDFs, DOCX, PPTX
  - Audio files (MP3, WAV)
  - Video files (MP4)
  - Images (JPG, PNG)
- Use Gemini 3 API (google-genai) for:
  - OCR on images (including handwritten text)
  - Audio transcription with speaker diarization
  - Video frame extraction and analysis
- Store extracted content as "Nodes" with metadata: content_hash, timestamp, source_provenance, modality_type
- Build a Neo4j knowledge graph that connects entities across modalities
- Implement a conflict detection module using Gemini 3's reasoning to identify contradictions between documents
- Create a StrategicReasoner class with a ReAct loop (Thought → Action → Observation) that:
  - Decomposes queries using Gemini 3's chain-of-thought
  - Executes sub-queries with tool use (search, compare, synthesize)
  - Returns answers with multimodal citations (PDF page numbers, audio timestamps, image regions)

**Frontend (Streamlit):**
- Clean, enterprise UI with sidebar for file upload
- Main panel shows:
  - Knowledge graph visualization (using PyVis)
  - Query input box with example: "What's our liability exposure?"
  - Reasoning chain display (show Gemini 3's step-by-step thinking)
  - Output panel with clickable citations that highlight source documents
  - Conflict flags dashboard showing contradictions found

**Key Features for Demo:**
- Pre-ingest sample documents (legal contract, scanned sticky note, Slack screenshot, audio recording)
- Show real-time conflict detection when contradictory info is uploaded
- Generate a strategic report with citations for a query like "What risks did we accept in this deal?"
- Make all source documents viewable with highlighted regions when clicking citations

**Tech Stack:**
- FastAPI, Streamlit, Neo4j, ChromaDB for vectors, google-genai for Gemini 3
- Dockerize everything with docker-compose.yml
- Include comprehensive tests with pytest

Focus on showcasing Gemini 3's multimodal and reasoning capabilities throughout. Optimize for a 3-minute hackathon demo video that demonstrates turning messy enterprise documents into strategic decisions.
