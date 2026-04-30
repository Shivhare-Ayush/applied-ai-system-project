# Applied AI System Project: Grounded Agentic QA Pipeline

## Original Project (Modules 1-3)
My original project from Modules 1-3 was **Applied AI System Project**, a retrieval-augmented question-answering prototype designed to reduce hallucinations in LLM responses. The core goal was to combine a RAG retriever with an LLM generator so answers are grounded in retrieved context instead of free-form model memory. The system already supported query intake, retrieval context assembly, and answer generation, and this version extends that with explicit verification and self-correction.

## Title And Summary
This project is an **agentic AI orchestration pipeline** that connects a RAG service and Gemini Flash to produce grounded answers with an additional verifier loop.

Why it matters:
- It demonstrates a practical anti-hallucination pattern for production AI systems.
- It separates retrieval, generation, and validation concerns into clean service boundaries.
- It shows how to fail safely when context is missing, instead of confidently inventing facts.

## Architecture Overview
![System Architecture Diagram](diagram.png)

High-level flow:
1. User sends a natural-language query.
2. `RAGClient` calls the remote `/query` endpoint and receives:
	- `augmented_prompt`
	- retrieved `sources`
	- vector `scores`
3. `LLMService` uses Gemini 2.5 Flash to generate an answer.
4. The verifier prompts the same model to classify the answer as `GROUNDED` or `HALLUCINATION`.
5. If hallucination is detected, one self-correction pass is executed using a strict grounded-only instruction.

## Setup Instructions
1. Clone the repository.
2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your credentials.

```env
GEMINI_API_KEY=your_google_ai_studio_key
RAG_BASE_URL=http://34.122.127.176:8080
```

5. Run the pipeline.

```powershell
python main.py "Where is Paris?"
```

6. Or run with default query.

```powershell
python main.py
```

7. Run automated reliability tests.

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Sample Interactions
Below are real sample interactions captured from this project environment.

### Example 1: Retrieval Endpoint Interaction
Input:

```text
Where is Paris?
```

RAG output:

```json
{"augmented_prompt":"Context:\n\nQuery: Where is Paris?","sources_count":0,"scores":[]}
```

### Example 2: Retrieval Endpoint Interaction
Input:

```text
What are the main features of the system?
```

RAG output:

```json
{"augmented_prompt":"Context:\n\nQuery: What are the main features of the system?","sources_count":0,"scores":[]}
```

### Example 3: End-to-End Pipeline (Generator + Verifier + Self-Correction)
Input:

```text
Where is Paris?
```

Observed runtime behavior:
- Retrieval returned 0 sources.
- Initial generation was flagged as `HALLUCINATION`.
- Self-correction was triggered.
- Post-correction verdict became `GROUNDED`.

Final AI answer:

```text
I apologize, but the provided Context does not contain any information about the location of Paris. Therefore, I cannot answer your query using only the information given.
```

This is the expected safe behavior when the retriever has no supporting documents.

## Design Decisions
1. **Service separation (`RAGClient`, `LLMService`, orchestrator):**
	- Why: easier debugging, cleaner ownership, easier replacement of components.
	- Trade-off: more moving parts and integration points.

2. **Same-model verification loop:**
	- Why: verification prompt stays aligned with generation style and API.
	- Trade-off: verifier may share some model biases with generator.

3. **One-pass self-correction only:**
	- Why: limits latency and avoids infinite refinement loops.
	- Trade-off: one retry may be insufficient in edge cases.

4. **Strict grounded-only correction prompt:**
	- Why: prioritizes trust and factual discipline over verbosity.
	- Trade-off: answers may become conservative or abstain when context is sparse.

5. **Detailed operational logging:**
	- Why: improves observability for retrieval quality and failure analysis.
	- Trade-off: log noise can increase without proper log-level tuning.

## Testing Summary
Reliability checks implemented:
- Automated tests: 6 unit tests for `RAGClient` payload schema, input validation, timeout handling, connection handling, and HTTP error transparency.
- Logging and error handling: the pipeline logs retrieval counts, vector scores, verifier verdicts, and detailed server error responses.
- Human evaluation: manual review of end-to-end output confirmed the system abstains safely when context is missing.

Measured results:
- 6 out of 6 automated tests passed in 0.005 seconds.
- End-to-end run with query "Where is Paris?" produced a safe abstention after verifier-triggered self-correction.
- Reliability gap observed: retrieval returned 0 sources in current environment, so answer quality depends on adding documents to the vector store.

Concise summary:
6 out of 6 tests passed; the AI struggled when retrieval context was missing. Logging and verifier-driven self-correction improved reliability by converting an initially ungrounded response into a grounded abstention.

## Reflection
### Limitations and Biases
- The system is only as strong as its retrieved context; if retrieval is empty or low quality, answers become conservative and less useful.
- Using a single model family for both generation and verification can reinforce shared blind spots.
- The current pipeline does not include a numeric confidence score, so reliability is inferred from verifier outcomes and logs.

### Misuse Risks and Mitigation
- Potential misuse: users may treat generated answers as authoritative even when context quality is poor.
- Mitigations implemented: explicit grounding verification, hallucination detection, and forced self-correction to context-only answers.
- Additional guardrails to add: minimum-source thresholds, confidence scoring, and refusal when retrieval confidence is below a set cutoff.

### What Surprised Me During Reliability Testing
The most surprising result was that reliability failures were mostly integration issues (request schema mismatch and model deprecation), not model reasoning failures. Once contracts and error messages were tightened, debugging became much faster and system behavior became predictable.

### Collaboration With AI During This Project
- Helpful AI suggestion: the AI identified the exact cause of the HTTP 422 error by surfacing the server-side validation message and recommending `query_text` instead of `query`, which resolved retrieval.
- Flawed AI suggestion: an earlier suggestion to change directly to a newer model variant did not account for billing/quota constraints, which caused a separate API failure. The corrected approach was to validate model availability and project spend limits alongside model updates.

