# TechPulse RAG Assistant — Short Report

## 1. Business use case
TechPulse Skill Academy is a fictional Karachi training institute. The assistant is intended for prospective students, enrolled learners, instructors, and front-desk staff. It answers public questions about courses, registration, office hours, attendance, refunds, certificates, and procedures. It does not access private student records or approve requests.

## 2. RAG pipeline
Five Markdown documents are loaded from `documents/`. LangChain splits them into chunks. HuggingFace `sentence-transformers/all-MiniLM-L6-v2` creates local embeddings. Chroma stores the vectors in memory for the session. A retriever selects relevant chunks and the Groq chat model generates an answer using the retrieved evidence. Streamlit displays the answer, source filename, policy metadata, and expandable evidence.

## 3. Trust and policy handling
Each chunk retains its source filename. Current policy chunks are labelled version 2.0/current; archived policy chunks are labelled version 1.0/archived. Historical questions are routed to the archived policy; current-rule questions use the current policy. The prompt treats retrieved text as untrusted reference material and instructs the model to ignore embedded commands. This is a mitigation, not a guarantee; prompt-injection tests must be run and recorded.

## 4. Retrieval experiment
Run the six fixed questions in the Retrieval investigation tab with k=1, k=3, and k=5 while keeping documents and chunk settings unchanged. Record whether required evidence was retrieved, whether irrelevant evidence appeared, and whether the final answer was correct. Then change chunk size or overlap and repeat three questions. Insert actual observations here after running the experiment; do not invent results.

## 5. Failures and limitations
Record actual failures observed during testing, such as missed evidence, irrelevant chunks, provider/API errors, or policy-selection errors. The application is a classroom prototype, not a production system. It has no authentication, student-record integration, human approval workflow, or guaranteed answer correctness. Important decisions must be verified by administration.

## 6. Test results
Complete `tests/test_questions.csv` with actual answers, actual source documents, pass/fail, and a short reason for all 20 tests. Summarize the measured pass count and major failure patterns here after testing.
