# Evidence-Verified Agentic RAG for Financial Research

**Team Members:**
Jordan Phillips

**Selected Track:**
Option 2 — LLMs + AI Agent System (Evaluation-First)

## Abstract

Financial research requires analysts to synthesize information from lengthy financial filings, structured financial statements, and numerical data. Conventional large language model systems can retrieve relevant information and generate fluent answers, but they remain vulnerable to hallucinations, incorrect numerical reasoning, incomplete evidence, and unsupported conclusions.

This project proposes an evidence-verified agentic Retrieval-Augmented Generation (RAG) system for financial research. Given a complex financial question, the system will autonomously plan an investigation, retrieve evidence from SEC filings and structured financial data, use external tools for numerical calculations and database queries, and generate an evidence-backed research report. The proposed architecture uses a Planner–Executor–Critic workflow in which the planner decomposes the research question, the executor performs retrieval and tool calls, and the critic verifies the resulting claims and citations.

The system will incorporate hybrid retrieval combining semantic and keyword search, structured financial data retrieval, persistent research memory, and citation verification. We will evaluate at least three LLMs, including an open-source model, using a common evaluation set containing at least 50 financial research questions. Evaluation will measure answer correctness, numerical accuracy, retrieval quality, citation grounding, unsupported-claim rate, latency, and cost.

The primary research question is whether explicit planning, tool use, persistent research memory, and critic-based evidence verification can improve the reliability of financial research agents compared with conventional RAG systems while maintaining reasonable computational cost and latency.
