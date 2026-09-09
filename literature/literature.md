# Literature & SOTA Survey Results

## 1. MultiFinRAG — 2025

**Paper:** *MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation Framework for Financial Question Answering*

MultiFinRAG proposes a multimodal RAG framework for financial question answering that can retrieve and reason over different types of information in financial documents, including text, tables, and figures. The work addresses the difficulty of answering financial questions when relevant information is distributed across multiple document formats. This work motivates my use of multiple information sources rather than relying solely on text-based vector retrieval. My project extends this idea toward an agentic research workflow that can retrieve evidence, perform calculations, use external tools, and verify its conclusions.

**Source:** https://arxiv.org/abs/2506.20821

---

## 2. HiREC — ACL 2025

**Paper:** *Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents*

HiREC proposes a hierarchical retrieval and evidence-curation approach for financial question answering over standardized documents such as SEC filings. The authors address the difficulty of retrieving useful evidence from long financial documents containing repetitive and highly similar information. The work also introduces the LOFin benchmark containing 145,897 SEC documents and 1,595 question-answer pairs. HiREC demonstrates that conventional retrieval can be insufficient for financial documents and motivates my use of more sophisticated retrieval. I plan to incorporate hybrid retrieval and evidence verification while additionally allowing an agent to decompose complex research questions and use financial tools.

**Source:** https://aclanthology.org/2025.findings-acl.855/

---

## 3. Bayesian RAG — 2025

**Paper:** *Bayesian RAG: Uncertainty-Aware Retrieval for Reliable Financial Question Answering*

Bayesian RAG investigates how uncertainty-aware retrieval can improve the reliability of RAG systems for financial question answering. The work evaluates financial QA using SEC 10-K filings and focuses on distinguishing reliable retrieved evidence from uncertain or less relevant information. Reliability and uncertainty are central to my proposed system. I similarly want the agent to recognize when evidence is insufficient rather than generate unsupported claims. My approach focuses on explicit evidence verification through a Critic stage that checks generated claims and citations.

**Source:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1668172/full

---

## 4. Financial Time-Series RAG — 2025

**Paper:** *Enhancing Financial Time-Series Forecasting with Retrieval-Augmented Large Language Models*

This work investigates the use of retrieval-augmented language models for financial time-series forecasting. Instead of relying only on the model's learned knowledge, relevant historical financial information is retrieved and incorporated into the forecasting process. The authors report improvements compared with baseline retrieval approaches. This work demonstrates the value of combining LLMs with specialized financial information rather than treating financial analysis as a purely language-based task. My project similarly combines unstructured financial documents with structured financial data and numerical tools, but focuses on research and analysis rather than forecasting alone.

**Source:** https://arxiv.org/abs/2502.05878

---

## 5. Multi-Retriever Financial QA — 2025

**Paper:** *Integrating Domain Knowledge for Financial QA: A Multi-Retriever RAG Approach with LLMs*

This work investigates a multi-retriever RAG architecture for financial question answering that incorporates domain knowledge alongside retrieved information. The approach highlights the importance of using multiple sources of relevant information when answering specialized financial questions. The multi-retriever approach supports my decision to combine different retrieval mechanisms and financial data sources. My proposed system builds upon this direction by adding agentic planning, tool use, persistent research memory, and a Critic stage for evidence and citation verification.

**Source:** https://arxiv.org/abs/2512.23848
