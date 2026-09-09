Yes—but not quite in its current form. After checking work through September 2026, I think you have a strong graduate project, but the novelty should be reframed around verifiable evidence sufficiency and calibrated abstention, rather than “agentic RAG + critic for finance.”

The reason is that several very recent papers now overlap strongly with your architecture. The closest is FinSAgent, released July 2026, which uses a multi-agent architecture for SEC filings, corpus-aware query decomposition, multi-path retrieval, and explicit evidence grounding. Even closer to your Critic idea, an August 2026 paper, Self-Improving RAG, uses Retrieval, Reasoning, and Judge agents; when the judge rejects an answer, it triggers additional retrieval/reasoning and logs decisions for auditability. A May 2026 paper also explicitly proposes an auditable financial RAG architecture with hybrid retrieval, validation, persistent audit artifacts, and SEC 10-K/10-Q data.

So I would keep the core project, but change what you claim as the contribution.

Where your idea sits relative to recent work

The literature has moved very quickly.

FinanceBench (2023) established that financial QA over SEC filings is difficult even with retrieval. It includes evidence associated with answers and showed substantial failure rates for contemporary RAG systems.

Optimizing LLM-Based RAG Pipelines in Finance (NAACL 2024) systematically varied models, retrieval configurations, and prompts, meaning simply comparing several RAG configurations is no longer novel by itself.

FinDVer (EMNLP 2024) is especially relevant to your Critic. It introduced 4,000 expert-annotated examples specifically for claim verification in long financial documents, finding that LLMs still lag humans significantly.

HiREC (ACL Findings 2025) attacks retrieval errors in SEC filings using hierarchical retrieval and explicit evidence curation.

FinGEAR (EMNLP Findings 2025) uses the structure of 10-K filings, finance terminology, hierarchical indices, and reranking rather than generic vector RAG.

A 2025/2026 study called Rethinking Retrieval compared vector agentic RAG using hybrid retrieval against non-vector hierarchical systems across 1,200 SEC filings and measured answer quality, retrieval accuracy, latency, and cost—almost exactly some of the comparisons in your proposal.

Finance Agent Benchmark (2025) is important because it already contains 537 expert-authored real-world financial research questions involving SEC filings and agent tools. Even the best system in that study achieved only 46.8% accuracy, demonstrating that complicated financial research remains genuinely unsolved.

Then 2026 gets particularly close to your idea. FinMRAGBench/FinMRAGAgent provides 887 expert-verified questions requiring cross-document and multi-page retrieval plus an agent capable of multi-step financial analysis. FinRank, from August 2026, contains 1,185 questions and specifically tests whether answers retrieve evidence from the correct company, reporting period, and disclosure context, including deliberately confusing hard-negative evidence.

That means this:

“I built a Planner–Executor–Critic financial RAG system with hybrid search, tools, and citations.”

is probably not a novel research contribution anymore.

But you have a very promising route around that.

The direction I would take

I'd change the project into something like:

Evidence-Sufficient Agentic RAG: Claim-Level Verification and Selective Abstention for Financial Research

The central research question becomes:

Can explicit claim-level evidence verification and accounting-aware consistency checks reduce unsupported financial conclusions while preserving answer coverage?

That is substantially stronger academically.

Instead of the Critic merely asking:

“Does this answer look supported?”

make the system convert its proposed answer into atomic claims.

For example:

NVIDIA's gross margin increased from X% in 2023 to Y% in 2024 primarily because of a richer mix of data-center products.

becomes:

NVIDIA gross margin was X% in FY2023.
NVIDIA gross margin was Y% in FY2024.
Gross margin increased by Y − X percentage points.
Management attributed the increase partly to product mix.
Data-center products contributed to that product-mix change.
Therefore product mix was a material driver of the margin increase.

Now require the verifier to produce a provenance object:

Claim 1
Evidence: 2024 10-K, page/section ...
Status: SUPPORTED

Claim 2
Evidence: 2024 10-K ...
Status: SUPPORTED

Claim 3
Inputs: Claim 1, Claim 2
Calculation: Y - X
Status: VERIFIED

Claim 4
Evidence: MD&A ...
Status: SUPPORTED

Claim 5
Evidence: ...
Status: PARTIALLY SUPPORTED

Claim 6
Depends on: Claims 4 and 5
Status: INSUFFICIENT EVIDENCE

The final answer can consequently say:

The filings support that gross margin increased and that product mix contributed to the increase. However, the available evidence does not establish that data-center mix was the primary cause.

That behavior is much more interesting than adding another critic agent.

Your actual research contribution

I would make your contribution consist of three things.

1. Claim–Evidence Provenance Graph

Instead of storing research memory as arbitrary text, maintain a structured graph:

Question
   ↓
Subquestions
   ↓
Retrieved Evidence
   ↓
Extracted Facts
   ↓
Calculations
   ↓
Atomic Claims
   ↓
Conclusion

Every conclusion must have a path back to evidence.

For numerical claims:

Claim
 ↓
Calculation
 ↙       ↘
operand A  operand B
   ↓         ↓
source     source

This lets you distinguish:

retrieval hallucination
citation hallucination
arithmetic error
wrong period
wrong unit
unsupported inference
contradictory evidence

That is a much more rigorous definition of evidence verification.

There is broader work on identifying minimal evidence groups necessary to fully support claims, which gives you theoretical grounding for this approach.

2. Accounting-aware verification

This is where you can make the finance aspect matter instead of simply applying generic RAG to finance.

Your verifier should enforce constraints such as:

Revenue - Cost of Revenue = Gross Profit

Gross Margin =
Gross Profit / Revenue

Assets =
Liabilities + Shareholders' Equity

and verify metadata:

company = NVIDIA
period = FY2025
unit = USD millions
filing date = ...
form = 10-K

This catches an extremely important class of RAG errors:

correct number, wrong period/company/unit.

FinRank shows this problem explicitly: plausible evidence from another period or comparable company can fool retrieval systems. Recent XBRL-aware work is also exploring schema grounding and cross-filing temporal/unit consistency, so you should cite it and differentiate your work by applying such constraints specifically to claim-level provenance and selective answering, rather than merely structured retrieval.

You could call this layer:

Financial Consistency Verifier

with rules for:

period consistency
fiscal versus calendar year
units
currency
percentage vs percentage-point changes
accounting identities
sign conventions
YoY calculations
ratios
company/entity consistency

That makes this unmistakably a financial AI research project, not just LangGraph with SEC documents.

3. Evidence sufficiency and calibrated abstention

This is probably your strongest novelty angle.

Don't merely evaluate:

Did it answer?

Evaluate:

Should it have answered?

You deliberately create questions with four evidence conditions:

A. Complete evidence
B. Partial evidence
C. No evidence
D. Conflicting/misleading evidence

Then test whether the system answers or abstains appropriately.

This is a very current open problem. ClaimDB, published at ACL 2026, found that both proprietary and open models still struggle with abstention when evidence is insufficient. An August 2026 controlled RAG study found something even more interesting: models often learn to abstain when evidence is missing, but perform much worse when retrieved evidence is actually misleading.

That gives you a very defensible research question:

Does claim-level verification improve selective answering under missing, conflicting, and misleading financial evidence?

Now your project is studying a failure mode instead of merely constructing a system.

A very good experiment

Your original five configurations are useful, but I would modify them to create an ablation study.

System	Retrieval	Agent	Claim verifier	Financial constraints	Abstention
Direct LLM	—	—	—	—	—
Basic RAG	Dense	—	—	—	—
Hybrid RAG	Dense + BM25	—	—	—	—
Agentic RAG	Hybrid	✓	—	—	—
+ Claim Verification	Hybrid	✓	✓	—	✓
+ Financial Verification	Hybrid	✓	✓	✓	✓

Your last system becomes your proposed method.

That makes the experiment answer:

Which component actually increases reliability?

rather than:

Is my giant system better than the baseline?

That distinction is academically important.

Change the evaluation metrics too

Your current metrics are a good start, but “citation quality” is too vague.

I'd measure:

Answer accuracy

Exact/F1/human rubric depending on question.

Numerical accuracy

Both final result and intermediate operands.

Evidence Recall

Did retrieval recover the gold evidence?

Citation precision

Of cited sources, how many actually support the associated claim?

Claim support rate

$$ CSR = \frac{\#\text{supported generated claims}} {\#\text{generated claims}} $$

Unsupported claim rate

$$ UCR = \frac{\#\text{unsupported claims}} {\#\text{claims}} $$

Abstention precision

When the model abstains, was evidence actually insufficient?

Abstention recall

Of truly unanswerable questions, how many did it refuse?

And especially:

Risk–coverage curve

As you increase your evidence-confidence threshold:

Low threshold → answer more questions / more errors
High threshold → answer fewer questions / fewer errors

That is much more informative than just “accuracy.”

You are effectively asking:

At 90% answer coverage, how trustworthy is the system?

That is directly relevant to high-stakes financial research.

Your 50-question dataset is probably too small

For a course project, 50 may satisfy the requirement.

For a convincing graduate research project, though, 50 hand-written questions isn't a particularly strong benchmark contribution when Finance Agent Benchmark has 537, FinMRAGBench has 887, FinRank has 1,185, FinDVer has 4,000, and FinanceBench's broader collection contains thousands.

Instead of writing hundreds from scratch, I would do:

Existing benchmark + your specialized stress test.

For example:

100 FinanceBench / FinRank questions
+
100 custom evidence-sufficiency variants

Generate controlled variants:

Original:
What was NVIDIA's gross margin in FY2025?

Missing evidence:
Remove relevant table.

Wrong-period evidence:
Provide FY2024 table instead.

Conflicting evidence:
Retrieve FY2024 and FY2025 figures.

Wrong-company evidence:
Inject AMD equivalent passage.

Unsupported causal:
Why did gross margin rise primarily because of X?
when filings only state correlation.

Now your dataset itself contains a real research contribution:

Financial Evidence Sufficiency Benchmark

That is much stronger than “50 financial questions.”

One major issue with your example questions

You mentioned questions like:

What factors could cause gross margin to change in the future?

Be careful.

That is a fundamentally different epistemic task from:

What caused gross margin to change from 2023–2025?

The latter can be grounded in historical disclosures.

The former involves:

management guidance
risk disclosures
hypothetical reasoning
economic inference
forecasts

There may not be one correct answer.

I would explicitly split questions into:

Factual
Numerical
Comparative
Causal attribution
Risk
Forward-looking
Insufficient evidence

And don't evaluate all of them with the same correctness metric.

Another important limitation: the Critic can hallucinate too

Your current architecture implicitly treats:

Planner → Executor → Critic

as though the Critic is authoritative.

It isn't.

If GPT-4.1 Mini generates the answer and GPT-4.1 Mini critiques it, you can get correlated errors.

The generator and critic may share the exact same misconception.

So compare:

same-model verification
vs
cross-model verification
vs
deterministic verification

For example:

Generator: Gemini
Verifier: Qwen

Generator: Gemini
Verifier: Gemini

Generator: Gemini
Verifier:
  deterministic calculator
  + metadata rules
  + entailment model

A potentially interesting result could be:

Deterministic + LLM verification outperforms a purely LLM-based Critic.

That is academically valuable.

Other limitations you'll want to discuss

There are several.

Retrieval and reasoning become confounded. If your final system improves, you need to determine whether it was hybrid retrieval, agent planning, retries, or verification.

LLM-as-a-judge can bias evaluation. A model may favor outputs resembling its own style. Use exact numerical scoring and human annotation where possible.

Cost isn't controlled automatically. Agentic RAG might simply make 10× more model calls. Report accuracy versus tokens, latency, and dollars.

SEC tables are difficult. Flattening tables can destroy row/column relationships; recent work continues to identify table representation and document hierarchy as important sources of failure.

One company is insufficient. NVIDIA is excellent for prototyping, but a final evaluation only on NVIDIA makes it difficult to claim generalization.

I would eventually include roughly:

Technology: NVIDIA
Banking: JPMorgan
Retail: Walmart
Industrial: Caterpillar
Healthcare: Pfizer

because their financial disclosures and accounting structures differ substantially.

Temporal leakage matters. If your LLM already knows NVIDIA's results from pretraining, direct-LLM performance isn't clean evidence of reasoning from your corpus.

You can partially address this with newer filings or controlled evidence perturbations.

What I think your thesis should actually say

Something close to this:

Existing financial RAG systems primarily optimize retrieval and answer accuracy, but a financially plausible answer may still rely on incomplete, temporally incorrect, or otherwise insufficient evidence. This project investigates whether explicit claim-level provenance, accounting-aware verification, and calibrated abstention can improve the reliability of agentic RAG for financial research. The proposed system decomposes generated conclusions into atomic claims and associates each claim with retrieved evidence or executable calculations. A verification layer evaluates evidential support, numerical correctness, entity and reporting-period consistency, and accounting constraints before deciding whether to answer, retrieve additional evidence, qualify a conclusion, or abstain. The system will be evaluated under complete, missing, conflicting, and misleading evidence conditions to measure the tradeoff between answer coverage and unsupported claims.

That is graduate-level.

It has a clearly identifiable research problem.

It has hypotheses.

It has baselines.

It has ablations.

And it doesn't depend on pretending agentic financial RAG itself hasn't already been done.

The papers I would put at the center of your literature review

The most important ones for your project are:

FinanceBench (2023) — establishes financial QA/RAG problem.
FinDVer (EMNLP 2024) — financial claim verification.
HiREC (ACL Findings 2025) — hierarchical SEC retrieval/evidence curation.
Finance Agent Benchmark (2025) — realistic agent-based financial research.
FinMRAGBench/FinMRAGAgent (ACL 2026) — multi-document financial RAG and agents.
FinSAgent (July 2026) — probably your closest architectural competitor.
FinRank (August 2026) — highly relevant for correct-evidence/period attribution.
Self-Improving RAG (August 2026) — probably your closest Critic/retry competitor.
ClaimDB (ACL 2026) — motivates insufficient-evidence abstention.
GRAB-RAG (August 2026) — motivates testing misleading versus merely absent evidence.

If I were presenting this to your professor, FinSAgent, FinRank, Self-Improving RAG, and FinDVer are the four papers I would be most prepared to explain how your project differs from.

Bottom line

Your current project is roughly:

Agentic financial RAG with evidence verification.

That's useful, but by September 2026 it has too much direct precedent to be the core novelty.

Change it to:

Claim-Level, Accounting-Aware Evidence Verification and Selective Abstention for Agentic Financial RAG

and make your main experimental question:

Can a system determine not merely what answer to give, but whether the available financial evidence is sufficient to justify each claim in that answer?

That is a much more defensible and interesting research contribution—and you can still keep almost all of the engineering you already planned.
Would you like to focus the project more on claim-level verification or on calibrated abstention under misleading evidence?
