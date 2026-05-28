# 1. Dataset Description

Reasoning With A Star (RWS) is a benchmark dataset of 158 question-steps-and-answer records drawn from the NASA and UCAR Living With a Star (LWS) summer school problem sets. It provides a curated set of heliophysics scientific reasoning questions, contexts, solution steps, and graded answers for benchmarking LLM and agentic reasoning systems on physical science tasks.

**A detailed description may be found in the project [Technical Memorandum](https://helioai.org/artifact/de35b1ad-a55b-47f7-b98e-720d6f3c3083/details) and related dataset [publication](https://doi.org/10.48550/arXiv.2511.20694).**

**The dataset itself may be found on [HuggingFace](https://huggingface.co/datasets/SpaceML/ReasoningWithAStar).**

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-orchestrator/Heliolab Badge_RWS.png?raw=true" width="400">
</p>


## 1.1 Raw Data

The raw data from which this dataset was constructed is graduate-level problem sets from the [NASA/UCAR Living With a Star (LWS) Summer School](https://heliophysics.ucar.edu/summer-school) (available at [links such as this](https://heliophysics.ucar.edu/resources-problem-sets-1)).


## 1.2 Processed Data

The dataset was built by taking the original LWS problem sets and normalizing them into a uniform JSONL schema suitable for LLM evaluation — preserving the pedagogical step-by-step solutions while adding machine-checkable fields (answer type, format hints, ground truth). This structure supports a companion programmatic grader that evaluates model responses with unit-aware numerical tolerance, symbolic equivalence checking, and schema validation, so the benchmark tests genuine physical reasoning (correct units, valid derivations, properly formatted outputs) rather than surface string matches.

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-orchestrator/ORCHESTRATOR Technical Showcase - RWS Processing Workflow.png?raw=true" width="400">
</p>

Each of the 158 records captures a full problem: the question with optional preamble context, a hint about the expected answer format, a list of intermediate steps showing the reasoning trace, a final ground-truth answer, an answer type (symbolic LaTeX, numeric, or textual — plus some structured JSON outputs), and meta fields recording the original author, year, and source document.

The dataset is structured as a JSONL (JSON-lines) file. Each record is a single question–and-answer pair with a machine-checkable target (`final`) and a type label that drives automatic grading: `numeric` (numeric value), `symbolic`(symbolic equivalence), or `textual` (textual equivalence). The JSONL file has the following schema:

| Field Name | Type | Description |
|-----------|---------|---------|
| `id` | String | Unique identifier for the QA set (e.g., `2010_Lee_4_a`). |
| `q_id` | Integer | Question identifier for the QA set from the original problem (e.g., 1, 2, 3). |
| `sub_id` | String | Sub-question identifier for the QA set from the original problem (e.g., a, b, c). |
| `preamble` | Array[String] | Optional ordered list of previous sub-QA sets. Each element is a short string (prior question and step/answer). |
| `question` | String | The problem statement; may include inline LaTeX for equations. |
| `hint`| String | Optional hint for the benchmark instruction prompt. |
| `step` | Array[String] | Reasoning steps to solve the problem. |
| `final` | String | Ground-truth target answer for grading. |
| `type` | String | Expected output type for grading: `numeric`, `symbolic`, or `text`. | 
| `meta` | Dictionary | Metadata (e.g., year, author, source).|

Problems are grouped by `id`, `q_id`, and `sub_id`, so multi-part questions (a, b, c...) stay linked as progressive derivations. For example, `id` would refer to the question source (which worksheet), `q_id` would correspond to e.g. "question 1" and `sub_id` to "question 1 part a". 

The scientific content spans the core LWS curriculum: solar wind and cosmic-ray transport (Parker equation, termination shock, stochastic acceleration), heliospheric and magnetospheric dynamics, and ionospheric E/F-region physics. The content is entirely drawn from peer-taught graduate level coursework.

Full details can be found in [the paper](https://doi.org/10.48550/arXiv.2511.20694).

# 2. Access Instructions

The dataset itself may be found on [HuggingFace](https://huggingface.co/datasets/SpaceML/ReasoningWithAStar) and can be downloaded directly from there.
