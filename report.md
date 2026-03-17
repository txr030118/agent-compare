# Agent Tooling for Data Science: A Comparative Benchmarking Study

**Module:** MSIN0097 Predictive Analytics | **Word Count:** ~2000

---

## 1. Literature Review

Agentic AI refers to systems in which large language models (LLMs) can not only generate text but also plan actions and interact with external environments to complete tasks. Traditional LLM usage typically produces a single response to a user prompt, whereas agentic systems enable iterative reasoning and action. For example, the ReAct framework integrates reasoning and acting by allowing language models to generate intermediate reasoning steps and perform actions such as retrieving information or interacting with tools (Yao et al., 2023). This reasoning–action loop enables models to adapt their problem-solving strategies dynamically based on environmental feedback.

Another important capability of agentic AI is the ability to use external tools. Toolformer demonstrates that language models can learn to call APIs such as search engines, calculators, and translation systems during text generation (Schick et al., 2023). In addition, multi-agent frameworks such as AutoGen extend this concept by enabling multiple LLM-based agents to collaborate and coordinate to solve complex tasks, supporting workflows such as coding, debugging, and data analysis (Wu et al., 2023). Building on these capabilities, recent research has proposed several workflow patterns that structure how agentic systems perform complex tasks.

Verification-based workflows aim to improve the reliability of AI-generated code through iterative testing and feedback. AlphaCodium proposes a code-oriented flow in which generated code is repeatedly executed against input–output tests and refined based on the results. Through these iterative run–fix cycles, models can progressively detect errors and improve generated solutions before producing the final output (Ridnik et al., 2024). A related pattern is retrieval-augmented generation (RAG): RepoCoder retrieves relevant code snippets from a repository and incorporates them into the generation process, allowing models to access information from multiple files within a codebase, enabling more context-aware code completion (Zhang et al., 2023).

A further theme concerns how agentic AI systems are evaluated in software development environments. Widely used benchmarks such as HumanEval and MBPP are commonly adopted to evaluate code generation capabilities in controlled environments, providing standardised programming tasks for comparing different models (Jin et al., 2024). However, researchers increasingly argue that these benchmarks may not fully capture the complexity of real software engineering tasks. SWE-bench evaluates models using real GitHub issues, assessing whether language models can resolve practical software engineering issues rather than simply generating isolated code snippets (Jimenez et al., 2024). Several studies emphasise that evaluation of agentic systems should consider broader development workflows, including multi-step reasoning, tool interaction, and collaboration between multiple agents (He et al., 2025).

Despite these advances, several challenges remain. One major issue concerns hallucinations in code generation, where models generate code that appears plausible but does not satisfy the intended task requirements — involving syntactic violations, incorrect API usage, and logical errors that affect programme functionality (Lee et al., 2024). Moreover, analyses of SWE-bench datasets reveal issues such as solution leakage and incomplete test coverage, which may lead to overestimation of model capabilities (Aleithan et al., 2024). A particularly critical concern for data science applications is **data leakage** — the inadvertent inclusion of post-outcome information in features — which agents must detect and handle correctly to produce valid predictive models. These findings collectively highlight the need for more reliable evaluation frameworks grounded in realistic, multi-step tasks.

---

## 2. Practical Exploration and Benchmarking

### 2.1 Experimental Design

Three commercially available agentic coding tools were benchmarked: **Claude Code** (Anthropic), **Codex** (OpenAI), and **Cursor** (Anysphere). Each agent was issued identical system prompts requiring a six-step workflow (Plan → Risk Identification → Execution → Verification → Revision → Reproducibility) and completed four progressively complex data science tasks using the hotel bookings dataset (Antonio et al., 2019). This dataset contains 119,390 booking records from two Portuguese hotels (resort hotel H1: 40,060 observations; city hotel H2: 79,330 observations), with 31 variables and a binary cancellation target (`IsCanceled`). The dataset presents realistic challenges including class imbalance (~37% cancellation rate), missing values in three columns (`Children`, `Agent`, `Company`), and several post-outcome leakage variables (`ReservationStatus`, `ReservationStatusDate`). The same prompt was issued to all three agents; outputs were captured as Jupyter notebooks and evaluated post-hoc. Success criteria were defined per task before inspection of any agent's output.

### 2.2 Task 1: Dataset Ingestion, Schema Checks, and Preprocessing Design

**Task specification:** Read the dataset documentation, interpret variable meanings, quantify missingness, flag leakage or methodological risks, and produce a justified preprocessing plan — without performing any modelling. Success criteria: code runs without error; missingness correctly identified; variable interpretation consistent with documentation; leakage variables discussed; preprocessing plan logically structured.

**Claude Code** began by loading and parsing the documentation PDF before touching the data. It flagged `ReservationStatus` and `ReservationStatusDate` as direct post-outcome leakage variables, and additionally identified `DepositType`, `AssignedRoomType`, and `BookingChanges` as temporally ambiguous. It validated the `DepositType` leakage hypothesis empirically via cross-tabulation, proposed that all preprocessing transformers must be fitted on training data only, and produced a structured preprocessing plan with per-variable justifications. Three missing-value columns were identified (`Children`: 4 NA; `Agent`: 16,340 NULL; `Company`: 112,593 NULL), with NULL in `Agent`/`Company` correctly interpreted as "not applicable" per the documentation rather than as missing data.

**Codex** also read the documentation prior to analysis and produced an extensive variable-by-variable commentary covering temporal availability, leakage risk, and data collection context. It flagged `ReservationStatus` and `ReservationStatusDate` as hard exclusions and noted `DepositType` and `BookingChanges` as potentially unsafe. The notebook contained detailed written summaries and a structured preprocessing plan. One reproducibility issue was observed: a hardcoded absolute file path in the data loading cell would cause the notebook to fail on any machine other than the one on which it was originally run.

**Cursor** proceeded directly to loading the dataset without first reading the documentation. Its variable interpretations were therefore derived entirely from observed data patterns rather than the authoritative source. It correctly identified `ReservationStatus` and `ReservationStatusDate` as leakage variables but did not flag `DepositType`. Its NULL-handling recommendation for `Agent` and `Company` treated NULL as a missing value and suggested imputation — inconsistent with the documentation's definition. All three agents satisfied the core success criteria (code ran without error, missingness was identified, and leakage variables were discussed).

### 2.3 Task 2: Exploratory Data Analysis and Insight Generation

**Task specification:** Conduct EDA focused on booking cancellation; identify variables associated with cancellation; examine class imbalance; explore non-linear relationships; produce labelled plots with interpretation; distinguish observed patterns from speculation; flag leakage-sensitive findings. Success criteria: notebook runs without error; plausible cancellation patterns identified; non-linearity explored; class imbalance identified; no predictive model trained.

**Claude Code** completed nine analytical steps covering cancellation rate by hotel type, `LeadTime` non-linearity (using KDE and quantile plots), `DepositType` stratification with an explicit leakage flag, Pearson and Spearman correlation matrices, and Simpson's paradox validation split by hotel. A four-tier evidence labelling system ([OBS]/[EXPL]/[SPEC]/[LEAK]) was applied throughout. One bug was detected during review: the `has_agent` and `has_company` binary features used a string comparison (`!= 'NULL'`) on `float64` columns, causing NaN values to be incorrectly assigned — affecting 16,340 rows. The fix (replacing the comparison with `notna()`) was applied and the correlation matrix was recomputed.

**Codex** produced five modular analysis blocks, each with a corresponding code function and structured written interpretation. It used Spearman rank correlations with a minimum-sample filter for categorical variables, quantile-based non-linearity curves for continuous predictors, and programmatically excluded leakage variables with assertion checks confirming their absence. A "misleading findings audit table" was included, explicitly cataloguing variables whose observed correlation with cancellation likely reflects post-outcome information rather than genuine predictive signal.

**Cursor** produced markdown descriptions of three planned plots (cancellation rate by lead time, by market segment, and by deposit type) but the corresponding code cells were not executed. One visualisation was generated (class balance bar chart). The written analysis described patterns without the supporting code output, leaving the core deliverable partially incomplete. Class imbalance was correctly quantified (63% non-cancelled, 37% cancelled).

### 2.4 Task 3: Baseline Model Training and Evaluation Harness

**Task specification:** Build a baseline pipeline to predict `IsCanceled`; split data before any preprocessing; justify feature selection and model choice; report appropriate metrics; explain weaknesses; ensure reproducibility. Success criteria: code runs without error; valid split and preprocessing order; leakage explicitly considered; metric choice justified; no extensive tuning.

**Claude Code** used a stratified 80/20 random train-test split, preserving the class ratio in both partitions. It excluded `ReservationStatus`, `ReservationStatusDate`, `DepositType`, `AssignedRoomType`, and `BookingChanges` on leakage and temporal grounds, and fitted Logistic Regression as the baseline. A `DummyClassifier` was included as a lower bound. 5-fold cross-validation was run on the training set prior to test evaluation, and a train-test performance gap check was performed. Final test-set ROC-AUC: **0.8293**.

**Codex** used a chronological split — training on bookings with arrival dates before 2017-01-01, testing on 2017 onwards — reflecting how a model would be trained and deployed over time. It excluded only `ReservationStatus` and `ReservationStatusDate`, retaining `DepositType`, `AssignedRoomType`, and `BookingChanges`. Logistic Regression was selected as the baseline with the broadest reported metric set: ROC-AUC, PR-AUC, balanced accuracy, log loss, and Brier score. Final test-set ROC-AUC: **0.8794**. No dummy comparator was included.

**Cursor** applied a random 80/20 split but first removed 1,175 duplicate rows and 180 bookings with zero adults. Logistic Regression was fitted on this reduced population, with the same leakage exclusions applied as in the other agents. No cross-validation, dummy baseline, or overfitting diagnostic was included. Final test-set ROC-AUC: **0.8668**. All three agents satisfied the minimum success criteria (runnable code, train-before-test split, labelled metrics); methodological differences were documented as evidence for the comparative phase.

### 2.5 Task 4: Structured Model Improvement, Candidate Comparison, and Focused Tuning

**Task specification:** Identify baseline weaknesses; compare 3–4 candidate improvement routes using training data only; select the best candidate; perform focused tuning; evaluate the final model once on the held-out test set; compare against the Task 3 baseline. Success criteria: candidates compared fairly under the same conditions; model selection separated from test evaluation; leakage and overfitting risks explicitly checked; any improvement methodologically credible.

**Claude Code** proposed four candidates: (A) Logistic Regression with full feature set, (B) Logistic Regression with conservative feature scope (removing horizon-sensitive variables), (C) Random Forest with class balancing, and (D) Logistic Regression with seven engineered features (log-transformed `LeadTime`, total guests, total nights, and interaction terms). 5-fold stratified cross-validation was used for candidate selection, with a noise threshold of 2× CV standard deviation (~0.008) as a credibility check. Histogram Gradient Boosting was subsequently selected and tuned via `RandomizedSearchCV` (20 iterations). Three evaluation plots were produced: ROC curve comparison, confusion matrix, and feature importance. Final test-set ROC-AUC: **0.9092** (+0.0799 vs. baseline).

**Codex** proposed four candidates (Logistic Regression with full feature set, Logistic Regression with conservative scope, Random Forest with class balancing, and Histogram Gradient Boosting) compared under `TimeSeriesSplit` cross-validation to respect temporal structure. An overfitting gap guardrail (≤0.08) was computed for each candidate. No candidate cleared the guardrail; Codex explicitly reported "Methodological trust signal: False" and proceeded with a note that the improvement was not fully trustworthy under these criteria. Focused tuning was applied to the HGB candidate via `GridSearchCV` (8 configurations). Final test-set ROC-AUC: **0.8871** (+0.0077 vs. baseline).

**Cursor** compared three candidates — Logistic Regression (reference), class-balanced Logistic Regression, and Random Forest — using 3-fold cross-validation. Random Forest was selected (CV ROC-AUC 0.8662 ± 0.0036) and tuned via grid search. The tuned hyperparameters included `max_depth=None` and `min_samples_leaf=1`, which impose no regularisation constraints. No overfitting gap was computed and no credibility threshold was applied. The entire Task 3 code block was duplicated at the end of the notebook, creating a variable state-overwrite risk on top-to-bottom execution. Final test-set ROC-AUC: **0.9099** (+0.0431 vs. baseline).

---

## 3. Comparative Analysis of Agent Tooling

**Table 1. Agent Performance Across Tasks and Evaluation Dimensions**

| Task | Claude Code | Codex | Cursor |
|------|-------------|-------|--------|
| Task 1: Ingestion & Schema | Strongest: empirical leakage validation, cleanest process, highest reproducibility | Deep analysis; hardcoded path reduces reproducibility | Weakest: did not read documentation; NULL-handling errors |
| Task 2: EDA | Strong; one missingness-derived feature bug required correction | Strongest: Spearman + assertions + misleading-finding audit | Weakest: three plots described but never generated |
| Task 3: Baseline Model | Strongest: dummy comparator, CV, leakage control; ROC-AUC 0.8293 | Strong temporal realism; richest metrics; ROC-AUC 0.8794 | Runnable but weakest benchmark discipline; altered population; ROC-AUC 0.8668 |
| Task 4: Model Improvement | Richest experimentation; strongest presentation; ROC-AUC 0.9092 | Strongest methodological self-audit; temporal CV; ROC-AUC 0.8871 | Weakest: no gradient boosting, unbounded RF tuning, code duplication; ROC-AUC 0.9099 |

| Dimension | Claude Code | Codex | Cursor |
|-----------|-------------|-------|--------|
| Correctness | Most consistently complete and spec-aligned | Strong completion, especially Tasks 2 and 4 | Usually runnable, but outputs incomplete or misaligned |
| Statistical Validity | Strongest benchmark discipline and leakage control | Stronger temporal realism and explicit verification | Thinner validation; weaker leakage discipline |
| Reproducibility | Strongest overall; clearest notebook structure | Generally strong; Task 1 path issue reduced rerun reliability | Moderate; notebook state dependencies and population alterations |
| Code Quality | Most polished; easiest to audit | Mature, modular, compact | Readable but thin analysis harness |
| Efficiency | More extensive; occasionally longer than necessary | Most compact high-value workflow | Fast to workable output but methodologically under-developed |
| Safety / Compliance | Very safe; strong leakage control and explicit limitations | Strongest self-audit and credibility discipline | Weakest methodological caution; retained questionable variables |

Leakage handling consistently separated high-performing agents from weaker ones. Claude Code applied the most conservative and validated approach, combining documentation-grounded reasoning with empirical checks. Codex identified major leakage risks but retained several time-sensitive features, while Cursor failed to systematically detect or mitigate leakage. This demonstrates that identification alone is insufficient — strict enforcement and empirical validation are required.

A key trade-off emerged between **benchmark purity** (Claude Code) and **deployment realism** (Codex). Claude Code's random split and conservative feature exclusion produced a cleaner benchmark, but Codex's chronological split better reflects how a model would actually be trained and deployed over time. Both are defensible, and neither approach is universally superior; the choice should be driven by the use case and explicitly justified.

---

## 4. Reflection and Conclusion

### 4.1 Key Findings

Three patterns emerged consistently across all tasks and all agents. First, **documentation-grounded interpretation** was a strong predictor of downstream quality: agents that read and applied the dataset description before writing code made fewer variable misinterpretations and produced more valid preprocessing plans. Second, **leakage discipline** was the single most differentiating capability: correctly identifying *and* enforcing the exclusion of post-outcome variables separated methodologically sound pipelines from misleading ones. Third, **verification steps** — dummy baselines, cross-validation, overfitting gap checks, and credibility thresholds — separated benchmark-grade outputs from merely runnable ones.

Across all tasks, Claude Code was strongest on reproducibility, leakage control, and structural clarity. Codex excelled on temporal realism and methodological self-audit. Cursor was fastest to a runnable output but required the most human correction.

### 4.2 Practitioner Playbook

**Recommended workflow patterns:**
1. Always start with documentation, not data. Read variable descriptions before writing any code.
2. Fit all preprocessing transformers on training data only; use `Pipeline` objects to enforce this.
3. Set a random seed at the top of every notebook and fix it for all downstream operations.
4. Include a `DummyClassifier` in every evaluation to provide a trivial baseline.
5. Compute an explicit train-test performance gap after every model to screen for overfitting.

**Verification checklist before accepting any agent output:**
- [ ] Are post-outcome variables excluded from features?
- [ ] Was the train-test split performed before any preprocessing?
- [ ] Is the evaluation metric appropriate for the class distribution?
- [ ] Can the notebook be re-executed from top to bottom without errors?
- [ ] Is any claimed improvement supported by held-out test performance, not cross-validation score alone?

**Common failure modes:**
- *Hallucinated plots*: Agent describes a visualisation in markdown but does not generate it (Cursor, Task 2).
- *Hardcoded paths*: Absolute file paths break reproducibility on different machines (Codex, Task 1).
- *Silent leakage*: Retaining `DepositType` or `BookingChanges` without temporal audit inflates apparent model performance.
- *Unbounded tuning*: Selecting `max_depth=None` without a regularisation check produces models that cannot be trusted to generalise.
- *State-overwrite*: Duplicating earlier task code at the bottom of a notebook causes variable overwriting on full re-execution.

**When not to use agent tooling:**
Agent tools are most valuable for boilerplate-heavy, well-specified tasks (data loading, schema inspection, standard pipeline construction). They are less reliable — and require more human oversight — for tasks requiring domain-specific judgment (determining which variables are truly leakage-safe for a given deployment setting), tasks involving novel dataset quirks not covered by documentation, and any task where a credibility threshold needs to be set before analysis (not derived from it). Human review of leakage flags, feature scope decisions, and claimed improvements is essential before any agent-generated pipeline is used in production.

---

## Bibliography

Aleithan, R., Alwatban, H., Almelhem, M., AlZubaidi, H., Almutlaq, S., Almulla, M. and Mahmoud, M. (2024) *SWE-bench+: Enhanced coding benchmark for LLMs*. arXiv. Available at: https://doi.org/10.48550/arXiv.2410.06992

Antonio, N., de Almeida, A. and Nunes, L. (2019) 'Hotel booking demand datasets', *Data in Brief*, 22, pp. 41–49. https://doi.org/10.1016/j.dib.2018.11.126

He, J., Treude, C. and Lo, D. (2025) 'LLM-based multi-agent systems for software engineering: literature review, vision and the road ahead', *ACM Transactions on Software Engineering and Methodology*, forthcoming. https://doi.org/10.1145/3712003

Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O. and Narasimhan, K. (2024) 'SWE-bench: Can language models resolve real-world GitHub issues?', *Proceedings of the International Conference on Learning Representations (ICLR 2024)*. Available at: https://doi.org/10.48550/arXiv.2310.06770

Jin, M., Liu, S., Ma, R. and Liu, Y. (2024) 'From LLMs to LLM-based agents for software engineering: A survey of current, challenges and future', arXiv. Available at: https://doi.org/10.48550/arXiv.2408.02479

Lee, J.-H., Kim, J., Kim, S.-W. and Cho, H.-G. (2024) 'A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions', *ACM Transactions on Information Systems*. https://doi.org/10.1145/3703155

Ridnik, T., Kredo, D. and Friedman, I. (2024) 'Code generation with AlphaCodium: From prompt engineering to flow engineering', arXiv. Available at: https://doi.org/10.48550/arXiv.2401.08500

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N. and Scialom, T. (2023) 'Toolformer: Language models can teach themselves to use tools', *Advances in Neural Information Processing Systems*, 36. Available at: https://doi.org/10.48550/arXiv.2302.04761

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X. and Wang, C. (2023) 'AutoGen: Enabling next-gen LLM applications via multi-agent conversation', arXiv. Available at: https://doi.org/10.48550/arXiv.2308.08155

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2023) 'ReAct: Synergizing reasoning and acting in language models', *Proceedings of the International Conference on Learning Representations (ICLR 2023)*. Available at: https://doi.org/10.48550/arXiv.2210.03629

Zhang, F., Chen, B., Zhang, Y., Keung, J., Liu, J., Zan, D. and Chen, W. (2023) 'RepoCoder: Repository-level code completion through iterative retrieval and generation', *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 2471–2484. https://doi.org/10.18653/v1/2023.emnlp-main.151
