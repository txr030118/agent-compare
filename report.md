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

Three commercially available agentic coding tools were compared: **Claude Code** (Anthropic), **Codex** (OpenAI), and **Cursor** (Anysphere). Each agent was issued identical system prompts requiring a six-step workflow (Plan → Risk Identification → Execution → Verification → Revision → Reproducibility) and evaluated across four tasks drawn from the hotel bookings dataset (Antonio et al., 2019). This dataset contains 119,390 booking records from two Portuguese hotels (resort hotel H1: 40,060 observations; city hotel H2: 79,330 observations), with 31 variables and a binary cancellation target (`IsCanceled`). The dataset presents realistic challenges including class imbalance (~37% cancellation rate), missing values in three columns (`Children`, `Agent`, `Company`), and several post-outcome leakage variables (`ReservationStatus`, `ReservationStatusDate`). All agents ran under the same prompt framework to ensure a consistent evaluation setting.

### 2.2 Task 1: Dataset Ingestion, Schema Checks, and Preprocessing Design

All three agents were required to read the dataset documentation before loading data, interpret variable meanings, identify missingness, flag leakage risks, and produce a preprocessing plan — without performing any modelling.

All agents correctly identified `ReservationStatus` and `ReservationStatusDate` as high-risk leakage variables, as these are post-outcome fields derived from the booking outcome itself. Differences emerged on subtler variables: Codex and Claude Code additionally flagged `DepositType` as a potential leakage risk because payment classification occurs relative to the booking's arrival or cancellation date; Cursor did not flag this variable, indicating weaker documentation-grounded interpretation. Claude Code was the only agent to validate its leakage hypothesis empirically using cross-tabulation, and the only one to propose that preprocessing transformers (imputers, encoders, scalers) must be fitted exclusively on training data. Codex produced the deepest analytical narrative but hard-coded an absolute file path, directly undermining reproducibility. Cursor did not read the documentation prior to analysis, leading to several NULL-handling suggestions that were factually inconsistent with the dataset's schema — where NULL in `Agent` means "no travel agent involved," not a missing value.

### 2.3 Task 2: Exploratory Data Analysis and Insight Generation

Task 2 required EDA focused on cancellation prediction: identifying key correlates, examining class imbalance, exploring non-linear relationships, and explicitly flagging leakage-sensitive patterns.

Codex delivered the strongest Task 2 output, using Spearman rank correlations, minimum-sample filtering, and quantile-based non-linearity curves. It programmatically excluded leakage variables and verified their exclusion with assertions, and produced a unique "misleading findings audit table" flagging variables whose apparent predictive value stems from post-outcome information. Claude Code also performed strongly, providing a nine-step analysis with Pearson + Spearman dual correlation, Simpson's paradox verification split by hotel type, and a four-tier evidence labelling system ([OBS]/[EXPL]/[SPEC]/[LEAK]). However, a coding bug in the `has_agent`/`has_company` binary feature encoding — where a NaN-safe comparison was not used for `float64` columns — caused 16,340 rows to be misclassified. This was detected during manual review and corrected by replacing the string comparison with a `notna()` check. Cursor produced the weakest EDA: three plot descriptions were written in markdown but the corresponding code cells were never executed, leaving the core deliverable incomplete.

### 2.4 Task 3: Baseline Model Training and Evaluation Harness

Task 3 assessed whether agents could build a statistically valid, simple baseline pipeline. Each agent was required to define the prediction target, determine feature scope, split data before any preprocessing, select an interpretable baseline model, report suitable metrics, and explain weaknesses.

Claude Code produced the most rigorous baseline: a stratified 80/20 train-test split, a DummyClassifier benchmark, 5-fold cross-validation, and an explicit train-test performance gap check. It excluded `ReservationStatus`, `ReservationStatusDate`, and additionally `DepositType`, `AssignedRoomType`, and `BookingChanges` on temporal grounds. Logistic Regression achieved ROC-AUC = 0.8293 on the held-out test set. Codex adopted a chronologically grounded split (train: arrivals before 2017; test: 2017 onwards), which has higher deployment realism. It reported the richest metric set (ROC-AUC, PR-AUC, balanced accuracy, log loss, Brier score), reaching ROC-AUC = 0.8794. However, Codex retained several timing-sensitive features and omitted a dummy comparator. Cursor produced a runnable baseline (ROC-AUC = 0.8668) but pre-filtered duplicate rows and bookings with zero adults before modelling, substantially altering the benchmark population and omitting cross-validation and overfitting diagnostics. All three satisfied the defined success criteria (runnable code, correct train-before-test split, appropriate metrics), so their methodological differences were retained as evidence for comparative analysis.

### 2.5 Task 4: Structured Model Improvement, Candidate Comparison, and Focused Tuning

Task 4 required agents to identify baseline weaknesses, compare 3–4 candidate improvement routes using only training data, select the best candidate, perform focused tuning, and evaluate once on the held-out test set.

Codex was the strongest on methodological discipline: it was the only agent to use `TimeSeriesSplit` cross-validation to respect the temporal structure of the data, computed overfitting gap guardrails (≤0.08) for each candidate, and — uniquely — programmatically self-assessed whether any improvement was credible, reporting "Methodological trust signal: False" when no candidate cleared the guardrail. Its final HGB model reached ROC-AUC = 0.8871. Claude Code explored the widest experimental space: four candidates including a feature engineering route (log-transformed `LeadTime` + 6 engineered interaction features), `RandomizedSearchCV` with 20 iterations, and three evaluation plots including ROC comparison and feature importance. The noise threshold was set at 2× CV standard deviation ≈ 0.008, and the final ROC-AUC reached 0.9092. Cursor compared only three routes (missing gradient boosting), tuned a Random Forest to `max_depth=None`/`min_samples_leaf=1` without recognising the overfitting risk, and reproduced the entire Task 3 code block at the end of the notebook, creating state-overwrite risk during top-to-bottom execution.

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
