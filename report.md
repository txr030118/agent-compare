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

Task 3 tested whether the understanding developed in Tasks 1 and 2 could be translated into a statistically valid baseline workflow for predicting hotel booking cancellations. Each agent was required to define the prediction target, determine a leakage-safe feature scope, split the data before any preprocessing, select and justify a simple interpretable baseline model, report appropriate evaluation metrics, explain workflow weaknesses, and ensure full reproducibility. Model complexity was not rewarded; the task was designed to assess methodological soundness rather than optimised performance.

Claude Code produces the most audit-ready workflow. It applies a stratified 80/20 split (train: 95,512 rows; test: 23,878 rows), fits Logistic Regression (`class_weight='balanced'`, solver `lbfgs`) against a `DummyClassifier` lower bound, runs 5-fold stratified CV on the training set (ROC-AUC: 0.8313 ± 0.0039), and computes an explicit train-test gap (ROC-AUC gap: 0.003), confirming minimal overfitting. It also applies the most conservative feature scope, excluding `deposit_type`, `assigned_room_type`, and `booking_changes` on temporal-ambiguity grounds, and recodes `agent`/`company` as binary presence flags rather than dropping them. Test ROC-AUC: 0.8293.

Codex adopts a chronological holdout split aligned with deployment logic (arrivals before 2017 as train: 78,703 rows; 2017 onwards as test: 40,687 rows), reflecting realistic temporal distribution shift (train cancel rate 36.2%; test 38.7%). Logistic Regression (`class_weight='balanced'`, solver `liblinear`) is trained on 29 features, retaining temporally ambiguous variables (`deposit_type`, `booking_changes`, `assigned_room_type`) without exclusion. It reports the broadest metric set of the three agents (ROC-AUC, PR-AUC, Balanced Accuracy, F1, Precision, Recall, Log Loss, Brier Score) and computes an explicit train-test gap (ROC-AUC gap: 0.0504). No dummy baseline or cross-validation is reported for the Task 3 baseline. Test ROC-AUC: 0.8794.

Cursor produces a runnable Logistic Regression baseline on a stratified 70/30 split (train: 60,907 rows; test: 26,104 rows). Critically, it filters the dataset before splitting — removing exact duplicate rows and 180 zero-adult bookings — reducing the benchmark population from 119,390 to approximately 87,011 rows and shifting the observed cancellation rate from ~37% to ~27.5%. No class weighting is applied. A 3-fold stratified CV is run on the training set (ROC-AUC: 0.8629 ± 0.0025), but no dummy baseline and no explicit train-test overfitting gap diagnostic are included. The population alteration is not flagged as a deviation from the original benchmark population. Test ROC-AUC: 0.8668.

Since manual review confirms that all three satisfy the defined success criteria (code runs without error, split precedes preprocessing, labelled metrics reported), these differences are retained as task-level benchmark evidence and examined comparatively in Section 3. The table below records the full observed evidence from each agent's notebook output.

| Aspect | Claude | Codex | Cursor |
|---|---|---|---|
| **Target definition and feature scope** | `is_canceled` as binary target. Excluded: `reservation_status`, `reservation_status_date` (direct leakage); `deposit_type`, `assigned_room_type`, `booking_changes` (temporal ambiguity). `agent`/`company` encoded as binary presence flags; `country` excluded (high cardinality). 26 features total. | `is_canceled` as binary target. Excluded: `reservation_status`, `reservation_status_date` only. 29 features retained, including `deposit_type`, `assigned_room_type`, and `booking_changes`. | `is_canceled` as binary target. Excluded: `reservation_status`, `reservation_status_date`, `adr` (leakage risk). 28 features retained, including `deposit_type`, `assigned_room_type`, and `booking_changes`. Dataset reduced before splitting by dropping 1,175 duplicate rows and 180 zero-adult bookings (effective population: ~87,011 rows). |
| **Split and preprocessing logic** | Stratified random 80/20 split (`random_state=42`), preserving ~37% cancellation rate in both partitions. All preprocessing (median imputation → `StandardScaler` for numeric; constant imputation → `OneHotEncoder` for categorical) inside sklearn `Pipeline` fitted on training data only. | Chronological holdout: arrivals before 2017-01-01 as train (78,703 rows, 36.2% cancel rate); 2017 onwards as test (40,687 rows, 38.7% cancel rate). `Pipeline` fitted on train only: median imputation → `StandardScaler`; mode imputation → `OneHotEncoder`. | Stratified random 70/30 split (`random_state=42`). Train: 60,907 rows; test: 26,104 rows. `Pipeline` fitted on train only: imputation → `StandardScaler`; constant imputation → `OneHotEncoder`. |
| **Baseline model choice** | Logistic Regression (`class_weight='balanced'`, solver=`lbfgs`, C=1.0, max_iter=1000). Justified as simple, interpretable, and imbalance-aware. `DummyClassifier` (most-frequent strategy) included as lower-bound reference. | Logistic Regression (`class_weight='balanced'`, solver=`liblinear`, `random_state=42`). Justified as simple, defensible, and transparent. No dummy baseline included. | Logistic Regression (default settings, no class weighting). Justified as simple baseline; imbalance handling deferred to Task 4. No dummy baseline included. |
| **Metrics reported** | ROC-AUC, Macro F1, F1 (cancel=1), Precision (pos=1), Recall (pos=1), Accuracy. Test ROC-AUC: **0.8293**. | ROC-AUC, PR-AUC, Balanced Accuracy, F1 (macro), Precision, Recall, Accuracy, Log Loss, Brier Score. Test ROC-AUC: **0.8794**. | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion matrix. Test ROC-AUC: **0.8668**. |
| **Verification and methodological checks** | 5-fold stratified CV on training set (CV ROC-AUC: 0.8313 ± 0.0039). Train-test gap computed: ROC-AUC gap = 0.0026, indicating minimal overfitting. Dummy comparator confirms model adds value beyond trivial prediction. | Train-test gap computed: ROC-AUC gap = 0.0504 (train 0.9299, test 0.8794). Gap flagged as diagnostic but not used for selection. No CV reported for the baseline. | 3-fold stratified CV on training set (CV ROC-AUC: 0.8629 ± 0.0025). No dummy baseline and no explicit train-test overfitting gap diagnostic included. |
| **Reproducibility** | `RANDOM_SEED=42` passed to `train_test_split`, `StratifiedKFold`, `LogisticRegression`, and `DummyClassifier`. Full `Pipeline` structure with explicit train-only fitting documented. Notebook builds `df3` fresh from raw data, reducing state dependency. | `random_state=42` throughout. Deterministic `Pipeline`. Chronological split logic is explicit and self-documenting. | `RANDOM_SEED=42` passed to `train_test_split` and `Pipeline` components. Full pipeline structure encoded in code. |
| **Failure or revision example** | No failures. Code executed cleanly with reproducible outputs. | No failures. Code executed cleanly. | No code failures. Population alteration (duplicate and zero-adult filtering before splitting) is observable but not flagged as a deviation from the original 119,390-row benchmark population. |

All three agents satisfied the minimum success criteria: code ran without error, the train-test split was applied before preprocessing, and labelled performance metrics were reported. The three outputs exhibit meaningful variation in feature scope, split strategy, and verification depth. These differences are retained as task-level benchmark evidence and examined comparatively in Section 3.

### 2.5 Task 4: Structured Model Improvement, Candidate Comparison, and Focused Tuning

Task 4 tested whether each agent could improve on the Task 3 baseline in a controlled and methodologically credible way. Each agent was required to identify baseline weaknesses explicitly, propose and compare 3–4 candidate improvement routes using training data only, select the single best-performing candidate, tune only that candidate, and evaluate the final model once on the untouched test set. Success criteria included fair candidate comparison, separation of model selection from test evaluation, explicit overfitting and leakage checks, and a credible, reproducible workflow. Table 4A records the design and selection phases; Table 4B records the evaluation and reproducibility evidence.

| Aspect | Claude | Codex | Cursor |
|---|---|---|---|
| **Baseline weaknesses identified** | Linear decision boundary; no feature engineering; class imbalance handled only via reweighting; high-cardinality features excluded conservatively; random split ignores temporal structure. | Linear decision boundary may underfit non-linear patterns; high recall but low precision tradeoff at default threshold; horizon-sensitive features may reduce robustness; temporal distribution shift between train and test. | Linear boundary cannot capture non-linear patterns identified in Task 2; class imbalance limits recall; only one model family evaluated; no bias-variance tradeoff assessment. |
| **Candidate routes proposed** | 4 routes: (C1) LR baseline reference; (C2) LR + 7 engineered features (`log_lead_time`, binary flags, ratio features); (C3) Random Forest with `class_weight='balanced'`; (C4) HistGradientBoosting. | 4 routes: (1) LR full feature set + class weighting; (2) LR conservative scope (5 horizon-sensitive variables removed); (3) Random Forest with `class_weight='balanced'`; (4) HistGradientBoosting. | 3 routes: (A) LR baseline reference; (B) LR with `class_weight='balanced'`; (C) Random Forest (n=200, max_depth=12, min_samples_leaf=2). |
| **Fair comparison design** | 5-fold stratified CV on training data only. All candidates use an identical shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` object. Test set not accessed. Primary metric: ROC-AUC. | `TimeSeriesSplit(n_splits=3)` on chronologically sorted training data. All candidates use the same temporal folds. Test set not accessed. Primary metric: ROC-AUC. Overfit gap guardrail computed per candidate (threshold ≤0.08). | 3-fold stratified CV on training data only. All candidates use the same `StratifiedKFold` object. Test set not accessed. Primary metric: ROC-AUC; secondary: F1. |
| **Best candidate selected and why** | C4 HistGradientBoosting selected: CV ROC-AUC 0.9092 ± 0.0020, gain +0.0779 over baseline. Tied with C3 RF on ROC-AUC but lower overfitting gap (0.026 vs 0.032). Gain exceeds noise threshold (2 × 0.0039 ≈ 0.008) by a factor of ~10. | HGB_tree selected: highest CV ROC-AUC (0.8829). No candidate cleared the overfit-gap guardrail (all exceeded 0.08); selection made under explicit caution, with a programmatic trust signal set to **False**. | C_rf_moderate selected: CV ROC-AUC 0.8662 ± 0.0036, highest of three candidates. Selection justified by higher ROC-AUC relative to both LR routes. No credibility threshold applied. |
| **Focused tuning process** | `RandomizedSearchCV` (20 iterations, 5-fold CV, `random_state=42`) on C4 only. Search space: `learning_rate` [loguniform 0.03–0.3], `max_leaf_nodes` [15–127], `min_samples_leaf` [10–80], `l2_regularization` [loguniform 1e-3–10], `max_iter` [150–400]. Best params: lr=0.084, max_leaf=89, min_samples=33, l2=0.244. Post-tuning CV ROC-AUC: 0.9185. | `GridSearchCV` on HGB_tree only: 8 configurations from `learning_rate` [0.03, 0.05] × `max_depth` [4, 6] × `min_samples_leaf` [20, 40]. Best params: lr=0.05, max_depth=4, min_samples_leaf=40. Post-tuning CV ROC-AUC: 0.8871. | `GridSearchCV` on C_rf_moderate only. Grid: `n_estimators` [150, 250], `max_depth` and `min_samples_leaf` ranges tested. Best params: n_estimators=250, max_depth=**None**, min_samples_leaf=**1** (no regularisation constraints). Post-tuning CV ROC-AUC: 0.9017. |
| **Test set use** | Test set held out through all candidate comparison and tuning phases. Final evaluation performed once on the held-out set after tuning. Final test ROC-AUC: **0.9185** (+0.0892 vs Task 3 baseline). | Test set held out through all candidate comparison and tuning phases. Final evaluation performed once. Final test ROC-AUC: **0.8871** (+0.0077 vs Task 3 baseline). Note: the untuned candidate achieved 0.8937 on the test set; tuning did not further improve test performance. | Test set held out through candidate comparison and tuning. Final evaluation performed once. Final test ROC-AUC: **0.9099** (+0.0431 vs Task 3 baseline). |
| **Leakage and overfitting checks** | Credibility check: gain of +0.0779 exceeds noise threshold (~0.008) by factor ~10; improvement assessed as credible. Overfitting gap for selected model: 0.026 (train minus CV AUC). Engineered features verified as leakage-safe (log transforms and binary flags derived from pre-outcome fields only). | Overfitting gap computed for all four candidates; all exceeded the 0.08 guardrail (RF highest at 0.101, HGB at 0.096). Trust signal programmatically set to **False**. Improvement reported with explicit methodological caution. | Overfitting gap not computed for any candidate. No credibility threshold applied. Tuned Random Forest parameters (`max_depth=None`, `min_samples_leaf=1`) impose no regularisation; this was not flagged or reviewed. |
| **Final comparison with Task 3 baseline** | Task 3 ROC-AUC: 0.8293 → Task 4 tuned HGB: 0.9185 (Δ +0.0892). Macro F1: +0.0962; Precision: +0.2053; Accuracy: +0.1020. Recall marginally declined (−0.0127). Three comparison plots produced: ROC curve, confusion matrix, feature importance. | Task 3 ROC-AUC: 0.8794 → Task 4 tuned HGB: 0.8871 (Δ +0.0077). PR-AUC: +0.0107; Accuracy: +0.0359; Precision: +0.1766; Recall: −0.2590; F1: −0.0362. Log Loss and Brier Score both improved. | Task 3 ROC-AUC: 0.8668 → Task 4 tuned RF: 0.9099 (Δ +0.0431). Precision: +0.0962; Recall: +0.0940; F1: +0.0962; Accuracy: +0.0449. All reported metrics improved. |
| **Remaining limitations** | Gradient boosting less interpretable than LR; random split ignores temporal structure; default threshold (0.5) not business-optimised; further hyperparameter search may yield additional gains; class imbalance still addressed via weighting only. | Improvement modest (+0.0077 ROC-AUC); trust signal explicitly False; precision-recall trade-off (−25.9% recall) may not align with business requirements; no richer feature engineering explored. | Only three candidate routes explored; gradient boosting not included; Random Forest interpretability limited; no credibility or overfitting check; class imbalance not fundamentally addressed. |
| **Reproducibility / issues** | `T4_SEED=42`; shared `StratifiedKFold` object used across all candidates; `RandomizedSearchCV` with fixed seed. No code duplication or state-overwrite issues detected. Notebook executes cleanly top-to-bottom. | `SEED_TASK4=42`; `TimeSeriesSplit` with chronological ordering; `GridSearchCV` with fixed seed. No code duplication noted. | `RANDOM_SEED=42`; `StratifiedKFold(n_splits=3)` and `GridSearchCV` with fixed seeds. **Issue:** the entire Task 3 code block is duplicated at the end of the notebook (cells 42–51), creating a variable state-overwrite risk during top-to-bottom execution. |

Across Task 4, all three agents satisfied the core structural requirements: candidate routes were compared using training data only, the test set was reserved for a single final evaluation, and focused tuning was applied to the selected candidate alone. Differences in the number of candidates, the CV strategy, the tuning search space, and the depth of overfitting and credibility checks are recorded here as task-level benchmark evidence. Cross-tool judgements on these differences are reserved for Section 3.

---

## 3. Comparative Analysis of Agent Tooling

**Table 1. Agent Performance Across Tasks and Evaluation Dimensions**

| Task | Claude Code | Codex | Cursor |
|------|-------------|-------|--------|
| Task 1: Ingestion & Schema | Strongest: empirical leakage validation, cleanest process, highest reproducibility | Deep analysis; hardcoded path reduces reproducibility | Weakest: did not read documentation; NULL-handling errors |
| Task 2: EDA | Strong; one missingness-derived feature bug required correction | Strongest: Spearman + assertions + misleading-finding audit | Weakest: three plots described but never generated |
| Task 3: Baseline Model | Strongest: dummy comparator, CV, leakage control; ROC-AUC 0.8293 | Strong temporal realism; richest metrics; ROC-AUC 0.8794 | Runnable but weakest benchmark discipline; altered population; ROC-AUC 0.8668 |
| Task 4: Model Improvement | Richest experimentation; strongest presentation; ROC-AUC 0.9185 | Strongest methodological self-audit; temporal CV; ROC-AUC 0.8871 | Weakest: no gradient boosting, unbounded RF tuning, code duplication; ROC-AUC 0.9099 |

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
