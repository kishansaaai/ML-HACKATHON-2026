> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 8. Academic Integrity, Plagiarism and AI Content Disclosure

This section is a disclosure draft. The facts below are taken from the repository (`README.md`, "AI assistance" and "Rules the code follows"; `Assessment and Plagiarism Policy.pdf`, section 5) and from how this draft itself was produced. Every sentence must be checked by the team against what actually happened; where the draft cannot know, it says so.

## 8.1 What AI assistance was used for

The README states: "AI tools were used to help write and review code and notes. Reported results come from the committed scripts and their output files. Team members are responsible for understanding and explaining all submitted work."

In more detail (team to confirm and correct):

* Writing and reviewing source code in `src/pathsnap/`, `scripts/`, `ui/` and tests in `tests/`.
* Writing and tidying documentation and notes under `docs/` and `docs/notes/`.
* Drafting report text. The section drafts in `docs/report/` (including this one, sections 6 and 7) were prepared by an AI assistant from repository files, and they are explicitly drafts that the team must rewrite in its own words.
* Reviewing the repository for stale or inconsistent documentation (for example the Person D UI notes in `docs/REPORT_SECTION_7.md` and `docs/VIVA_PREP_PERSON_D.md` were corrected to match `ui/app.py`).

[MISSING: which AI tools and versions each team member used, and for which components. The repository does not record this per member; the team must fill it in.]

## 8.2 What AI assistance was not used for

The policy prohibits using AI to substitute for experiments (`Assessment and Plagiarism Policy.pdf`, section 5). The team's position, to be confirmed by the team:

* Metrics, tables and graphs are produced by committed scripts and saved as files (`results/`, `reports/`, `artifacts/`). The README rule is "Every number in the report must come from a script and a file in this repository." AI tools were not used to invent, estimate or type in result values. In the drafts written with AI help, each number carries the path of the file it comes from, and where no file supports a number the draft says `[MISSING]`.
* Model comparison, Bayesian tuning and decoder tuning were run by executing the scripts listed in the README ("Reproduce the experiments"). Decoder parameters were tuned on out-of-fold scores for training trips and frozen before the `student_test` check (`results/hmm_tuning/`, `configs/decoder_best.json`), not on the evaluator partition (README, "Rules the code follows").
* No evaluator data is in the repository, and the evaluator ground truth is never read by the prediction pipeline or the UI (see section 7, 7.3).

## 8.3 What an AI tool cannot verify (the team must do this)

An AI assistant working from the repository text cannot confirm any of the following. Each needs a human check and a human sentence in the final report.

1. That the reported numbers are what the committed scripts actually produce on a fresh run. The drafts copy numbers from CSV/JSON files; they do not re-run the training or tuning. Example: `results/api_student_test.csv` appears to predate the final decoder settings (its HMM Top-1 for trip 00000052 differs from `results/hmm_tuning/final_student_test_check.csv`), so it must be regenerated before use.
2. That the demonstrated system on the two laptops matches the committed code (the policy requires this). No two-laptop run has been recorded in the repository yet (README status table: "to do at the demo").
3. That screenshots and maps in the report come from the team's own real runs on the stated partition. Maps and screenshots are not embedded in the drafts.
4. Why the models and the decoder behave as they do on particular trips (for example why the HMM lowers Top-1 on trips 00000007, 00000077 and 00000091). The data show that it does; the interpretation must be the team's own, supported by the team's own inspection.
5. That each team member can explain the code, experiments, results and conclusions in the report (viva topics listed in `Assessment and Plagiarism Policy.pdf`, section 6: dataset structure, preprocessing, feature engineering, model comparison and selection, Bayesian optimisation, HMM, Viterbi, API, UI, results).
6. That the report text is original and not substantially similar to AI-generated text that other teams may also have produced. AI-written text must be rewritten, not lightly edited.
7. That no secrets or private credentials are committed (`Deliverables.pdf`, section 1) and that the repository is public and accessible to evaluators.
8. That no code, weights, configurations or results were shared with or copied from another team.

## 8.4 Statements the team can adapt (draft wording, to be rewritten)

* "We used AI assistants to help write and review code and documentation and to draft report text. All experiments were run by us using the scripts in this repository, and every reported number is taken from a result file in the repository."
* "Each team member has read and can explain the parts of the system listed in the viva topics; the team rewrote AI-drafted report text in its own words."

These sentences are only true if the team makes them true.

## 8.5 Items still open

* [MISSING: names of AI tools used and a per-person description of use.]
* [MISSING: the event's required form or format for AI disclosure, if one exists beyond the text in the guideline documents (the guideline says "Disclose AI assistance as required by the event's policies", but the policy PDFs read here do not specify a format).]
* [MISSING: team confirmation of the statements in 8.2.]
