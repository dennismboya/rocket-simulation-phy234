# Benchmark feature checklist from non-public panic-selling studies (entry K)

Neither study's data is obtainable and none was sought. What follows is limited to what the papers'
abstracts and public summaries state, as retrieved by web search on 2026-09-17; the full texts were not
reachable from the build session. Use this as (1) a checklist of covariates the intake battery should
collect and (2) the classical benchmark "in spirit" for the report.

## Elkind, Kaminski, Lo, Siah & Wong (2022), "When Do Investors Freak Out? Machine Learning Predictions of Panic Selling", Journal of Financial Data Science 4(1):11–39
Source summaries: SSRN abstract page (https://privpapers.ssrn.com/sol3/papers.cfm?abstract_id=3898940), Semantic Scholar entry.
* Data: 653,455 individual brokerage accounts in 298,556 households (US retail brokerage), not public.
* Outcome definition ("panic sale"): a decline of 90% of a household account's equity assets within one month, of which 50% or more is due to trades. The dashboard's "sell" is the stated analog; the report must not equate the two.
* Reported demographic predictors of higher freak-out frequency: male; age above 45; married; more dependents; self-identified "excellent" investment experience or knowledge.
* Market-state finding: disproportionate panic sales during sharp market downturns; panic selling is predictable and distinct from overtrading and the disposition effect.
* Reported model performance: best deep neural network at 70% true-positive and 81% true-negative rate.
* Checklist implications for the intake battery and covariates JSON: gender is deliberately NOT collected (PII/fairness policy in README); age_band, marital status and dependents are candidates for an optional covariate block; self-reported experience/knowledge is already in the schema (invest_experience_yrs, financial_literacy_score, self_reported_risk_tolerance) — add an "overconfidence" item (self-rated knowledge minus quiz score) as a derived feature.

## Hiroshima University × Rakuten Securities "Survey on Life and Money" (2020–2023 waves; N ≈ 134,013 in the 2023 wave)
Source summaries: PLOS One (Overconfidence, financial literacy, and panic selling: Evidence from Japan; https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315622), Risks 12(10):162 (framing during the COVID-19 crisis; https://doi.org/10.3390/risks12100162), Behavioral Sciences 14(9):795 (hyperbolic discounting), Risks 12(12):203 (neuroticism), Cogent Economics & Finance 2025 (financial attitude and behavior).
* Panic-selling prevalence during COVID-19 as reported: 1.71% sold all stocks; 6.10% sold a portion.
* Reported predictors: financial knowledge, financial behavior and financial attitude all reduce panic selling (behavior and attitude add to knowledge); overconfidence increases it; neuroticism increases it; negative framing and fear are named triggers; hyperbolic discounting is associated with it.
* Checklist implications: keep the financial-literacy quiz; add optional short items for financial attitude/behavior, a two-item neuroticism screen and a one-item present-bias screen to the full form's optional block; the news-frame context in the design is the analog of the framing manipulation.

## How this feeds the project
* The classical benchmark "to beat in spirit" is a gradient-boosted / neural classifier on demographics + self-reports + market state (B5 in PLAN.md).
* None of the numbers above are used as model inputs; they are a literature checklist only.
