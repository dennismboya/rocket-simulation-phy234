# Q1 order-effect projector model and the QQ equality on dataset D

Generated 2026-09-17 by `python -m bre.models.quantum.q1_order --report`. Every number below is computed by that command from `question_order_tables.json` (`data/raw/order_effects/`); nothing is typed in by hand.

## Model and statistic

Q1 (PLAN.md §4) is a quantum-probability model on a classical CPU: state psi = (cos alpha, sin alpha) in R^2, question A = projector onto e0, question B = projector onto (cos phi, sin phi), sequential answers by the Born rule with Lüders update, P(Ay,By) = ||P_By P_Ay psi||^2 and so on for the eight cells. The QQ statistic is q = [P(AyBn) + P(AnBy)] − [P(ByAn) + P(BnAy)] (Wang & Busemeyer 2013; Wang, Solloway, Shiffrin & Busemeyer 2014); the equality q = 0 holds for every state, dimension and pair of projectors, so it is parameter-free. Cell key: first letter = question asked first, y/n = its answer; AyBn = P(A first, yes; then B, no). The fitted parameters are the canonical representatives (phi in [0, pi/2], alpha in [0, pi), theta in (−pi/2, pi/2]) of the symmetry orbits documented in the module. The fit is least squares on the eight cells with 20 restarts from a deterministic grid. In R^2 the model also forces the four agreement conditionals to equal cos^2 phi, so a nonzero residual is expected even when q = 0; the L2 distance is a descriptive measure of that, not a test.

## Provenance

> Transcribed from src/qinst_ozawa/data.py of https://github.com/k-kyoko/qinst_ozawa_khrennikov (commit bc877e8, accessed 2026-09-17). Keys: first letter = first question asked, y/n = answer; e.g. AyBn = P(A answered yes first, then B answered no). Question A = first-named item (Clinton, Black, Rose), B = second.

## Results

### clinton_gore — VERIFIED

* Description: Gallup poll, Moore (2002): 'Do you generally think Bill Clinton / Al Gore is honest and trustworthy?' Order A-then-B vs B-then-A.
* Verification (verbatim from the JSON): "VERIFIED against Ozawa & Khrennikov (2021), J. Math. Psych. 100:102491, eqs. (149)-(156), which cite Moore 2002, Wang & Busemeyer 2013 and Wang et al. 2014. PDF sha256 recorded in SOURCE.txt."
* Intervening information between the questions: no
* Observed marginals: P(A yes | A first) = 0.5346, P(A yes | A second) = 0.5880, order effect delta_A = -0.0534; P(B yes | B first) = 0.7616, P(B yes | B second) = 0.6666, delta_B = +0.0950
* Observed QQ statistic: q = -0.0032 (P(AyBn) + P(AnBy) = 0.2214 vs P(ByAn) + P(BnAy) = 0.2246; complementary form [P(ByAy) + P(BnAn)] − [P(AyBy) + P(AnBn)] = -0.0032)
* Observed agreement conditionals: P(By|Ay) = 0.9164, P(Bn|An) = 0.6203, P(Ay|By) = 0.7386, P(An|Bn) = 0.8930; the rank-1 R^2 model forces all four to equal cos^2 phi, which the fit puts at 0.8034
* Q1 fit (theta = 0): alpha = 0.7875 rad (45.12°), phi = 0.4594 rad (26.32°); loss (least_squares, sum of squared residuals) = 7.225e-02; L2 distance = 0.2688; max |residual| = 0.1575; 20/20 restarts converged; QQ statistic of the fitted table = +0.0000e+00

| cell | observed | predicted | residual (pred − obs) |
|---|---:|---:|---:|
| Ay,By | 0.4899 | 0.4001 | -0.0898 |
| Ay,Bn | 0.0447 | 0.0979 | +0.0532 |
| An,By | 0.1767 | 0.0987 | -0.0780 |
| An,Bn | 0.2887 | 0.4034 | +0.1147 |
| By,Ay | 0.5625 | 0.7200 | +0.1575 |
| By,An | 0.1991 | 0.1762 | -0.0229 |
| Bn,Ay | 0.0255 | 0.0204 | -0.0051 |
| Bn,An | 0.2129 | 0.0834 | -0.1295 |

Figure: `figures/q1_clinton_gore.png` (observed vs predicted cells).

### black_white — UNVERIFIED secondary transcription

**This table is an UNVERIFIED secondary transcription.** It appears here only as an implementation check of the code path, not as a reported result; the JSON verification note reproduced below is binding.

* Description: Survey pair on whether Black/White people are treated fairly (Wang et al. 2014 example).
* Verification (verbatim from the JSON): "UNVERIFIED secondary transcription: the repository attributes these to Wang et al. 2014 PNAS but the numbers do not appear in the bundled PDF and PNAS is unreachable from the build session. Use only in unit tests flagged 'unverified'; never in a reported result."
* Intervening information between the questions: no
* Observed marginals: P(A yes | A first) = 0.4161, P(A yes | A second) = 0.5391, order effect delta_A = -0.1230; P(B yes | B first) = 0.4609, P(B yes | B second) = 0.5599, delta_B = -0.0990
* Observed QQ statistic: q = -0.0190 (P(AyBn) + P(AnBy) = 0.1786 vs P(ByAn) + P(BnAy) = 0.1976; complementary form [P(ByAy) + P(BnAn)] − [P(AyBy) + P(AnBn)] = -0.0190)
* Observed agreement conditionals: P(By|Ay) = 0.9582, P(Bn|An) = 0.7239, P(Ay|By) = 0.8705, P(An|Bn) = 0.7442; the rank-1 R^2 model forces all four to equal cos^2 phi, which the fit puts at 0.9975
* Q1 fit (theta = 0): alpha = 0.8164 rad (46.78°), phi = 0.0497 rad (2.85°); loss (least_squares, sum of squared residuals) = 8.389e-02; L2 distance = 0.2896; max |residual| = 0.1599; 20/20 restarts converged; QQ statistic of the fitted table = +0.0000e+00

| cell | observed | predicted | residual (pred − obs) |
|---|---:|---:|---:|
| Ay,By | 0.3987 | 0.4679 | +0.0692 |
| Ay,Bn | 0.0174 | 0.0012 | -0.0162 |
| An,By | 0.1612 | 0.0013 | -0.1599 |
| An,Bn | 0.4227 | 0.5297 | +0.1070 |
| By,Ay | 0.4012 | 0.5174 | +0.1162 |
| By,An | 0.0597 | 0.0013 | -0.0584 |
| Bn,Ay | 0.1379 | 0.0012 | -0.1367 |
| Bn,An | 0.4012 | 0.4802 | +0.0790 |

Figure: `figures/q1_black_white.png` (observed vs predicted cells).

### rose_jackson — UNVERIFIED secondary transcription

**This table is an UNVERIFIED secondary transcription.** It appears here only as an implementation check of the code path, not as a reported result; the JSON verification note reproduced below is binding.

* Description: Pete Rose / Shoeless Joe Jackson Hall-of-Fame survey pair; background information shown between questions, so the QQ equality is expected to fail.
* Verification (verbatim from the JSON): "UNVERIFIED secondary transcription (same caveat as black_white)."
* Intervening information between the questions: yes
* Observed marginals: P(A yes | A first) = 0.6620, P(A yes | A second) = 0.5390, order effect delta_A = +0.1230; P(B yes | B first) = 0.4827, P(B yes | B second) = 0.3557, delta_B = +0.1270
* Observed QQ statistic: q = +0.1514 (P(AyBn) + P(AnBy) = 0.3419 vs P(ByAn) + P(BnAy) = 0.1905; complementary form [P(ByAy) + P(BnAn)] − [P(AyBy) + P(AnBn)] = +0.1514)
* Observed agreement conditionals: P(By|Ay) = 0.5104, P(Bn|An) = 0.9473, P(Ay|By) = 0.8610, P(An|Bn) = 0.7615; the rank-1 R^2 model forces all four to equal cos^2 phi, which the fit puts at 0.7504
* Q1 fit (theta = 0): alpha = 2.6773 rad (153.40°), phi = 0.5231 rad (29.97°); loss (least_squares, sum of squared residuals) = 1.685e-01; L2 distance = 0.4105; max |residual| = 0.2621; 20/20 restarts converged; QQ statistic of the fitted table = +0.0000e+00

| cell | observed | predicted | residual (pred − obs) |
|---|---:|---:|---:|
| Ay,By | 0.3379 | 0.6000 | +0.2621 |
| Ay,Bn | 0.3241 | 0.1995 | -0.1246 |
| An,By | 0.0178 | 0.0500 | +0.0322 |
| An,Bn | 0.3202 | 0.1505 | -0.1697 |
| By,Ay | 0.4156 | 0.2277 | -0.1879 |
| By,An | 0.0671 | 0.0757 | +0.0086 |
| Bn,Ay | 0.1234 | 0.1738 | +0.0504 |
| Bn,An | 0.3939 | 0.5227 | +0.1288 |

Information variant (rotation U(theta) applied between the questions in both orders, because information intervened): alpha = 2.4752 rad (141.82°), phi = 0.1108 rad (6.35°), theta = -0.5582 rad (-31.98°); loss (least_squares, sum of squared residuals) = 3.499e-02; L2 distance = 0.1871; max |residual| = 0.1291; 20/20 restarts converged; QQ statistic of the fitted table = +1.9746e-01. Closed-form q of this variant, −sin(2 theta) sin(2 phi) = +0.1975, versus observed q = +0.1514.

| cell | observed | predicted | residual (pred − obs) |
|---|---:|---:|---:|
| Ay,By | 0.3379 | 0.3803 | +0.0424 |
| Ay,Bn | 0.3241 | 0.2376 | -0.0865 |
| An,By | 0.0178 | 0.1469 | +0.1291 |
| An,Bn | 0.3202 | 0.2351 | -0.0851 |
| By,Ay | 0.4156 | 0.4131 | -0.0025 |
| By,An | 0.0671 | 0.0951 | +0.0280 |
| Bn,Ay | 0.1234 | 0.0920 | -0.0314 |
| Bn,An | 0.3939 | 0.3997 | +0.0058 |

Figure: `figures/q1_rose_jackson.png` (observed vs predicted cells).

## Uncertainty

The JSON carries proportions only, without respondent counts, so no confidence interval or significance test is computed for any statistic above; the weighted-multinomial fit option exists in the code but is unused.

## Interpretation

What this shows: the Q1 code reproduces the QQ equality exactly on model-generated tables (unit tests, |q| < 1e-12 for random parameters) and computes the statistic on the one verified published table (clinton_gore: q = -0.0032). What it does not show: this is an implementation check on a single verified table, not a replication of the 70-survey result of Wang et al. (2014); without sample sizes the observed q cannot be compared with its sampling error, so no claim that the equality holds or fails on real data is made here. The unverified tables (black_white: q = -0.0190, rose_jackson: q = +0.1514) are shown only to exercise the code path, including the information variant for the table with intervening information; their values are not results until the owner verifies the transcription (SOURCE.txt lists the action). The fitted (alpha, phi) are descriptive: two parameters against six free cell values, with the R^2 model imposing constraints beyond the QQ equality, so the residuals say how far each table is from the rank-1 R^2 family, nothing about causality or about whether a quantum-probability model is preferable to a classical one (that is the Phase 4 question).
