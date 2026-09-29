# Registration instructions for gated datasets (GATE — nothing has been registered or agreed to)

Per CLAUDE.md rule 4(b) the build session does not register for datasets or accept data-use terms.
The owner does the steps below; afterwards place files where indicated and run `make data`.

## E. Understanding America Study (USC CESR)
1. Create a UAS Data User account at https://uasdata.usc.edu (Registration Form page).
2. Read and submit the Data User Agreement (the site's summary: no sharing with unregistered
   collaborators, download only what the research needs, cite UAS, secure storage). Cost: free.
3. Download: core waves with financial literacy, stock-market expectation and risk items; the
   "Understanding Coronavirus in America" tracking survey (biweekly from April 2020) with financial
   behavior items; the codebooks; and any question/response-order randomization assignment variables.
4. Place under `data/raw/uas/<wave>/` with the codebooks; keep the files out of git if the agreement
   requires it (the .gitignore already excludes `data/raw/uas/`).
5. What the loader will do: map each risk/expectation item to a schema row per person-wave, tag the
   wave date relative to 2020-02-19 (pre/during/post crash) as context, and put reported financial
   actions in `outcome_behavior` with lag_days.

## H. LISS panel and DNB Household Survey (Centerdata, Netherlands)
1. Sign the LISS data user statement at https://statements.centerdata.nl/liss-panel-data-statement
   (non-commercial research only; no release of individual information). Cost: free.
2. On approval, download from the LISS Data Archive the Economic Situation: Assets module and the
   risk-attitude items (core study, yearly), for waves spanning 2008–2010 and 2019–2021.
3. DNB Household Survey: request access through Centerdata; download the psychological concepts
   (risk attitude) and assets modules for the same years.
4. Place under `data/raw/liss/` and `data/raw/dhs/`; both are git-ignored.
5. Note: Dutch sample; the report will label any RQ3 result as non-US.

## Also unreachable from the build session but free without registration
* FINRA NFCS Investor Survey: https://www.finrafoundation.org/nfcs-data-and-downloads → `data/raw/finra_nfcs/<year>/`.
* PNAS Wang et al. 2014 Dataset S1: doi:10.1073/pnas.1407756111 SI → `data/raw/order_effects/wang2014_si/`.
* Psych-101: Hugging Face marcelbinz/Psych-101 (check license on the card) → `data/raw/psych101/`.
* Global Preferences Survey: https://www.briq-institute.org/global-preferences/downloads (site registration) → `data/raw/gps/`.
* Robintrack: https://robintrack.net/data-download → `data/raw/robintrack/`.
* Zenodo originals for CPC18 (records 845873, 2571510): confirm the license and checksum against the mirror in `data/raw/cpc18/`.
