# Data map — which dataset serves which research question (2026-09-17)

| Dataset | Status | Unit | Loss framing | Context manipulation | Sequence / order | Later real action | RQ1 | RQ2 (a) | RQ2 (b) | RQ3 |
|---|---|---|---|---|---|---|---|---|---|---|
| A choices13k | acquired (aggregate) | problem | lottery losses | feedback on/off | block | no | – | transfer only | – | – |
| B CPC15 | mirror (aggregate) | problem×block | lottery losses | feedback | block | no | – | transfer target | – | – |
| B CPC18 raw | mirror (individual) | trial | experienced loss | experienced outcome sign | yes, randomized order | no | yes | yes | yes (pre-registered analog) | – |
| C Psych-201 spektor2024 | acquired (individual) | trial | mixed lotteries | gain/loss/mixed domain | sessions | no | yes | yes | partial | – |
| C Psych-201 spektor2019, olschewski2024 | acquired (individual) | trial | experienced | option set | blocks | no | partial | yes | – | – |
| D order tables | 1 verified + 2 unverified | aggregate | none | question order | yes | no | yes (QQ) | – | – | – |
| E UAS | GATE | person-wave | real market drawdown | natural (Mar 2020) | waves | reported actions | – | – | – | yes |
| F FINRA NFCS | instructions | person | stated | none | no | stated | priors | – | – | – |
| G HRS/SCF/PSID | instructions | household-wave | real | natural | waves | holdings | – | – | – | sanity |
| H LISS/DNB | GATE | person-wave | real | natural | waves | holdings | – | – | – | best candidate |
| I GPS | instructions | person | stated | none | no | no | priors | – | – | – |
| J Robintrack | instructions | ticker-day | real | natural | daily | aggregate | – | – | – | composition check |
| Intake battery (Phase 7) | to build | item | manipulated | manipulated | manipulated | via intervention_log | yes | yes | yes | later |

Still missing for the exact product question (manipulated loss × manipulated context × later real action, within subject): nothing public provides it; the intake battery plus intervention_log is the only path, and the PROLIFIC.md design costs it.
