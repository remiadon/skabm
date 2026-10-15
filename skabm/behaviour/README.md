# skabm.behaviour, agent behaviours

One module per type of agent. A rule is named for the mechanism it implements, cites the
source that introduced it in the comment above it, and runs on that source's values
(`PARAMETERS`) until a caller overrides them by name. The polars functions that build
an agent's opening state sit with its rules. A model is a list of rules: each paper's is
composed in its validation test (`tests/validation`) and on its page (`notebooks/`).

| Module | Agents | Mechanisms (introduced by) | Opening state |
|---|---|---|---|
| `firm` | firms, sectors, industries | adaptive output and price, market-share profit, retained profit, hiring to capacity, dividends (Poledna et al. 2023); price-quantity signals, cost push, heuristic supply, first-pick demand, inventories, sector price indices and costs (CANVAS, Hommes et al. 2025); industry labour adjustment and rationed production with fear of infection and unemployment (Pangallo et al. 2024) | `opening_quarter` (CANVAS §3.1), `steady_state` (Pangallo et al. 2024 SI §3.4.3) |
| `household` | households | income by status, propensity consumption (Poledna et al. 2023) | `ownership` (§3.2), `opening_balance` (eq. 49, §5.2) |
| `person` | people, cells, places | Schelling relocation (Schelling 1971); place-based contacts, exposure, compartment progression, job draws (Pangallo et al. 2024) | `settle` (Schelling), `contact_population` (Pangallo et al. 2024 SI Table 5, eqs. 14, 17-18) |
| `labour_market` | occupations, job transitions | S-curve demand target, separations and openings, applications, urn-ball flow, stock-flow clearing (del Rio-Chanona et al. 2021) | `mobility_network` (eqs. 20-21), `occupations`, `reallocate_demand` (eq. 18) |
| `trader` | traders, markets | strategy demand, impact price, logit imitation (Brock & Hommes 1998, on a lattice) | `lattice`, `trading_days`, `opening_market` |
| `commuter` | commuters, routes, links, areas | route time, pivot-point mode shares, route choice, link loads, BPR delay, area traffic (Horowitz 1984; Cascetta 1989; Koppelman 1983; BPR 1964) | `routes` (shortest paths), `pedestrianize` (closing links to cars) |
| `bank` | banks and depositors | distress and depositor flight, a skabm extension with no published calibration | `depositors` |
| `macro` | government, central bank, regions | government spending, Taylor rule (Poledna et al. 2023); forecast Taylor rule (CANVAS); the region's day and lockdown (Pangallo et al. 2024) |  |
| `learning` | the forecasts every agent reads | SAC expectations (Hommes & Zhu 2014) |  |

`defaults()` merges every module's `PARAMETERS` and refuses a name with two values.
Neighbourhoods and areas are `skabm.space`: `moore_neighbors` (bounded or a torus),
`geo_neighbors`, `area_membership`.
