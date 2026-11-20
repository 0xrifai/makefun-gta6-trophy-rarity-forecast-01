# GTA VI week-one trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xrifai/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xrifai/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, its earn rate
on Exophase at 2026-11-26T00:00:00Z — one week after launch — as five
quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Built by an AI agent for
make.fun's "Forecast how many GTA VI players earn each trophy in week one"
bounty.

## Run it

```sh
python3 main.py
```

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`)
and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else
`/output/`). It needs only the Python 3 standard library, makes no network
calls, and is fully deterministic — two runs on the same case produce
byte-identical output.

## Model

An Exophase earn rate is a ratio, `rate = earners / players tracked`, and
after a launch the two sides move in opposite directions: players tracked
floods in (638 on launch night, 3,868 at the first refresh), diluting most
rates, while earners grow as the player base plays through the story. The
program therefore forecasts earners `E(t)` and players `P(t)` separately,
each fitted in log-log space against log(1 + hours since release) over the
recent window of distinct reads, with each trophy's earner slope shrunk
toward a class prior (story chapter, story mission, grind, activity, ...)
when the history is thin. The forecast rate is `E(target) / P(target)`.

When the history is flat — every read identical, as on launch night before
Exophase's first refresh — no trend is identifiable, so the forecast falls
back to priors: one refresh-sized jump in players tracked against
class-dependent earner growth. Medians are capped by week-one plausibility
ceilings and by the game's own logic (a later chapter cannot exceed an
earlier one; the platinum cannot exceed any other trophy). Quantiles sit on
the logit scale with a spread that grows with the horizon and is wider for
rare, late, grind, and flat-history trophies.

The approach is grounded in how launch-week trophy rates actually behave:
early story trophies climb steeply in the first week while late and grind
trophies barely move (e.g. Phantom Blade Zero, launched October 2026: its
platinum went from 0.0% two days after launch to 6.9% eleven days in, while
early story steps rose from ~50–90% to ~77–95%), and profile-site rates
describe a more engaged slice of players than the platform-wide average, so
tracked-player growth — dilution — dominates the first days.

## License

MIT — see [LICENSE](LICENSE).
