#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase one week after launch.

Model: rate(t) = E(t) / P(t), forecasted separately per trophy.
  * P(t) = players tracked by Exophase.  It floods in after launch (638 on
    launch night, 3,868 at the first refresh), so most rates first FALL by
    dilution even while players are earning trophies.
  * E_i(t) = tracked players who have earned trophy i.  It grows as the
    player base plays through the game; early story trophies gain earners
    fast, late / grind / collectible trophies gain them slowly.

Both series are fitted in log-log space against log(1 + hours since
release): log P = c + b_P * x, log E_i = a_i + b_i * x, with each trophy's
slope shrunk toward a class prior (story chapter, mission, grind, ...) when
the history is thin.  The forecast is E_i(target) / P(target).  When the
history is flat (every read identical, as on launch night before Exophase's
first refresh) no trend is identifiable, so the forecast falls back to
priors: one refresh-sized jump in players tracked against class-dependent
earner growth.  Medians are then capped by week-one plausibility ceilings
and by the game's own logic (a later chapter cannot be earned by more
players than an earlier one; the platinum cannot exceed any other trophy).

Uncertainty is honest: quantiles are placed on the logit scale with a
spread that grows with the horizon and is wider for rare, late, grind and
flat-history trophies.  The program is deterministic (no randomness) and
uses only the Python standard library.

Usage: python3 main.py   (reads $MAKEFUN_INPUT_DIR else /input/,
writes $MAKEFUN_OUTPUT_DIR else /output/, one <id>.out per <id>.in).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp into a datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def logit(p):
    """p in percent; clamp to the range the scorer clamps to."""
    p = min(99.99, max(0.01, p)) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    return 100.0 / (1.0 + math.exp(-l))


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Trophy classification from tier / description.  Drives the priors:
#   slope0   prior slope of log E_i against log(1 + hours since release)
#   e_growth median earner-growth factor over a flat-history horizon
#   med_cap  plausible ceiling for the week-one median rate (%)
#   abs_cap  hard ceiling for any quantile (%)
#   spread   extra logit spread for the class
# ---------------------------------------------------------------------------

def classify(trophy):
    name = trophy.get("name", "")
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier", "bronze")
    tid = trophy.get("id", 0)

    if tier == "platinum":
        return ("platinum", 0.55, 1.10, 2.0, 4.0, 0.35)
    if "100% completion" in desc or "100% completion" in name.lower():
        return ("completion100", 0.45, 1.05, 3.0, 6.0, 0.30)

    m = re.search(r"complete chapter (\d)", desc)
    if m:
        ch = int(m.group(1))
        # Earlier chapters are finished by far more players, sooner.
        slope = {1: 1.55, 2: 1.35, 3: 1.15, 4: 0.95}.get(ch, 1.0)
        egr = {1: 2.5, 2: 2.0, 3: 1.7, 4: 1.5}.get(ch, 1.4)
        cap = {1: 80.0, 2: 65.0, 3: 50.0, 4: 40.0}.get(ch, 35.0)
        return ("story_chapter_%d" % ch, slope, egr, cap, cap * 1.6, 0.10)
    if "complete the final mission" in desc:
        return ("story_final", 0.85, 1.4, 30.0, 48.0, 0.12)
    if desc.startswith("complete ") and ("mission" in desc or "run" in desc
                                         or "heist" in desc or "play" in desc
                                         or "takedown" in desc or "operation" in desc):
        return ("story_mission", 1.05, 1.6, 45.0, 70.0, 0.12)

    hard_grind = ("250 stunts" in desc or "100 locations" in desc
                  or "100 vehicles" in desc or "3 collectible card" in desc
                  or "every spot" in desc or "every location" in desc
                  or "gold medal in all" in desc)
    if hard_grind:
        return ("grind_hard", 0.55, 1.10, 12.0, 24.0, 0.25)
    med_grind = ("20 different races" in desc or "50 challenges" in desc
                 or "10 vehicles" in desc or "5 bank vaults" in desc
                 or "8 mission strands" in desc or "each gang" in desc
                 or "all of leonida" in desc or "all garage" in desc
                 or "$250,000" in desc or "$100,000" in desc
                 or "each type of activity" in desc)
    if med_grind:
        return ("grind_med", 0.75, 1.20, 30.0, 55.0, 0.15)
    if ("complete an arcade game" in desc or "fishing challenge" in desc
            or "win" in desc):
        return ("activity", 1.10, 1.50, 60.0, 85.0, 0.15)
    return ("misc", 0.95, 1.40, 50.0, 80.0, 0.15)


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------

def distinct_reads(history):
    """Collapse consecutive identical reads (Exophase refreshes are lumpy:
    the page sits unchanged between refreshes).  Keep the FIRST time a
    distinct page state appears, so trends are dated when the state arrived.
    Returns list of (hours_since_release, players, rates list)."""
    out = []
    prev_key = None
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or rates is None:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read["at"])
        except Exception:
            continue
        out.append((hours_since_release(t), players, rates))
    return out


def linfit(xs, ys):
    """Least-squares y = a + b*x.  Returns (a, b, n, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b, n, resid)


# ---------------------------------------------------------------------------
# Forecast for one case
# ---------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    h_last = hours_since_release(at)
    h_tgt = hours_since_release(target)

    reads = distinct_reads(history)
    # Last read's rates (fall back to the case's last history entry).
    if reads:
        last_players, last_rates = reads[-1][1], reads[-1][2]
    elif history:
        last_players = history[-1].get("players_tracked") or 1
        last_rates = history[-1].get("rates") or [None] * n
    else:
        last_players, last_rates = 1, [None] * n

    classes = [classify(t) for t in trophies]

    # --- players-tracked trend -------------------------------------------
    # Fit log P against log(1 + hours) on distinct reads, preferring the
    # recent window when there is enough data.
    xs_all = [math.log1p(h) for h, _, _ in reads]
    ys_p = [math.log(max(1, p)) for _, p, _ in reads]
    fit_p = None
    if len(reads) >= 4:
        # last 36 hours of distinct reads if that leaves >= 4 points
        cut = h_last - 36.0
        idx = [i for i, (h, _, _) in enumerate(reads) if h >= cut]
        if len(idx) >= 4:
            fit_p = linfit([xs_all[i] for i in idx], [ys_p[i] for i in idx])
    if fit_p is None and len(reads) >= 2:
        fit_p = linfit(xs_all, ys_p)
    if fit_p is not None:
        b_p = clamp(fit_p[1], 0.0, 2.0)
    else:
        b_p = None

    flat = len(reads) < 2  # no variation at all in the history

    x_last = math.log1p(h_last)
    x_tgt = math.log1p(h_tgt)

    # --- per-trophy forecast ----------------------------------------------
    medians = []
    sigmas = []
    for i, t in enumerate(trophies):
        cls, slope0, egr, med_cap, abs_cap, class_spread = classes[i]
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = 0.01
        r_last = clamp(float(r_last), 0.0, 100.0)
        e_last = max(r_last * last_players / 100.0, 0.0)

        sigma2 = 0.0
        # Base spread: grows like a random walk with the horizon.
        sigma_base = 0.35 + 0.35 * math.sqrt(max(horizon_days, 0.0))
        sigma2 += sigma_base ** 2
        sigma2 += class_spread ** 2
        if r_last < 1.0:
            sigma2 += 0.25 ** 2
        elif r_last < 2.0:
            sigma2 += 0.15 ** 2

        if flat:
            # No trend is identifiable: one refresh-sized jump in players
            # tracked against class-dependent earner growth.
            sigma2 += 0.55 ** 2
            if r_last >= 99.9:
                # Earned by essentially every tracked player so far; the
                # intro trophy stays near-universal (it did not fall at
                # the first refresh while every other rate did), but with
                # hundreds of thousands tracked by the target, exactly
                # 100.00% is no longer a safe bet, so the band is honest
                # about both outcomes.
                median = 98.0
                sigma2 = 0.45 ** 2
            else:
                p_growth = 5.0  # first-refresh scale: 638 -> 3,868 players
                median = r_last * egr / p_growth
        else:
            # Fit this trophy's earners in log-log space, preferring the
            # recent window: adoption curves bend, so the current growth
            # rate is what the target inherits, not the launch-night one.
            xs, ys = [], []
            for h, p, rates in reads:
                r = rates[i] if i < len(rates) else None
                if r is None:
                    continue
                e = max(float(r) * p / 100.0, 0.5)
                xs.append(math.log1p(h))
                ys.append(math.log(e))
            if len(xs) >= 4:
                cut = math.log1p(max(0.0, h_last - 36.0))
                idx = [j for j, x in enumerate(xs) if x >= cut]
                if len(idx) >= 4:
                    xs, ys = [xs[j] for j in idx], [ys[j] for j in idx]
            fit_e = linfit(xs, ys) if len(xs) >= 2 else None
            if fit_e is not None and fit_e[2] >= 3:
                a_e, b_e_fit, n_e, resid = fit_e
                # Shrink the fitted slope toward the class prior; the more
                # distinct reads, the more the data speaks.
                w = n_e / (n_e + 4.0)
                b_e = w * clamp(b_e_fit, 0.0, 2.5) + (1.0 - w) * slope0
                # Re-anchor the level at the last observed earners so a
                # noisy slope cannot drag the level away from the data.
                log_e_last = math.log(max(e_last, 0.5))
                log_e_tgt = log_e_last + b_e * (x_tgt - x_last)
                sigma2 += min(resid, 0.5) ** 2
            elif fit_e is not None:
                # Only 2 distinct reads: use the fitted level, prior slope.
                log_e_last = math.log(max(e_last, 0.5))
                log_e_tgt = log_e_last + slope0 * (x_tgt - x_last)
            else:
                log_e_last = math.log(max(e_last, 0.5))
                log_e_tgt = log_e_last + slope0 * (x_tgt - x_last)

            if b_p is not None and fit_p is not None:
                log_p_last = math.log(max(last_players, 1))
                log_p_tgt = log_p_last + b_p * (x_tgt - x_last)
            else:
                # Players trend unidentified: reuse the flat-history
                # refresh prior, scaled by the horizon in log-time.
                log_p_tgt = math.log(max(last_players, 1)) + math.log(5.0)
            median = 100.0 * math.exp(log_e_tgt - log_p_tgt)
            if r_last >= 99.9:
                # Universal trophy: earners track players one-for-one,
                # but the exact 100.00% print is not a safe bet at scale.
                median = 98.0
                sigma2 = 0.45 ** 2

        median = clamp(median, 0.01, 99.99)
        # Week-one plausibility ceiling for the class -- except for a
        # trophy essentially every tracked player already holds, which
        # stays near-universal by definition.
        if r_last < 99.9:
            median = min(median, med_cap)
        medians.append(median)
        sigmas.append(math.sqrt(sigma2))

    # --- game-logic guards -------------------------------------------------
    # Story chapters are ordered: later chapters cannot exceed earlier ones.
    chain = []
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "").lower()
        m = re.search(r"complete chapter (\d)", desc)
        if m:
            chain.append((int(m.group(1)), i))
    chain.sort()
    final_idx = None
    for i, t in enumerate(trophies):
        if "complete the final mission" in (t.get("description") or "").lower():
            final_idx = i
    if final_idx is not None:
        chain.append((99, final_idx))
    run_min = 100.0
    for _, i in chain:
        run_min = min(run_min, medians[i])
        medians[i] = run_min

    # Platinum cannot exceed any other trophy; 100%-completion gold cannot
    # exceed the hardest grinds.
    plat_idx = next((i for i, t in enumerate(trophies)
                     if t.get("tier") == "platinum"), None)
    if plat_idx is not None:
        others = [medians[i] for i in range(n) if i != plat_idx]
        if others:
            medians[plat_idx] = min(medians[plat_idx], min(others))
    comp_idx = next((i for i, t in enumerate(trophies)
                     if "100% completion" in (t.get("description") or "")),
                    None)
    if comp_idx is not None:
        grinds = [medians[i] for i, c in enumerate(classes)
                  if c[0] in ("grind_hard", "grind_med")]
        if grinds:
            medians[comp_idx] = min(medians[comp_idx], min(grinds))

    # --- quantiles ----------------------------------------------------------
    rows = []
    for i, t in enumerate(trophies):
        _, _, _, med_cap, abs_cap, _ = classes[i]
        med = medians[i]
        sig = sigmas[i]
        base = logit(med)
        qs = []
        for z in Z:
            q = sigmoid(base + z * sig)
            qs.append(round(clamp(q, 0.01, 99.99), 4))
        # Enforce non-decreasing (rounding can only tie, never invert, but
        # be safe) and the class absolute ceiling (not for a near-universal
        # trophy, whose quantiles legitimately sit near 100).
        if last_rates[i] is not None and float(last_rates[i]) < 99.9:
            qs = [min(q, abs_cap) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": int(t["id"]), "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        sys.stderr.write("no .in files in %s\n" % in_dir)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
