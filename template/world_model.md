# World Model — MyDomain
# Agent instructions for brain, director, extractor, and specialist.
# Write as a precise specification. Agents act on this directly.

## Variables

# List every state variable: type, range, key threshold, what crossing it means.
# Example:
#   carry: storage cost per month. Range: -0.02 to +0.08.
#          Threshold: +0.02 (above = storage pays, hold bias increases sharply).

[variable_name]: [description]. Range: [min] to [max]. Threshold: [value] ([what it means]).


## Decision Rules

# Explicit IF/THEN rules that govern good decisions. Include variable interactions.
# Example:
#   IF carry > 0.02 AND sa_risk < elevated → hold bias (storage paying, supply risk ahead)
#   IF basis < -0.40 → sell regardless of carry (basis collapse overrides all)

IF [condition] → [action] because [mechanism]


## Strategy Space

# Directives for generating 16 archetypes. Be explicit about what MUST exist.
# Name the archetype types, the contrarian bet, the interesting combinations.
# Example:
#   MUST INCLUDE: one archetype that ignores carry entirely, trades only on basis momentum
#   MUST INCLUDE: one pure contrarian that bets against the obvious seasonal move
#   DIMENSION 1: aggressive vs conservative on [variable]
#   DIMENSION 2: timing-first vs basis-first

MUST INCLUDE: [archetype description]
DIMENSION: [what spans this axis]


## Extraction Guidance

# Explicit patterns to extract vs artifacts to reject.
# Example:
#   EXTRACT: IF [context] AND [condition] → [outcome] (conditional, has mechanism)
#   REJECT:  any principle claiming X always beats Y — sim cannot verify unconditional dominance
#   WATCH:   [variable] × [variable] interactions — these produce the most valuable principles

EXTRACT: IF [context] → [pattern worth learning]
REJECT:  [class of false principles this sim tends to produce]
WATCH:   [variable interaction worth paying attention to]

RETIRED TOPICS (do not regenerate these): {{RETIRED_TOPICS}}


## Director Watchlist

# Exact failure modes for this domain the director must flag.
# Be specific — name the condition, not a generic description.

- [exact condition that indicates reward hacking in this sim]
- [exact condition that indicates mode collapse]
- [principle pattern that is a sim artifact, not real domain knowledge]


## Success Criteria

Job: [one sentence — what decision does this specialist make]
Excellent: [specific and numeric — what good outcomes look like]
Never: [hard constraints — what must never happen]
Abstain when: [conditions where specialist should escalate to a human]
