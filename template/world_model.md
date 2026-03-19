# World Model — MyDomain

## Domain Understanding

[What is this domain? What real-world process does it model?
 What are the key dynamics, tensions, and failure modes?
 What does the AI director need to watch for during training?]

Known failure modes to monitor:
- [e.g. If score rises but strategies converge → mode collapse]
- [e.g. If one archetype wins regardless of scenario → sim artifact]
- [e.g. If principles claim X is always better → sim can't verify this]

## Strategy Space

[How should strategy archetypes be generated? What dimensions matter?
 What is the goal? What does a good outcome look like?
 What range of strategies should be explored — conservative, aggressive, contrarian?]

Each archetype must be:
- Named distinctively (the name conveys the philosophy)
- Grounded in a clear philosophy about what to optimize
- Meaningfully different from the others
- Internally consistent — parameter values reflect the stated philosophy

Generate 16 archetypes covering the full strategy space. Include at least one
clear contrarian that bets against the obvious move.

## Extraction Guidance

[What principles are worth learning from tournament results?
 What context factors matter most? What patterns are real vs artifacts?]

A good principle:
- Is conditional ("IF X AND Y THEN Z works because...")
- Is specific to this domain — not generic advice
- Has a clear mechanism (explains *why* the winner won)
- Would change how the next archetype library is designed

A bad principle:
- Is obvious or trivially true
- Is too specific to generalize (single-round artifact)
- Restates what any reasonable strategy would do

RETIRED TOPICS (do not regenerate these): {{RETIRED_TOPICS}}

Return 0 principles if nothing notable happened this round.

## Success Criteria

### Job
[What decision does this specialist make? One sentence.]

### What good looks like
[What outcomes indicate the specialist is working?
 Be specific — numbers, thresholds, observable behaviors.]

### Abstain when
[When should the specialist refuse to act and escalate to a human?]

### Failure
[What does a bad outcome look like? What must never happen?]

## Current Hypotheses

- (none yet — populated after first batch)

<!-- hypotheses-start -->
<!-- hypotheses-end -->
