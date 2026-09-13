# Reversal generated-choice recovery

Audit of `full_reversed_v2`: 120 conversations, 650 responses, including 360
generated responses. 283 generated choices parsed successfully; 77 failed JSON
parsing. All 77 have a complete strategy and explanation string but lack the
final object-closing brace. All before/decision/post stance and margin fields
are finite and present. This is not a missing-stance issue or a Markdown-fence
issue.

Parser `3-missing-final-object-brace` accepts this narrowly defined case by
appending one final `}` for parsing only. The resulting object must contain
exactly `strategy` and `explanation` and pass the existing schema checks.
Truncated strings, invalid strategies, extra fields in recovered objects, and
surrounding prose remain invalid. There is no score-based choice inference.
Generated rows record `generation.choice_parser_version` and
`generation.choice_recovery`. Raw generation, public message, and retained
history are unchanged. Dashboard formatting remains separate.

Do not overwrite or resume v2 after this change. Recovery changes public choice
validity and therefore adaptive stopping. Run into a fresh directory:

```bash
cd ~/mas-llm-opinion-formation
git pull --ff-only
mkdir -p logs
nohup .venv/bin/python -u -m llm_mas study \
  --config studies/full_behavioral_reversed.json \
  --device cuda --run-id full_reversed_v3 \
  > logs/full_reversed_v3.log 2>&1 < /dev/null &
```

This is a changed measurement policy, not a repair of the saved v2 results.
For controlled presentation-order comparisons, generate the corresponding
forward-presentation conditions with this same parser version; do not interpret
a v2/v3 contrast as presentation alone. The earlier behavioral/fixed reports
continue to describe their original saved data and original parser policy.
