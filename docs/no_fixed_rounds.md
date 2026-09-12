# No fixed rounds

Select **No fixed rounds** and set **Maximum update rounds**. After each complete update round, every agent must satisfy all three conditions against its own preceding-round response:

- Same public strategy.
- Full-message cosine similarity >= 0.8.
- Unchanged post-response margin sign: `(current >= 0) == (previous >= 0)`. Zero belongs to the Strategy 1 side.

There is no minimum-round gate. The first check compares round 1 with initialization round 0. Margin magnitude may change without preventing stopping. Agreement between agents is not required.

The full endorsement and explanation are embedded using `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. A small CPU encoder avoids competing with conversation models for GPU memory. Embeddings are cached within each conversation. Long messages are split into 254-token chunks, pooled by attention-masked mean, combined by chunk token count and L2 normalized. No message ending is silently truncated. Weights may download on first use.

If all conditions hold for every agent, termination is `system_detected_stability`; otherwise the cap produces `max_rounds`. Each check saves similarities, per-agent margin changes and sign consistency, strategy stability and similarity threshold. No CONTINUE/FINISH probes are used.

New CLI configurations use `mode: no_fixed_rounds`, `rounds` as the cap, optional `similarity_threshold` (0.8). They resolve to `stopping: sign_stability`; legacy explicit stopping policies retain their old behavior. Start a new run after restarting the dashboard.

This compares consecutive rounds only and does not guarantee that later oscillation would not occur. Shared endorsements may increase message similarity. The raw margin check avoids relying on the saturation of bounded stance.
