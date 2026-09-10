"""Phase progress shared by the CLI and dashboard."""
import time

class Progress:
    def __init__(self):self.phase=None;self.started=time.monotonic();self.phase_started=self.started;self.count=0
    def update(self,phase,completed=None,total=None):
        now=time.monotonic()
        if self.phase!=phase:self.phase=phase;self.phase_started=now;self.count=0
        if completed is not None:self.count=completed
        elapsed=now-self.phase_started
        eta=elapsed*(total-self.count)/self.count if total is not None and self.count>0 else None
        return dict(phase=phase,elapsed_seconds=now-self.started,phase_elapsed_seconds=elapsed,phase_completed=self.count,phase_total=total,estimated_remaining_seconds=max(0,eta) if eta is not None else None)
