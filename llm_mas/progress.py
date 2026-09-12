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


class JobProgress:
    """Monotone overall progress; stages carry equal weight, layers subdivide a stage.

    ETA is a rough extrapolation, not a claim that stages have equal costs.
    """
    def __init__(self, stages):
        self.stages=list(dict.fromkeys(stages))
        self.fractions={stage:0.0 for stage in self.stages}
        self.started=time.monotonic()

    def update(self, stage, completed=None, total=None):
        if stage=='complete':
            self.fractions={name:1.0 for name in self.stages}
        elif stage in self.stages:
            for name in self.stages[:self.stages.index(stage)]:self.fractions[name]=1.0
            if total and completed is not None:
                self.fractions[stage]=max(self.fractions[stage],min(1.0,completed/total))
        fraction=sum(self.fractions.values())/len(self.stages)
        elapsed=time.monotonic()-self.started
        eta=elapsed*(1-fraction)/fraction if 0<fraction<1 else 0 if fraction==1 else None
        return dict(overall_fraction=fraction,overall_estimated_remaining_seconds=eta,
                    stages_completed=sum(x==1 for x in self.fractions.values()),stages_total=len(self.stages))
