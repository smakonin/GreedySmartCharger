"""SCR v2: explicit primary-score tolerance for stable lexicographic action ties.

Only action-score tie selection changes. Physics, forecasts, calibration,
liability and strict economic admission remain those of v1.
"""
from grid_model import transition
from shadow_reserve import couple,liability,value_table,propose,Ledger as OriginalLedger

SCORE_TOL=1e-7


def choose_index(actions,scores,x,soc,b,indices=None):
    eligible=list(range(len(actions))) if indices is None else list(indices)
    if not eligible:raise ValueError('No actions')
    best=min(scores[i] for i in eligible)
    face=[i for i in eligible if scores[i]<=best+SCORE_TOL]
    return max(face,key=lambda i:transition(x,soc,actions[i],b).end_soc)


class Ledger(OriginalLedger):
    def choose(self,x,soc,shadow,shadow_action,candidates,scores,b):
        modified=list(scores)
        if self.price_low<=x.price<=self.price_high:
            sf=transition(x,shadow,shadow_action,b)
            safe=[]
            for i,a in enumerate(candidates):
                f=transition(x,soc,a,b)
                phi=liability(f.end_soc,sf.end_soc,b,self.price_low,self.price_high)
                if self.regret+f.cost-sf.cost+phi<=self.budget+self.breach_debit:safe.append(i)
            if safe:
                # Flatten only the admitted score face. The inherited chooser
                # then maximizes next SOC on this face. No safety slack is added.
                best=min(scores[i] for i in safe)
                for i in safe:
                    if scores[i]<=best+SCORE_TOL:modified[i]=best
        return super().choose(x,soc,shadow,shadow_action,candidates,modified,b)
