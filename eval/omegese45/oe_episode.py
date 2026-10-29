"""Interface v1.3 refinements around the immutable v1.1 sealed oracle.

The committed oe45lib.py and oe_sealgen.py bytes stay unchanged: this wrapper
only rejects newly clarified illegal episode shapes and handles terminal-job
binding without calling the frozen library with None. It changes neither the
sealed messages nor their reference answers, metrics or denominators.
"""
import oe45lib as L


def check_episode_shape(msgs):
    why = L.check_episode_shape(msgs)
    if why:
        return why
    refuses = [i for i, m in enumerate(msgs) if m.kind == "A" and m.jk == "refuse"]
    if refuses and not (refuses == [1] and len(msgs) == 3
                        and msgs[2].text == "rep refused bound end"):
        return "refuse-not-sole-first-A-followed-by-bound-P"
    return None


def bind_current(goal, act, prefix):
    j = L.current_job_index(prefix)
    if j is None:
        return False, "binding: no current job"
    return L.binding_ok(goal, act, j)
