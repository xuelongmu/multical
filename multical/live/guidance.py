"""Operator guidance from retained observations, without claiming metric accuracy."""
import numpy as np


def capture_progress(coverage, validation, seeded=False):
    serials = list(coverage['views'])
    minimum = 3 if seeded else 12
    remaining = set(range(len(serials)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        group = []
        while pending:
            index = pending.pop()
            group.append(serials[index])
            neighbours = {int(i) for i in np.flatnonzero(coverage['overlaps'][index] > 0)} & remaining
            remaining -= neighbours
            pending.extend(sorted(neighbours))
        groups.append(group)
    target = min(serials, key=lambda s: (coverage['views'][s], np.count_nonzero(coverage['cells'][s]))) if serials else None
    ready = sum(v >= minimum for v in coverage['views'].values())
    checked = sum(validation.get(s, 0) > 0 for s in serials)
    if seeded and not any(coverage['views'].values()):
        action = 'Seed loaded: reserve new shared poses with V, then C to validate; use Space to collect training for refinement.'
    elif ready < len(serials):
        action = f'Collect varied poses for {target}: {coverage["views"][target]}/{minimum}. Move, tilt, hold still, then Space.'
    elif len(groups) > 1:
        action = 'Connect the camera groups: capture the same board in cameras from different groups at once.'
    elif checked < len(serials):
        action = 'Reserve new poses with V. Include every camera, with at least two cameras seeing each validation pose.'
    else:
        action = 'Press C to fit and inspect held-out errors and missing predictions. Counts alone do not establish accuracy.'
    return dict(minimum=minimum, ready=ready, total=len(serials), groups=groups,
                validation_cameras=checked, target=target, action=action)


def inspection_advice(observation, novel):
    if observation is None:
        return 'Waiting for detection.'
    if not observation['usable']:
        return 'Face this camera; move closer.'
    marker = observation.get('marker_px')
    if marker is not None and marker < 24:
        return 'Small markers — move closer. Size hint is advisory.'
    if not novel:
        return 'View repeats saved coverage — change position or tilt.'
    return 'New view — hold still. Space to capture.'


def inspection_group(coverage, observations, anchor=None, limit=4):
    """Rank observed opportunities, never infer physical neighbours from serials."""
    serials = list(coverage['views'])
    if not serials:
        return []
    visible = {s for s, o in observations.items() if o['usable'] and s in serials}
    need = lambda s: (coverage['views'][s], np.count_nonzero(coverage['cells'][s]), s)
    if anchor not in serials:
        anchor = min(visible or set(serials), key=need)
    groups = capture_progress(coverage, {})['groups']
    membership = {s: i for i, g in enumerate(groups) for s in g}
    index = serials.index(anchor)
    def rank(s):
        shared = int(coverage['overlaps'][index, serials.index(s)])
        # A currently co-visible camera from another component can bridge the rig.
        if anchor in visible and s in visible:
            tier = 0 if membership[s] != membership[anchor] else 1
        elif shared:
            tier = 2
        else:
            tier = 3
        return (tier, *need(s))
    partners = sorted((s for s in serials if s != anchor), key=rank)
    result = [(anchor, 'Target')]
    for s in partners[:max(0, limit-1)]:
        shared = int(coverage['overlaps'][index, serials.index(s)])
        if anchor in visible and s in visible:
            reason = 'New bridge' if membership[s] != membership[anchor] else 'Co-visible'
        elif shared:
            reason = f'{shared} shared poses'
        else:
            reason = 'Scout: overlap not established'
        result.append((s, reason))
    return result[:limit]
