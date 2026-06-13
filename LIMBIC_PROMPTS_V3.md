# TODO_LIMBIC_V3.md
# 20 more biochemistry-grounded Limbic Hermes improvements

Each item is written as a testable prompt. Implemented in `tests/test_improvements_v3.py`.

## 1. Mammillary body / Papez circuit consolidation
During rest, recently weighted episodic event kinds are strengthened (consolidated)
into longer-term hippocampal weights, modeling mammillary-body/anterior-thalamic replay.

## 2. Entorhinal grid-cell novelty signal
Novel contexts (unknown event kinds) generate an entorhinal novelty signal that boosts
acetylcholine and theta, enhancing encoding of new associations.

## 3. Nucleus accumbens shell vs core
Track `nucleus_accumbens_shell` (motivational wanting/salience) and
`nucleus_accumbens_core` (action selection / response vigor) separately; shell follows
dopamine spikes, core follows task load and expected reward.

## 4. RMTg dopamine brake
Unexpected negative events activate a rostromedial tegmental GABAergic brake that
actively suppresses dopamine release beyond simple lateral habenula effects.

## 5. BNST sustained anxiety / CRF
Sustained low-safety contexts raise a `crf` neuropeptide level and `bnst_apprehension`,
creating a slowly-decaying anxious background that outlasts individual threat events.

## 6. PAG column specialization
Subdivide PAG output into active (dorsolateral) and passive (ventrolateral) columns;
dominance/threat context selects between fight/flight (active) and freeze (passive).

## 7. Dorsal vs median raphe serotonin
Split serotonin tone into `dorsal_raphe` (anxiety/avoidance, threat-reactive) and
`median_raphe` (memory/context stability); threat lowers dorsal and raises median.

## 8. Orexin sleep-pressure flip
Add a `sleep_pressure` variable; high sleep pressure flips orexin from promoting
wake to signaling transition need, reducing histamine and raising melatonin gating.

## 9. CRF stress amplifier
A dedicated `crf` signal amplifies cortisol and adrenaline under sustained uncertainty,
and is itself reduced by safety and oxytocin.

## 10. Neuropeptide Y resilience
Add `neuropeptide_y` as a stress-resilience buffer; it rises after successful coping
and dampens BNST/CRF-driven anxiety and amygdala threat reactivity.

## 11. Dynorphin / kappa opioid aversion
Add `dynorphin` as a counter-reward signal; it spikes on unexpected negative outcomes
and produces dysphoria / anhedonia that suppresses dopamine and seeking.

## 12. Anandamide / FAAH extinction gating
Model `anandamide` separately from generic eCB and add `faah_activity`; low FAAH +
high anandamide during safe exposures accelerates fear extinction (reduces threat weights).

## 13. GABA transporter (GAT) tone
Add `gat_activity` controlling GABA clearance; high GAT lowers effective GABAergic
inhibition, increasing excitability and irritability.

## 14. Astrocyte GLT-1 / glutamate clearance
Add `glt1_activity` (astroglial glutamate reuptake) and `lactate`; low GLT-1 lets
glutamate and quinolinic acid rise, increasing excitotoxicity risk.

## 15. Microglial priming / neuroimmune memory
Add `microglia_state` that primes upward with cytokine/cortisol load; once primed,
future stressors produce larger cytokine responses (sensitization).

## 16. Astrocyte glycogen-lactate shuttle
Add `glycogen` and `lactate` metabolic buffers; high task load consumes glycogen and
raises lactate, which transiently boosts neural capacity but causes fatigue if depleted.

## 17. Theta-gamma coupling
Add a `theta_gamma_coupling` index; high ACh + moderate arousal raises coupling,
which boosts working-memory encoding and hippocampal weight updates.

## 18. Sharp-wave ripple / place-cell replay
During rest, high-importance event sequences are "replayed", strengthening their
weights and consolidating them faster than simple decay recovery.

## 19. Ventral pallidum wanting vs liking
Add `ventral_pallidum` reflecting hedonic "liking"; separate from nucleus accumbens
"wanting" so reward events can raise wanting without raising liking if opioid is low.

## 20. Subgenual anterior cingulate rumination switch
Add `subgenual_acc` activity; high cortisol + low serotonin + repeated negative events
shift ACC from conflict monitoring to rumination, slowing recovery of negative valence.
