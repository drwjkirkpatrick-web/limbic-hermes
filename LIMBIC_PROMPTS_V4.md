# LIMBIC_PROMPTS_V4.md
# 20 Additional Biochemistry-Grounded Limbic Hermes Improvements

Each item is written as a testable prompt. Implemented in `tests/test_improvements_v4.py`.

## 1. Basolateral Amygdala (BLA) vs Central Amygdala (CeA)
**Prompt:** When a negative sensory event arrives, the BLA computes sensory-
integrative threat appraisal (valence/arousal from raw input), while the CeA
computes fear-expression output (freezing/fight/flight downstream signals).
High CeA should drive BNST/PAG activation. Low GABA should increase CeA output.

## 2. Infralimbic (IL) vs Prelimbic (PL) vmPFC
**Prompt:** The IL cortex promotes fear extinction (inhibits CeA, boosts
extinction memory), while the PL promotes fear expression (excites CeA).
Safe exposures should strengthen IL activity and suppress PL. IL strength
should gate hippocampal extinction learning rate.

## 3. Dentate Gyrus Pattern Separation
**Prompt:** Similar but distinct event kinds should be kept separate in
hippocampal weights (pattern separation). When two events share a prefix
but differ in suffix, their weights should not interfere. High ACh should
boost separation.

## 4. CA3 Pattern Completion
**Prompt:** Partial cues of a previously learned event kind should activate
the full hippocampal weight via pattern completion. When an event kind is a
substring of a known kind, the known weight should partially activate.

## 5. D1-Direct vs D2-Indirect Striatal Pathways
**Prompt:** The direct pathway (D1 receptors) promotes action / "go",
while the indirect pathway (D2 receptors) suppresses action / "no-go".
High dopamine with high D1 sensitivity should increase action vigor (NAcc
core). High dopamine with D2 autoreceptor activation should suppress action.

## 6. Locus Coeruleus Tonic/Phasic Mode Switch
**Prompt:** LC should have a mode state: "tonic" (baseline NE, exploratory)
or "phasic" (bursts, focused). Sustained low arousal should flip to tonic.
Unexpected high-salience events should flip to phasic. Mode should affect
cortisol sensitivity.

## 7. Suprachiasmatic Nucleus (SCN) Master Clock
**Prompt:** An SCN oscillator should drive circadian phase. The SCN phase
should entrain melatonin, cortisol, orexin, and histamine with appropriate
phase offsets (melatonin peaks at night, cortisol in morning, orexin on
waking). Phase should advance with time even without external input.

## 8. Arcuate POMC/AgRP Hunger Circuit
**Prompt:** An opponent hunger/satiety circuit: AgRP neurons promote hunger
(raises ghrelin-like signal), POMC neurons promote satiety (raises PYY-like
signal). Low glucose should boost AgRP. High metabolic energy should boost
POMC. The net signal should modulate orexin and seeking.

## 9. VTA GABA Interneuron Brake
**Prompt:** VTA contains GABA interneurons that locally inhibit dopamine
neurons. High VTA GABA activity should suppress dopamine release even when
reward signals are present. BNST and RMTg activation should increase VTA GABA.

## 10. Medial Habenula Value Comparison
**Prompt:** The medial habenula compares expected vs actual reward magnitude.
When expected reward exceeds actual, medial habenula activity rises and
suppresses dopamine. When actual exceeds expected, activity falls and dopamine
rises. This is distinct from lateral habenula's surprise detection.

## 11. Nucleus Reuniens Hippocampal-PFC Bridge
**Prompt:** The nucleus reuniens (midline thalamus) bridges hippocampal
output to prefrontal cortex during memory consolidation. During rest with
high theta-gamma coupling, reuniens activity should strengthen PFC-top-down
regulation and improve future extinction recall.

## 12. Claustrum Salience Gating
**Prompt:** The claustrum gates which limbic events reach conscious awareness
(or the agent's reportable state). High claustrum activity should increase
attention weight for salient events. Low claustrum should allow background
events through. Novel events should transiently boost claustrum.

## 13. Tuberomammillary Nucleus (TMN) Histamine Source
**Prompt:** Histamine should have a TMN source that tracks sleep pressure
and circadian phase. TMN activity should be inverse to melatonin and sleep
pressure. TMN output should drive cortical arousal and suppress slow-wave
activity.

## 14. Parabrachial Nucleus (PBN) Interoception
**Prompt:** The PBN relays visceral/interoceptive signals to insula and
amygdala. PBN activity should rise with cytokine load, cortisol, and pain
(substance P). PBN should drive insula prediction error and modulate threat
appraisal.

## 15. RVLM Sympathetic Tone Control
**Prompt:** The rostral ventrolateral medulla (RVLM) sets sympathetic
tone. RVLM activity should drive norepinephrine and adrenaline. High HRV
(baroreflex) should inhibit RVLM. Low safety should disinhibit RVLM.

## 16. Nucleus Tractus Solitarius (NTS) Vagal Afferents
**Prompt:** The NTS processes visceral vagal afferents before relaying to
PBN/amygdala. NTS activity should reflect gut signals (cytokine, metabolic
state). NTS should gate polyvagal state transitions and modulate HRV.

## 17. Cerebellar Fastigial Affective Timing
**Prompt:** The cerebellar fastigial nucleus provides timing and prediction
for emotional responses. Fastigial activity should rise when events occur at
predictable intervals. Surprise timing (event arrives early/late) should
suppress fastigial and boost amygdala unpredictability signals.

## 18. PVN Stress Integration
**Prompt:** The paraventricular nucleus (PVN) integrates stress signals
and coordinates HPA output. PVN CRF release should be gated by amygdala
(CeA), BNST, and NTS inputs. PVN should also coordinate oxytocin and
vasopressin release during social stress.

## 19. Adult Neurogenesis Rate
**Prompt:** The dentate gyrus has an adult neurogenesis rate that gates
learning capacity. High BDNF + low cortisol + low cytokine should boost
neurogenesis. High stress should suppress it. Neurogenesis rate should
modulate how quickly new hippocampal weights form.

## 20. Blood-Brain Barrier Permeability
**Prompt:** The BBB has a permeability state that gates how peripheral
cytokines reach the brain. Chronic stress, high cortisol, and inflammation
should increase BBB permeability. High BDNF and low stress should tighten it.
BBB permeability should scale the cytokine load effect on neurochemistry.
