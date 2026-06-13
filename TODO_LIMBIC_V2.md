# Limbic Hermes — 29 More Biochemistry/Gap Improvements

This document lists 29 testable improvements identified after reviewing the
current Limbic Hermes implementation and researching limbic neurology. They
are written as testable prompts so each can be verified with a pytest case.

## Architecture / Module gaps

1. **Add an explicit Insula module for interoceptive salience.**
   Prompt: "Given interoceptive signals (HRV, respiration, cytokine load), the
   Insula computes a `body_prediction_error` that amplifies threat appraisal when
   interoception is discordant."

2. **Add an Anterior Cingulate Cortex (ACC) conflict/error signal.**
   Prompt: "When valence is near zero, arousal is high, and dominance is low,
   the ACC emits a `conflict_signal` that boosts NE and reduces safety."

3. **Add a Lateral Habenula (LHb) aversion-learning module.**
   Prompt: "Unexpected negative events activate the LHb, which inhibits
   dopamine and increases serotonin release; repeated negative events should
   lower expected reward more than simple RPE."

4. **Add Septal nuclei social-approach modulation.**
   Prompt: "Oxytocin + low cortisol activates septal social-approach, raising
   valence toward affiliative events and reducing threat response to social
   cues."

5. **Add a Periaqueductal Gray (PAG) defensive tier detector.**
   Prompt: "Based on arousal/dominance and threat flag, classify defensive
   response as freeze (high arousal, very low dominance), flight (high arousal,
   low dominance), fight (high arousal, high dominance), or calm."

## Autonomic / interoceptive

6. **Add polyvagal state model (ventral vagal / sympathetic / dorsal vagal).**
   Prompt: "HRV, oxytocin, and safety map to a discrete vagal state; the state
   gates whether social engagement is possible or defensive shutdown occurs."

7. **Add vagal brake release under acute threat.**
   Prompt: "A sudden threat event should transiently lower HRV and increase
   adrenaline, modeling vagal withdrawal."

8. **Add baroreflex-like arousal dampening at high HRV.**
   Prompt: "Sustained high HRV should gradually reduce arousal and NE."

9. **Add respiration-driven entrainment of arousal.**
   Prompt: "A user-controlled respiration rate slider modulates NE and
   oscillates arousal with the respiratory cycle (inhalation up, exhalation
   down)."

## Neurochemistry depth

10. **Add gaba_a and glun2b receptor sensitivity tracking.**
    Prompt: "GABA-A and GluN2B receptor sensitivities exist in state, decrease
    under chronic GABA/glutamate exposure, and recover with rest; the dashboard
    displays them."

11. **Add kynurenine pathway shunt under inflammation.**
    Prompt: "High cytokine load reduces tryptophan availability, lowering
    serotonin synthesis and raising quinolinic acid, which increases glutamate
    excitotoxicity."

12. **Add D2 autoreceptor / short-loop feedback.**
    Prompt: "High dopamine triggers D2 autoreceptor inhibition, reducing further
    dopamine release and increasing dopamine pool recovery cost."

13. **Add MAO-A / MAO-B degradation dynamics.**
    Prompt: "Catecholamine and serotonin levels decay at rates proportional to
    effective MAO activity, which can be modulated by cofactors/heritage."

14. **Add COMT Val158Met-style dopamine clearance phenotype.**
    Prompt: "A tunable `dopamine_clearance_rate` biases how fast dopamine
    returns to baseline; slower clearance (COMT Met/Met) increases sustained
    dopamine but also receptor desensitization risk."

15. **Add synaptic serotonin transporter (SERT) reuptake modulation.**
    Prompt: "A `serotonin_reuptake` parameter controls serotonin decay rate,
    modeling SSRIs/heritage effects."

16. **Add dopamine pathway-specific signaling (mesolimbic vs mesocortical).**
    Prompt: "Track `dopamine_mesolimbic` (motivation/salience) separately from
    `dopamine_mesocortical` (cognitive control); the latter contributes more to
    dominance."

17. **Add noradrenergic locus coeruleus tonic/phasic mode.**
    Prompt: "LC mode is tonic (exploration, high NE baseline) or phasic (focused,
    bursts); task load and surprise switch between modes."

## Motivation / affective systems (Panksepp-inspired)

18. **Add SEEKING system strength.**
    Prompt: "Dopamine + orexin + low cortisol drive a `seeking` motivation state
    that increases reward gain and exploration/novelty bias."

19. **Add CARE system strength.**
    Prompt: "Oxytocin + prolactin + safety drive a `care` state that raises
    warmth expression and reduces threat to dependency/social events."

20. **Add FEAR/RAGE thresholds.**
    Prompt: "Threat events above a cortisol- and GABA-dependent threshold trigger
    `fear_active`; when dominance is high and GABA low, the same threat triggers
    `rage_active`."

21. **Add GRIEF/PANIC separation from simple sadness.**
    Prompt: "Social separation events (abandonment, silence after intimacy)
    activate `panic_grief` driven by opioid withdrawal and oxytocin drop."

## Learning / memory

22. **Add fear extinction learning in the hippocampus-PFC loop.**
    Prompt: "Repeated safe exposures to a previously threatening event kind
    reduce its hippocampal weight (extinction)."

23. **Add working-memory load as a drive and cholinergic demand.**
    Prompt: "A `working_memory_load` input increases ACh demand and task load,
    and reduces DMN activity more strongly."

24. **Add temporal contiguity for LTP (events close in time cluster).**
    Prompt: "Events arriving within a short window strengthen each other's
    weights, modeling hippocampal pattern completion."

## Circadian / metabolic

25. **Add cortisol awakening response (CAR) and ultradian rhythm.**
    Prompt: "Cortisol follows a circadian curve peaking in the morning; a
    `circadian_hour` input should shift the cortisol baseline."

26. **Add leptin/ghrelin hunger-energy state.**
    Prompt: "A `metabolic_energy` input modulates orexin, dopamine, and
    irritability; low energy increases rest_need and caution."

27. **Add glucose/insulin brain fuel dynamics.**
    Prompt: "Glucose availability modulates prefrontal top-down regulation; low
    glucose weakens dominance regulation of threat."

## Dashboard / UX

28. **Add dashboard event-injection controls (preset buttons).**
    Prompt: "The dashboard has buttons for common events (user message, success,
    error, praise, conflict, rest) that post to the server and update state."

29. **Add dashboard export/import of limbic state JSON.**
    Prompt: "The dashboard provides download/upload buttons for the full limbic
    state so users can save and restore configurations."

---

## Implementation approach

For each item:
1. Write a failing pytest in `tests/test_improvements.py` using the prompt text as
   the docstring.
2. Implement the minimal code change in `limbic_hermes/`.
3. Run `pytest tests/ -v` until the specific test passes.
4. Update the dashboard if a new state field is exposed.
5. Repeat in batches of 3–5 to keep the loop tight.
