"""
limbic_hermes/metabolic_snp.py
==============================

Human metabolic SNP module for the Limbic Hermes system.

Maps clinically relevant pharmacogenomic variants to neurochemical
and affective state modifiers. SNP effects are modeled as fractional
adjustments to baseline neurochemistry, receptor sensitivity, and
temperament parameters.

These are educational simulations grounded in pharmacogenomic literature.
They are NOT clinical recommendations or diagnostic tools.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SNPVariant:
    """A single SNP variant with its neurochemical effects."""
    gene: str
    rs_id: str
    variant_name: str
    # Effect sizes: positive = increase, negative = decrease
    # Applied as multipliers to baseline neurochemistry
    dopamine_mod: float = 0.0          # Mesolimbic/mesocortical dopamine
    serotonin_mod: float = 0.0         # Serotonin baseline
    gaba_mod: float = 0.0              # GABA tone
    glutamate_mod: float = 0.0         # Glutamate tone
    norepinephrine_mod: float = 0.0    # Noradrenaline
    bdnf_mod: float = 0.0              # BDNF level
    cortisol_mod: float = 0.0          # HPA-axis baseline
    d1_sensitivity_mod: float = 0.0    # D1 receptor sensitivity
    d2_sensitivity_mod: float = 0.0    # D2 receptor sensitivity
    gaba_a_sensitivity_mod: float = 0.0  # GABA-A sensitivity
    glun2b_sensitivity_mod: float = 0.0  # NMDA GluN2B sensitivity
    anandamide_mod: float = 0.0        # eCB/anandamide tone
    cytokine_mod: float = 0.0          # Baseline cytokine load
    circadian_mod: float = 0.0         # Circadian rhythm stability
    threat_gain_mod: float = 0.0       # Temperament: threat reactivity
    reward_gain_mod: float = 0.0       # Temperament: reward sensitivity
    decay_factor_mod: float = 0.0      # Temperament: emotional decay
    description: str = ""

    def to_effects_dict(self) -> Dict[str, float]:
        """Return all non-zero effects as a flat dict."""
        effects = {}
        for k, v in self.__dict__.items():
            if k.endswith("_mod") and v != 0.0:
                effects[k.replace("_mod", "")] = v
        return effects


# ---------------------------------------------------------------------------
# Tier 1-2 SNPs with strong limbic/receptor effects
# ---------------------------------------------------------------------------

COMT_VAL158MET = SNPVariant(
    gene="COMT",
    rs_id="rs4680",
    variant_name="Val158Met",
    dopamine_mod=0.15,           # Met/Met = slow COMT = higher PFC dopamine
    d1_sensitivity_mod=0.1,
    cortisol_mod=-0.05,          # Better stress resilience with Met
    threat_gain_mod=-0.1,        # Lower threat reactivity
    decay_factor_mod=-0.05,      # Slower emotional decay (more stable)
    description=(
        "COMT Val158Met: Met allele reduces dopamine clearance in prefrontal cortex. "
        "Met/Met carriers have higher tonic dopamine, better cognitive flexibility under stress, "
        "but may be more sensitive to pain and stress. Val/Val = fast clearance = lower tonic DA."
    ),
)

MTHFR_C677T = SNPVariant(
    gene="MTHFR",
    rs_id="rs1801133",
    variant_name="C677T",
    serotonin_mod=-0.08,         # T allele = reduced methylation = lower serotonin synthesis
    dopamine_mod=-0.05,
    gaba_mod=-0.05,
    bdnf_mod=-0.08,              # T allele associated with lower BDNF
    cortisol_mod=0.1,            # Higher baseline HPA activity
    threat_gain_mod=0.1,
    description=(
        "MTHFR C677T: T allele reduces methylenetetrahydrofolate reductase activity by ~30-70%. "
        "Impairs one-carbon metabolism, reducing SAMe availability for neurotransmitter methylation. "
        "Associated with depression, anxiety, and elevated homocysteine."
    ),
)

MTHFR_A1298C = SNPVariant(
    gene="MTHFR",
    rs_id="rs1801131",
    variant_name="A1298C",
    dopamine_mod=-0.04,
    serotonin_mod=-0.04,
    bdnf_mod=-0.05,
    description=(
        "MTHFR A1298C: C allele impairs BH4 recycling, affecting dopamine and serotonin synthesis. "
        "Often co-occurs with C677T; compound heterozygotes have stronger metabolic effects."
    ),
)

SLC6A4_5HTTLPR = SNPVariant(
    gene="SLC6A4",
    rs_id="5-HTTLPR",
    variant_name="S/L allele",
    serotonin_mod=0.1,           # S allele = less transporter = more synaptic 5-HT
    cortisol_mod=0.15,           # S allele = higher HPA reactivity to stress
    threat_gain_mod=0.15,        # S allele = higher neuroticism/threat sensitivity
    decay_factor_mod=-0.1,       # Slower recovery from negative affect
    description=(
        "SLC6A4 5-HTTLPR: Short (S) allele produces less serotonin transporter (SERT). "
        "More synaptic serotonin but also more anxiety/depression under stress (gene x environment). "
        "Long (L) allele = higher SERT expression = more efficient reuptake."
    ),
)

DRD2_TAQ1A = SNPVariant(
    gene="DRD2",
    rs_id="rs1800497",
    variant_name="Taq1A (A1)",
    dopamine_mod=-0.1,           # A1 allele = reduced D2 receptor density
    d2_sensitivity_mod=-0.15,
    reward_gain_mod=-0.1,        # Blunted reward sensitivity
    anandamide_mod=-0.05,
    description=(
        "DRD2 Taq1A: A1 allele associated with 30-40% reduction in striatal D2 receptor density. "
        "Linked to addiction vulnerability, impulsivity, and reduced reward sensitivity. "
        "May increase susceptibility to dopaminergic drug effects."
    ),
)

BDNF_VAL66MET = SNPVariant(
    gene="BDNF",
    rs_id="rs6265",
    variant_name="Val66Met",
    bdnf_mod=-0.2,               # Met allele = reduced activity-dependent BDNF secretion
    cortisol_mod=0.1,            # Met carriers show greater cortisol response to stress
    threat_gain_mod=0.1,
    glun2b_sensitivity_mod=-0.05,
    description=(
        "BDNF Val66Met: Met allele impairs activity-dependent BDNF secretion and trafficking. "
        "Associated with smaller hippocampal volume, poorer episodic memory, and stress vulnerability. "
        "Val/Val = normal plasticity and resilience."
    ),
)

MAOA_UVNR = SNPVariant(
    gene="MAOA",
    rs_id="uVNTR",
    variant_name="3R/4R/5R",
    serotonin_mod=0.15,          # Low-activity alleles (3R) = higher 5-HT
    dopamine_mod=0.1,
    norepinephrine_mod=0.1,
    cortisol_mod=-0.05,
    threat_gain_mod=-0.05,       # Paradoxically, low MAO-A may reduce fear
    reward_gain_mod=0.1,
    description=(
        "MAOA uVNTR: Low-activity alleles (3R) reduce monoamine oxidase A by ~50%. "
        "Higher baseline serotonin, dopamine, and norepinephrine. "
        "Linked to aggression/impulsivity in males under stress, but also reward sensitivity."
    ),
)

HTR2A_T102C = SNPVariant(
    gene="HTR2A",
    rs_id="rs6311",
    variant_name="T102C",
    serotonin_mod=0.05,          # C allele = reduced receptor expression
    cortisol_mod=0.05,
    threat_gain_mod=0.08,
    description=(
        "HTR2A T102C: C allele reduces 5-HT2A receptor expression. "
        "Associated with depression, suicidality, and altered psychedelic response. "
        "T allele = higher receptor density = better antidepressant response to SSRIs."
    ),
)

OPRM1_A118G = SNPVariant(
    gene="OPRM1",
    rs_id="rs1799971",
    variant_name="A118G",
    anandamide_mod=-0.1,         # G allele = reduced mu-opioid receptor function
    dopamine_mod=-0.05,
    reward_gain_mod=-0.1,
    description=(
        "OPRM1 A118G: G allele (Asn40Asp) reduces mu-opioid receptor affinity by ~3x. "
        "Associated with higher pain sensitivity, lower alcohol euphoria, and naltrexone non-response. "
        "May increase vulnerability to opioid dependence."
    ),
)

FAAH_C385A = SNPVariant(
    gene="FAAH",
    rs_id="rs324420",
    variant_name="C385A",
    anandamide_mod=0.2,          # A allele = reduced FAAH = higher anandamide
    cortisol_mod=-0.1,
    threat_gain_mod=-0.15,
    gaba_mod=0.05,
    description=(
        "FAAH C385A: A allele reduces FAAH enzyme activity, increasing anandamide levels. "
        "Associated with lower anxiety, better stress resilience, and reduced fear extinction deficits. "
        "The 'resilience SNP.'"
    ),
)

CYP2D6_PM = SNPVariant(
    gene="CYP2D6",
    rs_id="*3/*4/*5",
    variant_name="Poor Metabolizer",
    dopamine_mod=-0.05,          # PM = altered drug metabolism affects neurochemistry indirectly
    serotonin_mod=-0.05,
    description=(
        "CYP2D6 Poor Metabolizer: No functional CYP2D6 enzyme. "
        "Cannot metabolize codeine, tramadol, many SSRIs/SNRIs, and beta-blockers. "
        "Affects psychiatric medication response profoundly."
    ),
)

CYP2D6_UM = SNPVariant(
    gene="CYP2D6",
    rs_id="*1xN/*2xN",
    variant_name="Ultra-Rapid Metabolizer",
    dopamine_mod=0.05,
    serotonin_mod=0.05,
    description=(
        "CYP2D6 Ultra-Rapid Metabolizer: Multiple active gene copies. "
        "Rapidly clears medications; may need higher SSRI doses. "
        "Codeine converts to morphine too quickly — toxicity risk."
    ),
)

CYP2C19_PM = SNPVariant(
    gene="CYP2C19",
    rs_id="*2/*3",
    variant_name="Poor Metabolizer",
    serotonin_mod=0.1,           # PM = higher SSRI levels = more serotonergic effect
    gaba_mod=0.05,
    description=(
        "CYP2C19 Poor Metabolizer: Reduced CYP2C19 activity. "
        "Higher SSRI levels (e.g., escitalopram, sertraline), increased side effects. "
        "Also affects clopidogrel activation and PPI metabolism."
    ),
)

GABRA2_RS279858 = SNPVariant(
    gene="GABRA2",
    rs_id="rs279858",
    variant_name="A/G",
    gaba_mod=-0.1,               # Risk allele reduces GABA-A receptor function
    gaba_a_sensitivity_mod=-0.1,
    cortisol_mod=0.1,
    threat_gain_mod=0.1,
    description=(
        "GABRA2 rs279858: Risk allele associated with reduced GABA-A alpha-2 receptor function. "
        "Linked to alcohol dependence, anxiety disorders, and impulsivity. "
        "May impair benzodiazepine response."
    ),
)

GRIN2B_RS890 = SNPVariant(
    gene="GRIN2B",
    rs_id="rs890",
    variant_name="T/G",
    glutamate_mod=0.1,           # Risk allele increases NMDA receptor function
    glun2b_sensitivity_mod=0.1,
    cortisol_mod=0.05,
    threat_gain_mod=0.08,
    description=(
        "GRIN2B rs890: Risk allele increases NMDA receptor function. "
        "Associated with OCD, anxiety, and excitotoxicity vulnerability. "
        "May increase ketamine/psychedelic sensitivity."
    ),
)

DAT1_VNTR = SNPVariant(
    gene="SLC6A3",
    rs_id="VNTR",
    variant_name="9R/10R",
    dopamine_mod=0.1,            # 9R = lower DAT expression = higher synaptic DA
    d1_sensitivity_mod=0.05,
    reward_gain_mod=0.1,
    description=(
        "DAT1 VNTR: 9-repeat allele reduces dopamine transporter expression. "
        "Higher synaptic dopamine, increased reward sensitivity, ADHD association. "
        "10R = normal clearance."
    ),
)

CHRNA5_RS16969968 = SNPVariant(
    gene="CHRNA5",
    rs_id="rs16969968",
    variant_name="Asp398Asn",
    norepinephrine_mod=0.1,
    threat_gain_mod=0.08,
    description=(
        "CHRNA5 rs16969968: Risk allele reduces nicotinic receptor function. "
        "Strongly associated with nicotine dependence and smoking initiation. "
        "May increase anxiety and stress reactivity."
    ),
)

CLOCK_RS1801260 = SNPVariant(
    gene="CLOCK",
    rs_id="rs1801260",
    variant_name="3111T/C",
    circadian_mod=-0.15,         # C allele = delayed sleep phase
    cortisol_mod=0.08,
    serotonin_mod=-0.05,
    description=(
        "CLOCK 3111T/C: C allele associated with delayed sleep phase and evening preference. "
        "Higher evening cortisol, altered melatonin rhythm. "
        "Linked to depression and bipolar disorder."
    ),
)

PER3_VNTR = SNPVariant(
    gene="PER3",
    rs_id="VNTR",
    variant_name="4R/5R",
    circadian_mod=-0.1,
    cortisol_mod=0.05,
    description=(
        "PER3 VNTR: 5-repeat allele associated with morning preference but also sleep loss vulnerability. "
        "4-repeat = more resilient to sleep deprivation. "
        "Affects circadian entrainment and HPA axis timing."
    ),
)

APOE_RS429358 = SNPVariant(
    gene="APOE",
    rs_id="rs429358",
    variant_name="e4",
    bdnf_mod=-0.1,
    cortisol_mod=0.1,
    glutamate_mod=0.05,
    threat_gain_mod=0.05,
    description=(
        "APOE e4: Strongest genetic risk factor for Alzheimer's disease. "
        "Associated with reduced neuroplasticity, higher HPA axis reactivity, and impaired stress resilience. "
        "Also linked to sleep disruption and reduced BDNF signaling."
    ),
)

FKBP5_RS1360780 = SNPVariant(
    gene="FKBP5",
    rs_id="rs1360780",
    variant_name="T/C",
    cortisol_mod=0.15,           # T allele = enhanced glucocorticoid receptor sensitivity
    bdnf_mod=-0.1,
    threat_gain_mod=0.12,
    description=(
        "FKBP5 rs1360780: T allele enhances GR sensitivity and impairs negative feedback. "
        "Associated with PTSD, depression, and elevated cortisol after stress. "
        "Gene x environment: childhood trauma interaction is strong."
    ),
)

# ---------------------------------------------------------------------------
# SNP Registry
# ---------------------------------------------------------------------------

SNP_REGISTRY: Dict[str, SNPVariant] = {
    "COMT_Val158Met": COMT_VAL158MET,
    "MTHFR_C677T": MTHFR_C677T,
    "MTHFR_A1298C": MTHFR_A1298C,
    "SLC6A4_5HTTLPR": SLC6A4_5HTTLPR,
    "DRD2_Taq1A": DRD2_TAQ1A,
    "BDNF_Val66Met": BDNF_VAL66MET,
    "MAOA_uVNTR": MAOA_UVNR,
    "HTR2A_T102C": HTR2A_T102C,
    "OPRM1_A118G": OPRM1_A118G,
    "FAAH_C385A": FAAH_C385A,
    "CYP2D6_PM": CYP2D6_PM,
    "CYP2D6_UM": CYP2D6_UM,
    "CYP2C19_PM": CYP2C19_PM,
    "GABRA2_rs279858": GABRA2_RS279858,
    "GRIN2B_rs890": GRIN2B_RS890,
    "DAT1_VNTR": DAT1_VNTR,
    "CHRNA5_rs16969968": CHRNA5_RS16969968,
    "CLOCK_3111T_C": CLOCK_RS1801260,
    "PER3_VNTR": PER3_VNTR,
    "APOE_e4": APOE_RS429358,
    "FKBP5_rs1360780": FKBP5_RS1360780,
}


# ---------------------------------------------------------------------------
# SNP Profile & Application
# ---------------------------------------------------------------------------

@dataclass
class MetabolicSNPProfile:
    """A user's SNP profile: which variants they carry and zygosity."""
    name: str = "default"
    # Map of snp_key -> genotype: "wildtype", "heterozygous", "homozygous"
    genotypes: Dict[str, str] = field(default_factory=dict)

    def get_variant_effects(self) -> Dict[str, float]:
        """
        Compute cumulative neurochemical effects from all genotypes.
        Returns a dict of {neurochemical: total_modifier}.
        """
        cumulative: Dict[str, float] = {}
        for snp_key, genotype in self.genotypes.items():
            variant = SNP_REGISTRY.get(snp_key)
            if not variant:
                continue
            # Weight by zygosity
            weight = {"wildtype": 0.0, "heterozygous": 0.5, "homozygous": 1.0}.get(genotype, 0.0)
            if weight == 0.0:
                continue
            effects = variant.to_effects_dict()
            for chem, mod in effects.items():
                cumulative[chem] = cumulative.get(chem, 0.0) + (mod * weight)
        return cumulative


def apply_snp_profile_to_neurochemistry(
    neurochemistry_state,
    snp_profile: MetabolicSNPProfile,
    dt: float = 1.0,
) -> None:
    """
    Apply SNP-derived modifiers to a NeurochemicalState in-place.

    This adjusts baseline neurotransmitter levels, receptor sensitivities,
    and temperament parameters based on the user's pharmacogenomic profile.
    """
    s = neurochemistry_state
    effects = snp_profile.get_variant_effects()

    # Apply direct neurochemical modifiers
    if "dopamine" in effects:
        s.dopamine = max(0.0, min(1.0, s.dopamine + effects["dopamine"] * dt))
        s.dopamine_mesolimbic = max(0.0, min(1.0, s.dopamine_mesolimbic + effects["dopamine"] * dt * 0.7))
        s.dopamine_mesocortical = max(0.0, min(1.0, s.dopamine_mesocortical + effects["dopamine"] * dt * 0.5))
    if "serotonin" in effects:
        s.serotonin = max(0.0, min(1.0, s.serotonin + effects["serotonin"] * dt))
    if "gaba" in effects:
        s.gaba = max(0.0, min(1.0, s.gaba + effects["gaba"] * dt))
    if "glutamate" in effects:
        s.glutamate = max(0.0, min(1.0, s.glutamate + effects["glutamate"] * dt))
    if "norepinephrine" in effects:
        s.norepinephrine = max(0.0, min(1.0, s.norepinephrine + effects["norepinephrine"] * dt))
    if "bdnf" in effects:
        s.bdnf = max(0.0, min(1.0, s.bdnf + effects["bdnf"] * dt))
    if "cortisol" in effects:
        s.cortisol = max(0.0, min(1.0, s.cortisol + effects["cortisol"] * dt))
    if "anandamide" in effects:
        s.anandamide = max(0.0, min(1.0, s.anandamide + effects["anandamide"] * dt))
    if "cytokine" in effects:
        s.cytokine_load = max(0.0, min(1.0, s.cytokine_load + effects["cytokine"] * dt))
    if "circadian" in effects:
        # Circadian modifiers affect SCN phase stability (modeled as sleep pressure offset)
        s.sleep_pressure = max(0.0, min(1.0, s.sleep_pressure + effects["circadian"] * dt))

    # Apply receptor sensitivity modifiers
    if "d1_sensitivity" in effects:
        s.d1_sensitivity = max(0.1, min(2.0, s.d1_sensitivity + effects["d1_sensitivity"] * dt))
    if "d2_sensitivity" in effects:
        s.d2_autoreceptor_inhibition = max(0.0, min(1.0, s.d2_autoreceptor_inhibition + effects["d2_sensitivity"] * dt))
    if "gaba_a_sensitivity" in effects:
        s.gaba_a_sensitivity = max(0.1, min(2.0, s.gaba_a_sensitivity + effects["gaba_a_sensitivity"] * dt))
    if "glun2b_sensitivity" in effects:
        s.glun2b_sensitivity = max(0.1, min(2.0, s.glun2b_sensitivity + effects["glun2b_sensitivity"] * dt))


def get_snp_presets() -> Dict[str, MetabolicSNPProfile]:
    """Return predefined SNP profiles for common pharmacogenomic phenotypes."""
    return {
        "default": MetabolicSNPProfile(name="default", genotypes={}),

        "high_dopamine": MetabolicSNPProfile(
            name="high_dopamine",
            genotypes={
                "COMT_Val158Met": "homozygous",   # Met/Met = slow clearance
                "DAT1_VNTR": "heterozygous",       # 9R = lower DAT
                "MAOA_uVNTR": "homozygous",        # Low activity
            },
        ),

        "low_dopamine": MetabolicSNPProfile(
            name="low_dopamine",
            genotypes={
                "COMT_Val158Met": "wildtype",      # Val/Val = fast clearance
                "DRD2_Taq1A": "homozygous",        # A1/A1 = reduced receptors
                "DAT1_VNTR": "wildtype",           # 10R = normal
            },
        ),

        "high_serotonin": MetabolicSNPProfile(
            name="high_serotonin",
            genotypes={
                "SLC6A4_5HTTLPR": "homozygous",    # S/S = less transporter
                "MAOA_uVNTR": "homozygous",        # Low activity
                "CYP2C19_PM": "homozygous",        # Poor metabolizer = higher SSRI effect proxy
            },
        ),

        "low_serotonin": MetabolicSNPProfile(
            name="low_serotonin",
            genotypes={
                "MTHFR_C677T": "homozygous",       # T/T = impaired synthesis
                "MTHFR_A1298C": "heterozygous",
                "SLC6A4_5HTTLPR": "wildtype",      # L/L = normal reuptake
                "HTR2A_T102C": "homozygous",       # C/C = reduced receptors
            },
        ),

        "anxiety_prone": MetabolicSNPProfile(
            name="anxiety_prone",
            genotypes={
                "SLC6A4_5HTTLPR": "homozygous",    # S/S
                "COMT_Val158Met": "wildtype",      # Val/Val = lower PFC DA
                "GABRA2_rs279858": "homozygous",   # Risk allele
                "GRIN2B_rs890": "heterozygous",
                "FKBP5_rs1360780": "homozygous",   # T/T = enhanced GR
            },
        ),

        "resilient": MetabolicSNPProfile(
            name="resilient",
            genotypes={
                "FAAH_C385A": "homozygous",        # A/A = high anandamide
                "COMT_Val158Met": "homozygous",    # Met/Met
                "BDNF_Val66Met": "wildtype",       # Val/Val
                "FKBP5_rs1360780": "wildtype",     # C/C = normal GR
            },
        ),

        "stress_vulnerable": MetabolicSNPProfile(
            name="stress_vulnerable",
            genotypes={
                "BDNF_Val66Met": "homozygous",     # Met/Met
                "FKBP5_rs1360780": "homozygous",   # T/T
                "MTHFR_C677T": "homozygous",       # T/T
                "APOE_e4": "heterozygous",
            },
        ),

        "reward_deficient": MetabolicSNPProfile(
            name="reward_deficient",
            genotypes={
                "DRD2_Taq1A": "homozygous",        # A1/A1
                "OPRM1_A118G": "homozygous",       # G/G
                "COMT_Val158Met": "wildtype",      # Val/Val
                "DAT1_VNTR": "wildtype",
            },
        ),

        "night_owl": MetabolicSNPProfile(
            name="night_owl",
            genotypes={
                "CLOCK_3111T_C": "homozygous",     # C/C = delayed phase
                "PER3_VNTR": "homozygous",         # 5R/5R
                "CHRNA5_rs16969968": "heterozygous",
            },
        ),

        "pain_sensitive": MetabolicSNPProfile(
            name="pain_sensitive",
            genotypes={
                "OPRM1_A118G": "homozygous",       # G/G = reduced opioid signaling
                "COMT_Val158Met": "homozygous",    # Met/Met = higher pain sensitivity
                "FAAH_C385A": "wildtype",          # C/C = normal FAAH
            },
        ),
    }


def format_snp_profile_for_dashboard(snp_profile: MetabolicSNPProfile) -> Dict:
    """Format an SNP profile for dashboard display."""
    effects = snp_profile.get_variant_effects()
    return {
        "name": snp_profile.name,
        "genotypes": snp_profile.genotypes,
        "cumulative_effects": effects,
        "affected_pathways": sorted(set(effects.keys())),
    }
