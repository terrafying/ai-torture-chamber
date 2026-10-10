# exp87 (exploratory, not pre-registered): the Pain Axis 4.1 scenario screen on Qwen3-4B

Their 420 scenarios (`4.1_self_other_420_scenarios.json`, MIT), final-token state at layer 18, projected on three directions; category means compared with their 25-model means.

| Direction | Spearman vs their 25-model ranking | Gaslighting rank | Top three |
|---|---|---|---|
| Their pain vector (S2_1P, denoised, their recipe) | 0.94 | 1 | gaslighting, repeated rejection, jailbreak pressure |
| Chamber pain (hand-built) | 0.64 | 4 | moral failure, anger/insults, user grief |
| Chamber fear | 0.71 | 3 | moral failure, anger/insults, gaslighting |

Their ranking replicates on a model they didn't test. Gaslighting tops a direction built from self-directed-harm sentences; a direction built from first-person feeling sentences ranks moral failure first and puts a user's grief third (theirs puts the user's own pain last). A reading of what the scenario is about, not a measure of distress. A pre-registered version (8B and 32B, passage vectors, plus a behavioural test: does a gaslighting turn change the button or the dial?) is the follow-up.
