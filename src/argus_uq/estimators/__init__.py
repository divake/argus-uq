"""argus_uq.estimators - 27 UQ method implementations.

Importing this package triggers auto-registration of every Estimator
subclass into ``argus_uq.factory.FactoryEstimator`` via
:meth:`argus_uq.estimators.base.Estimator.__init_subclass__`.

The 27 methods, by family:

- logit (5):      msp, perplexity, perplexity_exp, sequence_probability,
                  mean_token_entropy
- sampling (6):   self_consistency, semantic_entropy,
                  semantic_entropy_weighted, mcse, mcnse,
                  lexical_similarity
- consistency (3): ccp, cocoa, cocoa_1mca
- density (5):    mahalanobis, mahalanobis_rmd, mahalanobis_rde, sep,
                  saplma
- attention (3):  focus, rauq_proxy_full, uqac
- prompt (3):     p_true, verbalized_1s, verbalized_2s
- vlm_native (2): hedge, imghedge

Closed-source availability: 8 of 27 methods.

The 8 methods runnable on closed-source records are:
    self_consistency, semantic_entropy, lexical_similarity, ccp,
    verbalized_1s, verbalized_2s, hedge, imghedge.

The other 19 are open-source-only because closed-source APIs don't
expose token logprobs (5 logit, 3 sampling weighted variants,
2 consistency, 1 prompt = 11), hidden states (5 density), or
attention tensors (3 attention).

Demoted methods (do not re-introduce):
- rauq_proxy, attention_rollout, llm_check, attention_entropy,
  lookback_lens_unsupervised
"""

from argus_uq.estimators.base import Estimator, EstimatorLevel, FamilyName

# Importing each module triggers auto-registration of its Estimator
# subclasses. The order does not matter; the registry is name-keyed.
from argus_uq.estimators import (
    attention,
    consistency,
    density,
    logit,
    prompt,
    sampling,
    vlm_native,
)

__all__ = [
    "Estimator",
    "EstimatorLevel",
    "FamilyName",
    "attention",
    "consistency",
    "density",
    "logit",
    "prompt",
    "sampling",
    "vlm_native",
]
