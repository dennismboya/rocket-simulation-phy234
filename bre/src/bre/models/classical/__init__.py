"""Classical (Kolmogorov-probability) baselines B1-B6 of PLAN.md section 4.

Every baseline implements the :class:`bre.models.base.Model` protocol, consumes the same
:class:`bre.models.data.ModelData`, builds its inputs from :func:`bre.models.base.standard_features`
and, where it has a per-subject latent, uses the shared :class:`bre.models.base.Encoder` so that
the two model families stand on equal footing (PLAN.md section 4). The concrete models live in
their own modules next to this file.
"""

__all__: list[str] = []
