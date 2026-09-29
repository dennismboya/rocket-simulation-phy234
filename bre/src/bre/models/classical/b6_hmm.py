"""B6 — two-state hidden Markov model over a discrete risk state for sequential answers within a
session (PLAN.md section 4, classical baseline).

Model
-----
Within one session a subject's answers are ordered by ``position_in_session``. A latent risk
state ``s_t in {0, 1}`` (0 = "calm", 1 = "alarmed"; state 1 has the higher sell intercept by
construction, see the ordering constraint below) follows a Markov chain across the sell answers
of the session, and every sell answer is emitted from the current state through a logistic
regression on the row's features::

    P(s_1 = 1) = pi_1 = sigmoid(init_logit)
    P(s_{t+1} = s' | s_t = s) = T[s, s'],   T = [[p_00, 1 - p_00], [1 - p_11, p_11]],
                                            p_ss = sigmoid(trans_logit[s])
    P(sell_t | s_t = s, row t) = sigmoid(eta_{t,s}),
    eta_{t,s} = a_s + w_row[s] . f_t + w_x . x_{subject(t)}

``f_t`` are the row-level columns of :func:`bre.models.base.standard_features` (``loss``, the
per-position context one-hots ``ctx{k}={tag}``, ``order_flag``, ``tol_answered``, ``tol_yes``;
``d_row`` of them) with **state-specific** weights ``w_row (2, d_row)`` and intercepts
``a = (a0, a0 + exp(log_da))`` (the ordering constraint that removes label switching); the
subject covariates ``x`` (``d_x`` columns) enter through **shared** weights ``w_x`` because they
are constant within a subject and would otherwise be confounded with the state occupancy.
The transition matrix and the initial distribution are shared across subjects (2 + 1 free
scalars). Contexts enter additively per position and no pair parameters exist.

Filtering and the likelihood
----------------------------
:meth:`predict_proba` is the **filtered predictive** probability of each row given the earlier
sell answers of the same ``(subject, session)`` (only prior answers; never the row's own or
later ones)::

    P(sell_t | y_{<t}) = sum_s alpha_t(s) sigmoid(eta_{t,s}),   alpha_t = P(s_t = . | y_{<t})

computed by the forward algorithm (``jax.lax.scan`` over positions, all sequences batched, in
log space): ``alpha_1 = pi``; after observing ``y_t`` with weight ``w_t``,
``alpha_t(s) <- alpha_t(s) exp(w_t [y_t log p + (1 - y_t) log(1 - p)])`` normalized, then
``alpha_{t+1} = alpha_t T``. Rows outside ``sell_rows()`` (tolerance rows and others) sit in the
sequence, receive the predictive of the current state distribution, and do **not** update or
advance the chain. Since ``log P(y_1..y_T) = sum_t log P(y_t | y_{<t})``, the shared per-row
log-likelihood of :class:`bre.models.base.ModelBase` (Bernoulli on the filtered predictive) sums
exactly to the HMM's forward log-likelihood, so B6 stays comparable row by row with every other
model and its held-out NLL per response means the same thing.

The sequences are built from the rows *present in the table*: a row subset (a training split or
a held-out fold) is filtered on the answers it contains, in position order, as documented
behaviour of split evaluation for a sequential model.

Parameters and count
--------------------
``{"w_x": (d_x,), "w_row": (2, d_row), "a0": (), "log_da": (), "trans_logit": (2,),
"init_logit": ()}``; ``n_params = d_x + 2 d_row + 5`` (with the shared design ``d_x = 21``,
``d_row = 12``: 50). No per-subject block. Prior: ridge ``-(l2 / 2) (||w_x||^2 + ||w_row||^2)``
(default ``l2 = 1e-2``, as B1).

Initialization (restart folded in): weights ``N(0, 0.01^2)``, ``a0 = -0.5``, ``log_da = 0``
(state 1 one logit unit above state 0), ``trans_logit = (2, 2)`` (stay probability 0.88),
``init_logit = 0``, each plus ``N(0, 0.1^2)`` noise. Starting points, not fitted values.

Structural identities (``tests/test_b3_b5_b6.py``): with ``pi = (1, 0)`` and ``T = I`` (a very
negative ``init_logit`` and very large ``trans_logit``) the model is the plain logistic
regression of state 0; the row-wise log-likelihood sums to a numpy forward algorithm.
"""

from __future__ import annotations

import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
from jax.scipy.special import logsumexp

from bre.models.base import PROB_EPS, ModelBase, Params, standard_features
from bre.models.data import ModelData

INIT_SCALE = 0.01
"""Standard deviation of the initial weights."""
INIT_NOISE = 0.1
"""Standard deviation of the noise on the initial scalars."""
INIT_A0 = -0.5
INIT_TRANS_LOGIT = 2.0
L2_DEFAULT = 1e-2

N_STATES = 2


def sequence_layout(data: ModelData) -> dict[str, np.ndarray]:
    """Sequences of a table: rows grouped by ``(subject_idx, session_id)`` and sorted by
    ``position`` (stable, so ties keep frame order). Returns ``idx (n_seq, T)`` row indices with
    ``-1`` padding, ``valid (n_seq, T)``, ``sell (n_seq, T)`` (valid and a sell row) and
    ``seq_of_row (n,)`` / ``pos_of_row (n,)`` for scattering per-row outputs back."""
    n = data.n
    if n == 0:
        return {"idx": np.zeros((0, 0), dtype=np.int64), "valid": np.zeros((0, 0), dtype=bool), "sell": np.zeros((0, 0), dtype=bool), "seq_of_row": np.zeros(0, dtype=np.int64), "pos_of_row": np.zeros(0, dtype=np.int64)}
    key = pd.DataFrame({"s": data.subject_idx, "sess": data.session_id.astype(str), "pos": data.position, "row": np.arange(n)})
    key = key.sort_values(["s", "sess", "pos", "row"], kind="stable")
    codes = key.groupby(["s", "sess"], sort=False).ngroup().to_numpy()
    key = key.assign(seq=codes)
    key["t"] = key.groupby("seq").cumcount()
    n_seq = int(codes.max()) + 1
    T = int(key["t"].max()) + 1
    idx = np.full((n_seq, T), -1, dtype=np.int64)
    idx[key["seq"].to_numpy(), key["t"].to_numpy()] = key["row"].to_numpy()
    valid = idx >= 0
    sell_mask = data.sell_mask()
    sell = np.zeros_like(valid)
    sell[valid] = sell_mask[idx[valid]]
    seq_of_row = np.empty(n, dtype=np.int64)
    pos_of_row = np.empty(n, dtype=np.int64)
    seq_of_row[key["row"].to_numpy()] = key["seq"].to_numpy()
    pos_of_row[key["row"].to_numpy()] = key["t"].to_numpy()
    return {"idx": idx, "valid": valid, "sell": sell, "seq_of_row": seq_of_row, "pos_of_row": pos_of_row}


def transition_matrix(trans_logit) -> jnp.ndarray:
    """``T`` from the two stay-probability logits (module docstring)."""
    p = jax.nn.sigmoid(jnp.asarray(trans_logit, dtype=jnp.float64))
    return jnp.array([[p[0], 1.0 - p[0]], [1.0 - p[1], p[1]]])


def log_transition_matrix(trans_logit) -> jnp.ndarray:
    """``log T`` computed exactly in log space (``log p_ss = log_sigmoid(l_s)``,
    ``log (1 - p_ss) = log_sigmoid(-l_s)``), so a very large logit gives a genuinely absorbing
    state instead of a clipped leak."""
    l = jnp.asarray(trans_logit, dtype=jnp.float64)
    ls = jax.nn.log_sigmoid
    return jnp.array([[ls(l[0]), ls(-l[0])], [ls(-l[1]), ls(l[1])]])


def forward_filter(log_pi, log_T, eta, y, w, sell) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched forward algorithm. ``eta (n_seq, T, 2)`` state logits, ``y``/``w`` ``(n_seq, T)``,
    ``sell (n_seq, T)`` bool (rows that update the chain). Returns ``pred (n_seq, T)``, the
    filtered predictive ``P(sell_t | y_<t)`` per slot, and ``alpha (n_seq, T, 2)``, the filtered
    state distribution used at each slot."""
    eta_t = jnp.moveaxis(jnp.asarray(eta, dtype=jnp.float64), 1, 0)  # (T, n_seq, 2)
    y_t = jnp.moveaxis(jnp.asarray(y, dtype=jnp.float64), 1, 0)
    w_t = jnp.moveaxis(jnp.asarray(w, dtype=jnp.float64), 1, 0)
    obs_t = jnp.moveaxis(jnp.asarray(sell, dtype=bool), 1, 0)
    n_seq = eta_t.shape[1]
    log_alpha0 = jnp.broadcast_to(jnp.asarray(log_pi, dtype=jnp.float64)[None, :], (n_seq, N_STATES))

    def step(log_alpha, xs):
        eta_s, y_s, w_s, obs = xs
        p = jnp.clip(jax.nn.sigmoid(eta_s), PROB_EPS, 1.0 - PROB_EPS)  # (n_seq, 2)
        alpha = jnp.exp(log_alpha)
        pred = jnp.sum(alpha * p, axis=1)
        ll = w_s[:, None] * (y_s[:, None] * jnp.log(p) + (1.0 - y_s[:, None]) * jnp.log1p(-p))
        post = log_alpha + ll
        post = post - logsumexp(post, axis=1, keepdims=True)
        nxt = logsumexp(post[:, :, None] + log_T[None, :, :], axis=1)
        new = jnp.where(obs[:, None], nxt, log_alpha)
        return new, (pred, alpha)

    _, (pred, alpha) = lax.scan(step, log_alpha0, (eta_t, y_t, w_t, obs_t))
    return jnp.moveaxis(pred, 0, 1), jnp.moveaxis(alpha, 0, 1)


class B6(ModelBase):
    """Two-state HMM over a discrete risk state with logistic emissions (module docstring)."""

    name = "B6"
    family = "classical"

    def __init__(self, data: ModelData, l2: float = L2_DEFAULT) -> None:
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.l2 = float(l2)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_ctx_positions = int(data.n_ctx_positions)
        fm = standard_features(data)
        self.x_columns: tuple[str, ...] = tuple(data.X_columns)
        self.row_columns: tuple[str, ...] = tuple(c for c in fm.columns if c not in self.x_columns)
        self.d_x = len(self.x_columns)
        self.d_row = len(self.row_columns)
        self._cache: weakref.WeakKeyDictionary[ModelData, dict[str, Any]] = weakref.WeakKeyDictionary()

    # -- static arrays ----------------------------------------------------------------------------

    def arrays(self, data: ModelData) -> dict[str, Any]:
        """Per-table arrays (cached per ``ModelData`` object): the row-level feature block
        ``F_row (n, d_row)``, the covariate rows ``X_row (n, d_x)``, ``y``, ``w`` and the
        :func:`sequence_layout`."""
        arr = self._cache.get(data)
        if arr is None:
            if tuple(data.ctx_vocab) != self.ctx_vocab or data.n_ctx_positions != self.n_ctx_positions or tuple(data.X_columns) != self.x_columns:
                raise ValueError("B6: data has a different ctx_vocab / n_ctx_positions / covariate columns than the model was built on")
            fm = standard_features(data)
            col = {c: j for j, c in enumerate(fm.columns)}
            F_row = fm.F[:, [col[c] for c in self.row_columns]]
            arr = {
                "F_row": jnp.asarray(F_row, dtype=jnp.float64),
                "X_row": jnp.asarray(data.X[data.subject_idx], dtype=jnp.float64),
                "y": jnp.asarray(data.y, dtype=jnp.float64),
                "w": jnp.asarray(data.w, dtype=jnp.float64),
                **sequence_layout(data),
            }
            self._cache[data] = arr
        return arr

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        k_x, k_r, k_a, k_d, k_t, k_i = jax.random.split(key, 6)
        f64 = jnp.float64
        noise = lambda k, shape=(): INIT_NOISE * jax.random.normal(k, shape, dtype=f64)  # noqa: E731
        return {
            "w_x": INIT_SCALE * jax.random.normal(k_x, (self.d_x,), dtype=f64),
            "w_row": INIT_SCALE * jax.random.normal(k_r, (N_STATES, self.d_row), dtype=f64),
            "a0": INIT_A0 + noise(k_a),
            "log_da": noise(k_d),
            "trans_logit": INIT_TRANS_LOGIT + noise(k_t, (N_STATES,)),
            "init_logit": noise(k_i),
        }

    def intercepts(self, params: Params) -> jnp.ndarray:
        """``(a0, a0 + exp(log_da))``: state 1 has the higher sell intercept."""
        a0 = jnp.asarray(params["a0"], dtype=jnp.float64)
        return jnp.stack([a0, a0 + jnp.exp(jnp.asarray(params["log_da"], dtype=jnp.float64))])

    def state_logits(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n, 2)`` emission logits ``eta_{i,s}`` for every row and state."""
        arr = self.arrays(data)
        shared = arr["X_row"] @ jnp.asarray(params["w_x"], dtype=jnp.float64)  # (n,)
        per_state = arr["F_row"] @ jnp.asarray(params["w_row"], dtype=jnp.float64).T  # (n, 2)
        return per_state + shared[:, None] + self.intercepts(params)[None, :]

    def _filter(self, params: Params, data: ModelData) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(pred (n,), alpha (n, 2))`` in row order."""
        arr = self.arrays(data)
        eta = self.state_logits(params, data)
        idx = jnp.asarray(np.where(arr["valid"], arr["idx"], 0))
        eta_seq = eta[idx]  # (n_seq, T, 2)
        y_seq = arr["y"][idx]
        w_seq = arr["w"][idx]
        log_pi1 = jax.nn.log_sigmoid(jnp.asarray(params["init_logit"], dtype=jnp.float64))
        log_pi0 = jax.nn.log_sigmoid(-jnp.asarray(params["init_logit"], dtype=jnp.float64))
        log_T = log_transition_matrix(params["trans_logit"])
        pred, alpha = forward_filter(jnp.stack([log_pi0, log_pi1]), log_T, eta_seq, y_seq, w_seq, arr["sell"])
        s, t = jnp.asarray(arr["seq_of_row"]), jnp.asarray(arr["pos_of_row"])
        return pred[s, t], alpha[s, t]

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` filtered predictive ``P(sell_i | earlier sell answers of the session)``."""
        return self._filter(params, data)[0]

    def filtered_states(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n, 2)`` filtered state distribution ``P(s_t = . | y_<t)`` at every row."""
        return self._filter(params, data)[1]

    def log_prior(self, params: Params) -> jnp.ndarray:
        return -0.5 * self.l2 * (jnp.sum(jnp.asarray(params["w_x"]) ** 2) + jnp.sum(jnp.asarray(params["w_row"]) ** 2))

    def n_params(self, params: Params) -> int:
        """``d_x + 2 d_row + 5``."""
        return int(np.asarray(params["w_x"]).size + np.asarray(params["w_row"]).size + 5)

    # -- reporting --------------------------------------------------------------------------------

    def natural_params(self, params: Params) -> dict[str, Any]:
        """``pi_1``, the transition matrix, the two intercepts and the state weights by column."""
        a = np.asarray(self.intercepts(params))
        w = np.asarray(params["w_row"], dtype=np.float64)
        return {
            "pi_1": float(jax.nn.sigmoid(params["init_logit"])),
            "transition": np.asarray(transition_matrix(params["trans_logit"])).tolist(),
            "intercepts": a.tolist(),
            "w_x": {c: float(v) for c, v in zip(self.x_columns, np.asarray(params["w_x"]))},
            "w_row": {f"state{s}": {c: float(v) for c, v in zip(self.row_columns, w[s])} for s in range(N_STATES)},
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: mean filtered occupancy of state 1 over the subject's sell rows."""
        alpha = np.asarray(self.filtered_states(params, data))[:, 1]
        sell = data.sell_mask()
        occ = pd.Series(alpha[sell]).groupby(data.subject_idx[sell]).mean()
        out = pd.DataFrame({"subject_id": data.subject_ids, "state1_occupancy": np.nan})
        out.loc[occ.index.to_numpy(), "state1_occupancy"] = occ.to_numpy()
        return out


__all__ = ["B6", "INIT_A0", "INIT_TRANS_LOGIT", "L2_DEFAULT", "N_STATES", "forward_filter", "log_transition_matrix", "sequence_layout", "transition_matrix"]
