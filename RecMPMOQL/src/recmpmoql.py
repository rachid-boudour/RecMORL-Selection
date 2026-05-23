"""
RecMPMOQL: Recommendation-Driven Multi-Policy Multi-Objective Q-Learning
for QoS-Aware IoT Service Composition in Multi-Cloud Environments

Paper implementation covering:
  - MOMDP environment (state, action, reward, transition, termination)
  - MPMOQL agent (multi-objective Q-learning with epsilon-greedy)
  - SASRec model (Self-Attention Sequential Recommendation)
  - RecMPMOQL hybrid algorithm (MPMOQL + SASRec dynamic QoS)
  - Lexicographic ordering of Pareto solutions
"""

import numpy as np
import random
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# =============================================================================
# 1.  DATA STRUCTURES
# =============================================================================

class Service:
    """
    Represents a single candidate IoT service offered by a cloud provider.

    Attributes
    ----------
    service_idx : int   – abstract service index  (i, 1-based)
    cloud_idx   : int   – cloud provider index     (j, 0-based)
    static_qos  : np.ndarray shape (11,)  – normalized static QoS attributes  ∈ [0,1]
    dynamic_qos : np.ndarray shape (2,)   – [CCRS, ICRS]  ∈ [0,1]
    """

    def __init__(self,
                 service_idx: int,
                 cloud_idx: int,
                 static_qos: np.ndarray,
                 dynamic_qos: Optional[np.ndarray] = None):
        self.service_idx = service_idx          # i  (1 … K)
        self.cloud_idx   = cloud_idx            # j  (0 … M-1)
        self.static_qos  = np.asarray(static_qos,  dtype=float)   # shape (11,)
        self.dynamic_qos = (np.asarray(dynamic_qos, dtype=float)  # shape (2,)
                            if dynamic_qos is not None
                            else np.zeros(2))

    # Convenience: full reward vector for objective o
    def qos_value(self, obj_idx: int) -> float:
        """Return the QoS value for objective obj_idx (0-10 static, 11-12 dynamic)."""
        all_qos = np.concatenate([self.static_qos, self.dynamic_qos])
        return float(all_qos[obj_idx])

    def __repr__(self):
        return f"Service(i={self.service_idx}, cloud={self.cloud_idx})"


# =============================================================================
# 2.  MOMDP ENVIRONMENT
# =============================================================================

class MultiCloudEnv:
    """
    Multi-Objective Markov Decision Process (MOMDP) for IoT service composition.

    State   : (current_service_idx, current_cloud_idx)
    Actions : 0 … M-1  → pick cloud j+k+1 for current abstract service
              M         → progression action (confirm & move to next service)
    Reward  : vector of 13 objectives = 11 static + 2 dynamic QoS attributes
    """

    N_STATIC  = 11   # static QoS attributes
    N_DYNAMIC = 2    # dynamic QoS attributes (CCRS, ICRS)
    N_OBJ     = N_STATIC + N_DYNAMIC   # total objectives = 13

    def __init__(self,
                 candidate_services: Dict[int, List[Service]],
                 qos_min: np.ndarray,
                 qos_max: np.ndarray):
        """
        Parameters
        ----------
        candidate_services : dict {service_idx → list of Service objects}
                             Keys are abstract service indices 1 … K.
        qos_min / qos_max  : shape (N_OBJ,) – user-defined QoS constraint bounds.
        """
        self.candidates  = candidate_services          # {i: [Service, …]}
        self.K           = len(candidate_services)     # number of abstract services
        self.qos_min     = np.asarray(qos_min, dtype=float)
        self.qos_max     = np.asarray(qos_max, dtype=float)

        # Episode variables
        self.current_service_idx  = 1      # i  (1 … K)
        self.current_cloud_idx    = 0      # j  (0-based)
        self.selected_services: List[Service] = []
        self.accumulated_qos     = np.zeros(self.N_OBJ)
        self.done                = False

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def _state(self) -> Tuple[int, int]:
        """Return (service_idx, cloud_idx)."""
        return (self.current_service_idx, self.current_cloud_idx)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        """Reset to initial state s0 = (service=1, cloud=0)."""
        self.current_service_idx = 1
        self.current_cloud_idx   = 0
        self.selected_services   = []
        self.accumulated_qos     = np.zeros(self.N_OBJ)
        self.done                = False
        return self._state()

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    def action_space(self) -> List[int]:
        """
        Actions 0 … M-1 : select cloud (current_cloud + k + 1) % M
        Action  M        : progression (confirm current service, move to next)
        """
        M = len(self.candidates[self.current_service_idx])
        return list(range(M + 1))   # M cloud-selection + 1 progression

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[Tuple[int,int], np.ndarray, bool]:
        """
        Execute action, return (next_state, reward_vector, done).

        Cloud-selection action (0 … M-1):
            Switch current_cloud_idx to (current_cloud_idx + action + 1) % M
            Reward = QoS vector of the newly pointed service.

        Progression action (M):
            Confirm current service, accumulate QoS, move to next abstract service.
            At terminal step: return final normalised accumulated reward.
        """
        assert not self.done, "Episode already finished; call reset()."

        M          = len(self.candidates[self.current_service_idx])
        action_M   = M   # progression action index

        reward = np.zeros(self.N_OBJ)

        if action < action_M:
            # ── Cloud-selection action ──────────────────────────────────
            new_cloud = (self.current_cloud_idx + action + 1) % M
            self.current_cloud_idx = new_cloud
            svc = self.candidates[self.current_service_idx][self.current_cloud_idx]
            reward = np.concatenate([svc.static_qos, svc.dynamic_qos])

        else:
            # ── Progression action ──────────────────────────────────────
            svc = self.candidates[self.current_service_idx][self.current_cloud_idx]
            step_reward = np.concatenate([svc.static_qos, svc.dynamic_qos])
            self.accumulated_qos += step_reward
            self.selected_services.append(svc)

            # Check QoS constraint violation (termination condition 2)
            avg_qos = self.accumulated_qos / len(self.selected_services)
            if self._violates_constraints(avg_qos):
                self.done = True
                reward    = avg_qos   # penalised terminal reward
            elif self.current_service_idx == self.K:
                # All K services selected (termination condition 1)
                self.done = True
                reward    = self.accumulated_qos / self.K   # normalised
            else:
                # Advance to next abstract service
                self.current_service_idx += 1
                self.current_cloud_idx    = 0
                reward                    = step_reward

        return self._state(), reward, self.done

    # ------------------------------------------------------------------
    # Constraint check
    # ------------------------------------------------------------------
    def _violates_constraints(self, avg_qos: np.ndarray) -> bool:
        """True if any QoS attribute is outside user-defined bounds."""
        return bool(np.any(avg_qos < self.qos_min) or np.any(avg_qos > self.qos_max))

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    def get_final_reward(self) -> np.ndarray:
        """Return the final accumulated reward split into static & dynamic."""
        K = len(self.selected_services)
        if K == 0:
            return np.zeros(self.N_OBJ)
        return self.accumulated_qos / K


# =============================================================================
# 3.  SASRec  –  Self-Attention Sequential Recommendation
# =============================================================================

class SASRec:
    """
    Lightweight NumPy implementation of SASRec for next-service prediction.

    Supports two interaction histories:
      - global (collective) → CCRS
      - per-user            → ICRS

    Architecture: single self-attention head + feed-forward layer.
    """

    def __init__(self,
                 n_services: int,
                 embed_dim: int  = 32,
                 max_seq_len: int = 50,
                 beta: float     = 1e-6,
                 seed: int       = 42):
        rng = np.random.default_rng(seed)
        self.n   = n_services
        self.d   = embed_dim
        self.L   = max_seq_len
        self.beta = beta

        # Item embeddings  E ∈ R^{n × d}
        self.E = rng.standard_normal((n_services, embed_dim)) * 0.01
        # Positional embeddings  P ∈ R^{L × d}
        self.P = rng.standard_normal((max_seq_len, embed_dim)) * 0.01

        # Projection matrices
        self.Wq = rng.standard_normal((embed_dim, embed_dim)) * 0.01
        self.Wk = rng.standard_normal((embed_dim, embed_dim)) * 0.01
        self.Wv = rng.standard_normal((embed_dim, embed_dim)) * 0.01
        self.W_out = rng.standard_normal((embed_dim, n_services)) * 0.01

        # Interaction histories  {service_id: count}
        self.global_history: Dict[int, int] = defaultdict(int)
        self.user_history:   Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def record_interaction(self, service_id: int, user_id: Optional[str] = None):
        """Log a service selection to the interaction histories."""
        self.global_history[service_id] += 1
        if user_id is not None:
            self.user_history[user_id][service_id] += 1

    # ------------------------------------------------------------------
    # Cold-start scores (first abstract service)
    # ------------------------------------------------------------------
    def cold_start_ccrs(self, candidate_ids: List[int]) -> np.ndarray:
        """CCRS_0 based on global frequency (Eq. before SASRec forward pass)."""
        freqs = np.array([self.global_history[s] + self.beta for s in candidate_ids])
        return freqs / freqs.max()

    def cold_start_icrs(self, candidate_ids: List[int],
                        user_id: str) -> np.ndarray:
        """ICRS_0 based on individual user frequency."""
        hist = self.user_history[user_id]
        freqs = np.array([hist[s] + self.beta for s in candidate_ids])
        return freqs / freqs.max()

    # ------------------------------------------------------------------
    # Forward pass (attention)
    # ------------------------------------------------------------------
    def _forward(self, session: List[int]) -> np.ndarray:
        """
        Given a session (list of service IDs), return the hidden state h_t
        at the last position via causal self-attention.
        Returns h_t ∈ R^d.
        """
        n = min(len(session), self.L)
        ids = session[-n:]

        # X = item_embed + position_embed  (n × d)
        X = self.E[ids] + self.P[:n]

        # Q, K, V
        Q = X @ self.Wq   # (n × d)
        K = X @ self.Wk
        V = X @ self.Wv

        # Causal attention mask
        scale  = math.sqrt(self.d)
        scores = (Q @ K.T) / scale   # (n × n)
        mask   = np.triu(np.full_like(scores, -1e9), k=1)
        attn   = self._softmax(scores + mask)  # (n × n)
        H      = attn @ V                      # (n × d)

        return H[-1]   # h_t at last position

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    # ------------------------------------------------------------------
    # Score candidate services given a session
    # ------------------------------------------------------------------
    def score_ccrs(self, session: List[int],
                   candidate_ids: List[int]) -> np.ndarray:
        """
        CCRS(S_i^j) = σ( h_t^T · e(S_i^j) )  for each candidate.
        """
        if not session:
            return self.cold_start_ccrs(candidate_ids)
        h = self._forward(session)
        emb = self.E[candidate_ids]          # (M × d)
        raw = emb @ h                        # (M,)
        return self._sigmoid(raw)

    def score_icrs(self, session: List[int],
                   candidate_ids: List[int],
                   user_id: str) -> np.ndarray:
        """
        ICRS^u(S_i^j) = σ( (h^u_t)^T · e(S_i^j) )
        We simulate per-user hidden state by perturbing the global session
        with the user's personal history session.
        """
        if not session:
            return self.cold_start_icrs(candidate_ids, user_id)
        # Merge global session with user-specific history for personalisation
        user_hist_ids = list(self.user_history[user_id].keys())
        personalised  = (user_hist_ids + session)[-self.L:]
        h_u = self._forward(personalised)
        emb = self.E[candidate_ids]
        raw = emb @ h_u
        return self._sigmoid(raw)


# =============================================================================
# 4.  MPMOQL AGENT
# =============================================================================

class MPMOQLAgent:
    """
    Multi-Policy Multi-Objective Q-Learning agent.

    Maintains a Q-table  Q̂(s, a, o)  for each objective o.
    Uses ε-greedy with linear scalarization for action selection.
    """

    def __init__(self,
                 n_objectives: int,
                 alpha: float   = 0.1,
                 gamma: float   = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float  = 0.05,
                 epsilon_decay: float = 0.995,
                 weight_vector: Optional[np.ndarray] = None):
        self.n_obj        = n_objectives
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay

        # Equal weights if none provided
        self.weights = (np.asarray(weight_vector, dtype=float)
                        if weight_vector is not None
                        else np.ones(n_objectives) / n_objectives)

        # Q-table:  { (state, action) : np.ndarray(n_obj) }
        self.Q: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.random.uniform(-0.01, 0.01, n_objectives)
        )

    def _q_scalar(self, state: tuple, action: int) -> float:
        """Scalarized Q-value = w · Q̂(s, a, ·)"""
        return float(np.dot(self.weights, self.Q[(state, action)]))

    def select_action(self, state: tuple, available_actions: List[int]) -> int:
        """ε-greedy action selection using scalarized Q-values."""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        return max(available_actions,
                   key=lambda a: self._q_scalar(state, a))

    def update(self,
               state:      tuple,
               action:     int,
               reward_vec: np.ndarray,
               next_state: tuple,
               next_actions: List[int],
               done:       bool):
        """
        For each objective o:
        Q̂(s,a,o) ← Q̂(s,a,o) + α [ r(o) + γ · Q̂(s', greedy_a', o) - Q̂(s,a,o) ]
        """
        if done or not next_actions:
            target = reward_vec
        else:
            best_next = max(next_actions,
                            key=lambda a: self._q_scalar(next_state, a))
            target = reward_vec + self.gamma * self.Q[(next_state, best_next)]

        td_error = target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def apply_preference_weights(self, prefs: Dict[int, float]):
        """
        Temporarily adjust objective weights based on qualitative user preferences.
        prefs = {obj_index: multiplier}  e.g. {0: 3.0, 2: 0.25}
        """
        w = self.weights.copy()
        for idx, mult in prefs.items():
            w[idx] *= mult
        # Re-normalise
        self.weights = w / w.sum()


# =============================================================================
# 5.  RecMPMOQL  –  Main Algorithm
# =============================================================================

class RecMPMOQL:
    """
    RecMPMOQL: Recommendation-driven Multi-Policy Multi-Objective Q-Learning.

    Integrates:
      - MOMDP environment
      - MPMOQL agent
      - SASRec dynamic QoS updater

    Parameters
    ----------
    env           : MultiCloudEnv
    agent         : MPMOQLAgent
    sasrec        : SASRec
    n_episodes    : int
    user_id       : str  – for ICRS personalisation
    pref_weights  : dict {obj_idx: multiplier}  – qualitative user preferences
    """

    def __init__(self,
                 env:           MultiCloudEnv,
                 agent:         MPMOQLAgent,
                 sasrec:        SASRec,
                 n_episodes:    int = 500,
                 user_id:       str = "default_user",
                 pref_weights:  Optional[Dict[int, float]] = None):
        self.env          = env
        self.agent        = agent
        self.sasrec       = sasrec
        self.n_episodes   = n_episodes
        self.user_id      = user_id
        self.pref_weights = pref_weights or {}

        self.pareto_front: List[Tuple[List[Service], np.ndarray]] = []

    # ------------------------------------------------------------------
    # Dynamic QoS update (SASRec)
    # ------------------------------------------------------------------
    def _update_dynamic_qos(self, session: List[int]):
        """
        For every candidate service of the current abstract service,
        recompute CCRS and ICRS via SASRec and update their dynamic_qos field.
        """
        i          = self.env.current_service_idx
        candidates = self.env.candidates[i]
        c_ids      = [s.cloud_idx for s in candidates]

        ccrs_vals = self.sasrec.score_ccrs(session, c_ids)
        icrs_vals = self.sasrec.score_icrs(session, c_ids, self.user_id)

        for k, svc in enumerate(candidates):
            svc.dynamic_qos = np.array([ccrs_vals[k], icrs_vals[k]])

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------
    def _run_episode(self) -> Tuple[List[Service], np.ndarray]:
        state   = self.env.reset()
        session: List[int] = []   # sequence of selected service cloud indices
        done    = False

        # Apply qualitative preference weights (temporary)
        original_weights = self.agent.weights.copy()
        if self.pref_weights:
            self.agent.apply_preference_weights(self.pref_weights)

        while not done:
            # ── 5a. Update dynamic QoS for current abstract service ──
            self._update_dynamic_qos(session)

            # ── 5b. Choose action (ε-greedy) ────────────────────────
            avail   = self.env.action_space()
            action  = self.agent.select_action(state, avail)

            # ── 5c. Execute action ───────────────────────────────────
            next_state, reward_vec, done = self.env.step(action)

            # ── 5d. Q-value update for each objective ────────────────
            next_avail = [] if done else self.env.action_space()
            self.agent.update(state, action, reward_vec,
                              next_state, next_avail, done)

            # ── 5e. Progression action → record to session ───────────
            M = len(self.env.candidates[self.env.current_service_idx
                                        if not done
                                        else self.env.current_service_idx])
            if action == M:   # progression action
                if self.env.selected_services:
                    last = self.env.selected_services[-1]
                    session.append(last.cloud_idx)
                    self.sasrec.record_interaction(last.cloud_idx, self.user_id)

            state = next_state

        # Restore original weights
        self.agent.weights = original_weights

        final_reward = self.env.get_final_reward()
        selected     = list(self.env.selected_services)
        return selected, final_reward

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def run(self) -> List[Tuple[List[Service], np.ndarray]]:
        """
        Run n_episodes episodes.
        Returns the accumulated Pareto front (non-dominated solutions).
        """
        for ep in range(self.n_episodes):
            selected, reward = self._run_episode()
            self.agent.decay_epsilon()

            if len(selected) == self.env.K:   # complete composition only
                self._update_pareto(selected, reward)

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{self.n_episodes} | "
                      f"ε={self.agent.epsilon:.3f} | "
                      f"Pareto size={len(self.pareto_front)}")

        return self.pareto_front

    # ------------------------------------------------------------------
    # Pareto front management
    # ------------------------------------------------------------------
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if solution a Pareto-dominates solution b (maximisation)."""
        return bool(np.all(a >= b) and np.any(a > b))

    def _update_pareto(self, selected: List[Service], reward: np.ndarray):
        """Add solution to Pareto front if non-dominated."""
        dominated_idxs = []
        for idx, (_, r) in enumerate(self.pareto_front):
            if self._dominates(r, reward):
                return   # new solution is dominated → discard
            if self._dominates(reward, r):
                dominated_idxs.append(idx)

        # Remove solutions dominated by the new one
        self.pareto_front = [self.pareto_front[i]
                             for i in range(len(self.pareto_front))
                             if i not in dominated_idxs]
        self.pareto_front.append((selected, reward.copy()))


# =============================================================================
# 6.  LEXICOGRAPHIC ORDERING
# =============================================================================

def lexicographic_order(
        pareto_front: List[Tuple[List[Service], np.ndarray]],
        priority_objectives: List[int]
) -> List[Tuple[List[Service], np.ndarray]]:
    """
    Order a Pareto front using lexicographic dominance.

    priority_objectives : ordered list of objective indices to compare,
                          e.g. [0, 2] means compare obj-0 first, then obj-2.
                          Objectives NOT in the list are ignored.

    Returns the sorted list (best first).
    """
    def lex_key(item):
        _, reward = item
        return tuple(-reward[o] for o in priority_objectives)   # negate for max

    return sorted(pareto_front, key=lex_key)


# =============================================================================
# 7.  UTILITIES
# =============================================================================

def normalize_qos(data: np.ndarray,
                  benefit_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Min-max normalize each QoS column to [0,1].

    benefit_mask : boolean array of shape (n_attributes,)
                   True  → higher is better (benefit criterion)
                   False → lower is better  (cost criterion, invert after norm)
    """
    lo  = data.min(axis=0)
    hi  = data.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    out = (data - lo) / rng

    if benefit_mask is not None:
        cost_mask        = ~np.asarray(benefit_mask, dtype=bool)
        out[:, cost_mask] = 1.0 - out[:, cost_mask]

    return out


def generate_synthetic_dataset(
        n_abstract_services: int = 5,
        n_clouds:             int = 20,
        n_static_attrs:       int = 11,
        seed:                 int = 0
) -> Dict[int, List[Service]]:
    """
    Generate a random candidate-service dataset for quick testing.

    Returns dict {service_idx: [Service, …]} with n_clouds services per slot.
    """
    rng = np.random.default_rng(seed)
    dataset: Dict[int, List[Service]] = {}

    for i in range(1, n_abstract_services + 1):
        raw   = rng.random((n_clouds, n_static_attrs))
        normd = normalize_qos(raw)
        dataset[i] = [
            Service(service_idx=i, cloud_idx=j, static_qos=normd[j])
            for j in range(n_clouds)
        ]
    return dataset


# =============================================================================
# 8.  DEMO  /  ENTRY POINT
# =============================================================================

def main():
    print("=" * 60)
    print("RecMPMOQL  –  Multi-Cloud IoT Service Composition")
    print("=" * 60)

    # ── Hyper-parameters ──────────────────────────────────────────────
    K         = 5      # number of abstract services
    M         = 20     # cloud providers
    N_EP      = 200    # training episodes
    N_OBJ     = MultiCloudEnv.N_OBJ   # 13 objectives

    # ── 1. Synthetic dataset ─────────────────────────────────────────
    candidates = generate_synthetic_dataset(
        n_abstract_services=K, n_clouds=M, seed=42
    )

    # ── 2. User-defined QoS constraints (all objectives in [0,1]) ────
    qos_min = np.zeros(N_OBJ)          # no lower bound (all 0)
    qos_max = np.ones(N_OBJ)           # no upper bound (all 1)

    # ── 3. Build environment ─────────────────────────────────────────
    env = MultiCloudEnv(
        candidate_services=candidates,
        qos_min=qos_min,
        qos_max=qos_max
    )

    # ── 4. Build MPMOQL agent ────────────────────────────────────────
    # Qualitative preferences: obj-0 (battery) is very important (×3),
    #                          obj-2 (security) is important (×2),
    #                          obj-5 (cost) is discouraged (÷4)
    pref_weights = {0: 3.0, 2: 2.0, 5: 0.25}

    agent = MPMOQLAgent(
        n_objectives=N_OBJ,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99
    )

    # ── 5. Build SASRec recommender ──────────────────────────────────
    # Total unique service IDs = M (cloud indices 0 … M-1)
    sasrec = SASRec(n_services=M, embed_dim=32, max_seq_len=K * M)

    # ── 6. RecMPMOQL training ────────────────────────────────────────
    algo = RecMPMOQL(
        env=env,
        agent=agent,
        sasrec=sasrec,
        n_episodes=N_EP,
        user_id="user_001",
        pref_weights=pref_weights
    )

    pareto_front = algo.run()

    # ── 7. Lexicographic ordering ────────────────────────────────────
    # Priority: obj-0 (battery) ≻ obj-2 (security); cost (obj-5) ignored
    ordered = lexicographic_order(pareto_front, priority_objectives=[0, 2])

    # ── 8. Print results ─────────────────────────────────────────────
    print(f"\nPareto front size: {len(ordered)}")
    print("\nTop-3 solutions (lexicographic order: battery ≻ security):")
    for rank, (services, reward) in enumerate(ordered[:3], 1):
        print(f"\n  Rank {rank}:")
        for svc in services:
            print(f"    {svc}  static_qos={np.round(svc.static_qos[:4], 3)}…"
                  f"  dynamic_qos={np.round(svc.dynamic_qos, 3)}")
        print(f"  Avg QoS (obj 0,2,5) = "
              f"{reward[0]:.3f}, {reward[2]:.3f}, {reward[5]:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
