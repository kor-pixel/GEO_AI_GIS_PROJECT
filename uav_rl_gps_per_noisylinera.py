# =============================
# 1. DQN NETWORK (MLP)
# =============================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque, namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

# =============================
class NoisyLinear(nn.Module):
    """
    Noisy linear layer (Fortunato et al., 2017)
    - W = μ_w + σ_w ⊙ ε_w
    - b = μ_b + σ_b ⊙ ε_b
    - ε is resampled every time reset_noise() is called
    """

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Parameters "mean" and "sigma"
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffer (not learnable)
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize according to NoisyNet proposal
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # in eval, can use only μ (or keep noise as you prefer)
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN with NoisyNet:
      - shared feature: FC + ReLU
      - value stream: NoisyLinear
      - advantage stream: NoisyLinear
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_noisy = NoisyLinear(hidden_dim, 1)

        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.adv_noisy = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = torch.relu(self.value_fc(f))
        v = self.value_noisy(v)

        a = torch.relu(self.adv_fc(f))
        a = self.adv_noisy(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q

    def reset_noise(self):
        # Reset noise for all NoisyLinear layers in the network
        self.value_noisy.reset_noise()
        self.adv_noisy.reset_noise()




# =============================
# 2. REPLAY BUFFER
# =============================

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay 
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        """Add a new transition with priority = current max_priority (or 1.0 if buffer is empty)."""
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Return: batch Transition, indices, weights (tensor)."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        # P(i) ~ p_i^alpha
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # fallback if all prios = 0
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize max = 1

        batch = Transition(*zip(*samples))
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return batch, indices, weights_t

    def update_priorities(self, indices, priorities):
        """Update priorities for samples used in a batch."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


# =============================
# 3. ENVIRONMENT FOR ROUTING
# =============================

class UAVRoutingEnvTorch:
    def __init__(
        self,
        weighted_graph: dict,
        max_hops: int,
        start_id: str,
        goal_id: str,
        node_positions: dict,          # node_id -> (lat, lon)
        target_bw_mbps: float = 5.0,
        max_delay_ms: float = 400.0,
        max_jitter_ms: float = 30.0,
        max_loss: float = 0.01,
        # weights cho reward
        w_bw: float = 1.0,
        w_delay: float = 1.0,
        w_jitter: float = 0.5,
        w_loss: float = 2.0,
        w_hop: float = 0.2,
        w_progress: float = 1.0,       # reward/penalty for moving closer/farther from the goal
        goal_bonus: float = 50.0,
        fail_penalty: float = -30.0,
    ):
        if start_id not in weighted_graph:
            raise ValueError(f"start_id={start_id} not in weighted_graph")
        if goal_id not in weighted_graph:
            raise ValueError(f"goal_id={goal_id} not in weighted_graph")

        self.weighted_graph = weighted_graph
        self.max_hops = max_hops
        self.start_id = start_id
        self.goal_id = goal_id

        # QoS scale
        self.target_bw_mbps = target_bw_mbps
        self.max_delay_ms = max_delay_ms
        self.max_jitter_ms = max_jitter_ms
        self.max_loss = max_loss

        # reward weights
        self.w_bw = w_bw
        self.w_delay = w_delay
        self.w_jitter = w_jitter
        self.w_loss = w_loss
        self.w_hop = w_hop
        self.w_progress = w_progress
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty

        # mapping node_id <-> index
        self.node_ids = sorted(list(weighted_graph.keys()))
        self.id2idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx2id = {i: nid for nid, i in self.id2idx.items()}
        self.num_nodes = len(self.node_ids)

        if start_id not in self.id2idx or goal_id not in self.id2idx:
            raise ValueError("start_id or goal_id not in node_ids of weighted_graph")

        self.start_idx = self.id2idx[start_id]
        self.goal_idx = self.id2idx[goal_id]

        # node coordinates: (lat, lon) -> can be seen as (y, x)
        self.coords = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for nid, idx in self.id2idx.items():
            if nid not in node_positions:
                raise ValueError(f"node_positions does not have coordinates for node_id={nid}")
            lat, lon = node_positions[nid]
            self.coords[idx, 0] = lat
            self.coords[idx, 1] = lon

        # Euclidean distance to goal for each node
        goal_lat, goal_lon = self.coords[self.goal_idx]
        dist_list = []
        for i in range(self.num_nodes):
            lat, lon = self.coords[i]
            dx = goal_lat - lat
            dy = goal_lon - lon
            dist = (dx ** 2 + dy ** 2) ** 0.5
            dist_list.append(dist)
        self.dist_to_goal = np.array(dist_list, dtype=np.float32)
        self.max_dist_to_goal = float(self.dist_to_goal.max()) if self.dist_to_goal.size > 0 else 1.0

        # build edge_dict và neighbors_idx
        self.edge_dict = {}
        self.neighbors_idx = {self.id2idx[u_id]: [] for u_id in self.node_ids}
        for u_id, nbrs in weighted_graph.items():
            u_idx = self.id2idx[u_id]
            for (v_id, dist, delay_ms, bw, jitter_ms, loss) in nbrs:
                v_idx = self.id2idx[v_id]
                self.edge_dict[(u_idx, v_idx)] = (dist, delay_ms, bw, jitter_ms, loss)
                self.neighbors_idx[u_idx].append(v_idx)

        # State: one-hot(node_idx) + hops_norm + cum_delay_norm + e2e_loss_norm + dist_to_goal_norm
        self.state_dim = self.num_nodes + 4
        self.action_dim = self.num_nodes

        # internal variables
        self.current_idx: int | None = None
        self.hops_used: int = 0
        self.cum_delay: float = 0.0
        self.cum_jitter: float = 0.0
        self.success_prob: float = 1.0  # P_success = ∏ (1 - loss_i)

    # -------------------------
    # STATE ENCODING
    # -------------------------

    def _encode_state(self):
        """
        [one_hot(node_idx),
         hops_used / max_hops,
         cum_delay / max_delay_ms,
         e2e_loss / max_loss,
         dist_to_goal / max_dist_to_goal]
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[self.current_idx] = 1.0

        state[self.num_nodes + 0] = self.hops_used / max(1, self.max_hops if self.max_hops is not None else 1)
        state[self.num_nodes + 1] = self.cum_delay / max(self.max_delay_ms if self.max_delay_ms is not None else 1e-6, 1e-6)

        e2e_loss = 1.0 - self.success_prob
        if self.max_loss > 0:
            e2e_loss_norm = min(e2e_loss / self.max_loss, 1.0)
        else:
            e2e_loss_norm = e2e_loss
        state[self.num_nodes + 2] = e2e_loss_norm

        d = self.dist_to_goal[self.current_idx]
        d_norm = d / max(self.max_dist_to_goal, 1e-6)
        state[self.num_nodes + 3] = d_norm

        return state

    # -------------------------
    # GYM-LIKE API
    # -------------------------

    def reset(self):
        self.current_idx = self.start_idx
        self.hops_used = 0
        self.cum_delay = 0.0
        self.cum_jitter = 0.0
        self.success_prob = 1.0
        return self._encode_state()

    def step(self, action_idx: int):
        done = False

        if self.current_idx == self.goal_idx:
            # already at goal, should terminate
            return self._encode_state(), 0.0, True, {"info": "already_at_goal"}

        key = (self.current_idx, action_idx)
        if key not in self.edge_dict:
            # invalid move
            reward = self.fail_penalty
            done = True
            info = {"info": "invalid_move"}
            return self._encode_state(), reward, done, info

        # distance to goal before moving
        dist_prev_goal = self.dist_to_goal[self.current_idx]

        # apply move
        dist, delay_ms, bw, jitter_ms, loss = self.edge_dict[key]

        self.current_idx = action_idx
        self.hops_used += 1
        self.cum_delay += delay_ms
        self.cum_jitter += jitter_ms
        self.success_prob *= (1.0 - loss)
        e2e_loss = 1.0 - self.success_prob

        # normalize QoS
        bw_norm = bw / max(self.target_bw_mbps, 1e-6)
        bw_norm = min(bw_norm, 2.0)

        hop_delay_norm = delay_ms / max(self.max_delay_ms, 1e-6)
        hop_jitter_norm = jitter_ms / max(self.max_jitter_ms, 1e-6)
        e2e_loss_norm = e2e_loss / max(self.max_loss, 1e-6) if self.max_loss > 0 else e2e_loss
        e2e_loss_norm = min(max(e2e_loss_norm, 0.0), 2.0)

        # QoS-based step reward
        reward_step = (
            + self.w_bw     * bw_norm
            - self.w_delay  * hop_delay_norm
            - self.w_jitter * hop_jitter_norm
            - self.w_loss   * e2e_loss_norm
            - self.w_hop
        )

        # progress shaping: if moving closer to the goal, reward the agent
        dist_next_goal = self.dist_to_goal[self.current_idx]
        delta_d = dist_prev_goal - dist_next_goal  # >0 if closer
        delta_d_norm = delta_d / max(self.max_dist_to_goal, 1e-6)
        reward_progress = self.w_progress * delta_d_norm

        reward = reward_step + reward_progress

        info = {
            "dist": dist,
            "delay_ms": delay_ms,
            "bw": bw,
            "jitter_ms": jitter_ms,
            "hop": self.hops_used,
            "e2e_loss": e2e_loss,
            "cum_delay": self.cum_delay,
            "cum_jitter": self.cum_jitter,
        }

        # ----------------------
        # Termination checks
        # ----------------------
        avg_jitter = self.cum_jitter / max(self.hops_used, 1)

        if (self.max_loss is not None) and (e2e_loss > self.max_loss):
            reward += self.fail_penalty
            done = True
            info["info"] = "e2e_loss_exceeded"

        elif self.cum_delay > self.max_delay_ms:
            reward += self.fail_penalty
            done = True
            info["info"] = "delay_exceeded"

        elif avg_jitter > self.max_jitter_ms:
            reward += self.fail_penalty
            done = True
            info["info"] = "jitter_exceeded"

        elif self.hops_used >= self.max_hops and self.current_idx != self.goal_idx:
            reward += self.fail_penalty
            done = True
            info["info"] = "max_hops_exceeded"

        elif self.current_idx == self.goal_idx:
            # check QoS end-to-end
            qos_ok = (
                (self.cum_delay <= self.max_delay_ms)
                and (e2e_loss <= self.max_loss)
                and (avg_jitter <= self.max_jitter_ms)
            )

            if qos_ok:
                reward += self.goal_bonus
                # more remaining hops -> more reward
                reward += 5.0 * max(self.max_hops - self.hops_used, 0)
                info["info"] = "reached_goal_qos_ok"
            else:
                reward += -10.0
                info["info"] = "reached_goal_qos_bad"

            done = True

        else:
            info["info"] = "step"

        next_state = self._encode_state()
        return next_state, reward, done, info

    # -------------------------
    # INFERENCE
    # -------------------------

    def get_valid_actions(self, node_idx=None):
        if node_idx is None:
            node_idx = self.current_idx
        if node_idx is None:
            return []
        return self.neighbors_idx.get(node_idx, [])

    def decode_path(self, indices):
        return [self.idx2id[i] for i in indices]


# =============================
# 4. DQN AGENT
# =============================

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 500,
        device: str = "cpu",

        # --- PER params ---
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-5,

        epsilon_start: float = 0.0,
        epsilon_min: float = 0.0,
        epsilon_decay: float = 1.0,
        
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        # self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_update_freq = target_update_freq
        self.device = device

        self.train_steps = 0

        # ----------------
        # PER / Replay
        # ----------------
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta_start
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.per_eps = per_eps

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha
            )
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.train_steps = 0

    def select_action(self, state, valid_actions=None, explore: bool = True):
        """
        - explore=True : Used in TRAIN, with reset_noise.
        - explore=False: Used in EVAL, without reset_noise.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, state_dim)

        if explore and hasattr(self.policy_net, "reset_noise") and self.policy_net.training:
            self.policy_net.reset_noise()

        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)  # (action_dim,)

        if valid_actions is None or len(valid_actions) == 0:
            action = torch.argmax(q_values).item()
        else:
            q_sub = q_values[valid_actions]
            idx_sub = torch.argmax(q_sub).item()
            action = valid_actions[idx_sub]

        return action



    def store_transition(self, *args):
        self.buffer.push(*args)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        # =============================
        # 1) get batch from buffer
        # =============================
        if self.use_per:
            batch, indices, weights = self.buffer.sample(
                self.batch_size, beta=self.per_beta
            )
            weights = weights.to(self.device).unsqueeze(1)  # (B,1)
        else:
            batch = self.buffer.sample(self.batch_size)
            indices = None
            weights = None

        state_batch = torch.tensor(
            np.array(batch.state),
            dtype=torch.float32,
            device=self.device,
        )
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.array(batch.next_state),
            dtype=torch.float32,
            device=self.device,
        )
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # >>> reset noise for both policy & target <<<
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise()
        # =============================
        # 2) Q(s,a) and target (Double DQN)
        # =============================

        # Q(s,a) from policy_net
        q_values = self.policy_net(state_batch).gather(1, action_batch)  # (B,1)

        with torch.no_grad():
            # online net chose best action at s'
            next_q_online = self.policy_net(next_state_batch)             # (B,A)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)     # (B,1)

            # target net evaluates Q for that action
            next_q_target = self.target_net(next_state_batch).gather(1, next_actions)

            target = reward_batch + self.gamma * next_q_target * (1.0 - done_batch)

        # TD-error
        td_errors = target - q_values  # (B,1)

        # =============================
        # 3) Loss (PER: weighted)
        # =============================
        if self.use_per and weights is not None:
            # Weighted SmoothL1Loss
            loss = (weights * torch.nn.functional.smooth_l1_loss(
                q_values, target, reduction="none"
            )).mean()
        else:
            loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # =============================
        # 4) Update priority & beta
        # =============================
        if self.use_per and indices is not None:
            # prio_i = |TD error| + eps
            prios = td_errors.detach().abs().cpu().numpy().squeeze(1) + self.per_eps
            if isinstance(self.buffer, PrioritizedReplayBuffer):
                self.buffer.update_priorities(indices, prios)

            # anneal beta gradually to 1.0
            self.per_beta = min(
                1.0,
                self.per_beta + (1.0 - self.per_beta_start) / max(1, self.per_beta_frames)
            )

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


    def decay_epsilon(self):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        pass


# =============================
# 5. TRAINING LOOP (ONE ENV)
# =============================

def train_dqn_routing(
    env: UAVRoutingEnvTorch,
    num_episodes: int = 2000,
    gamma: float = 0.95,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    target_update_freq: int = 500,
    device: str = "cpu",
    log_interval: int = 500,
):
    """
    Train DQN on a routing environment (1 time_slot).

    Logging:
      - Count episodes ended by each QoS reason:
        + reached_goal_qos_ok
        + reached_goal_qos_bad
        + e2e_loss_exceeded
        + delay_exceeded
        + jitter_exceeded
        + max_hops_exceeded
        + invalid_move
      - Calculate avg cum_delay / cum_jitter / e2e_loss / hops for success/fail.
    """
    device = torch.device(device)

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        device=device,

        # bật PER:
        use_per=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100_000,
    )

    # ---- statistics for logging ----
    term_counts = {
        "reached_goal_qos_ok": 0,
        "reached_goal_qos_bad": 0,
        "e2e_loss_exceeded": 0,
        "delay_exceeded": 0,
        "jitter_exceeded": 0,
        "max_hops_exceeded": 0,
        "invalid_move": 0,
        "other": 0,
    }

    # accumulate QoS at the end of episode for success/fail
    success_stats = {
        "count": 0,
        "cum_delay": 0.0,
        "cum_jitter": 0.0,
        "e2e_loss": 0.0,
        "hops": 0.0,
    }
    fail_stats = {
        "count": 0,
        "cum_delay": 0.0,
        "cum_jitter": 0.0,
        "e2e_loss": 0.0,
        "hops": 0.0,
    }

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        last_info = None

        while not done:
            valid_actions = env.get_valid_actions()
            # TRAIN: explore=True
            action_idx = agent.select_action(state, valid_actions=valid_actions, explore=True)


            next_state, reward, done, info = env.step(action_idx)

            agent.store_transition(state, action_idx, reward, next_state, done)
            loss = agent.update()

            state = next_state
            total_reward += reward
            last_info = info

        # ----- logging episode-level -----
        reason = last_info.get("info") if last_info is not None else "other"
        if reason not in term_counts:
            reason = "other"
        term_counts[reason] += 1

        # QoS last episode
        cum_delay = last_info.get("cum_delay", 0.0) if last_info else 0.0
        cum_jitter = last_info.get("cum_jitter", 0.0) if last_info else 0.0
        e2e_loss = last_info.get("e2e_loss", 0.0) if last_info else 0.0
        hops = last_info.get("hop", 0) if last_info else 0

        if reason == "reached_goal_qos_ok":
            success_stats["count"] += 1
            success_stats["cum_delay"] += cum_delay
            success_stats["cum_jitter"] += cum_jitter
            success_stats["e2e_loss"] += e2e_loss
            success_stats["hops"] += hops
        else:
            fail_stats["count"] += 1
            fail_stats["cum_delay"] += cum_delay
            fail_stats["cum_jitter"] += cum_jitter
            fail_stats["e2e_loss"] += e2e_loss
            fail_stats["hops"] += hops

        # ----- logging episode-level -----
        if ep % log_interval == 0 or ep == num_episodes:
            print(
                f"[DQN] Episode {ep}/{num_episodes}, "
                f"reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}"
            )

            total_episodes = sum(term_counts.values())
            if total_episodes == 0:
                total_episodes = 1  # tránh chia 0

            print("  Termination reasons (from start):")
            for k, v in term_counts.items():
                pct = 100.0 * v / total_episodes
                print(f"    - {k:22s}: {v:5d} ({pct:5.1f}%)")

            if success_stats["count"] > 0:
                c = success_stats["count"]
                print("  Success QoS (reached_goal_qos_ok):")
                print(
                    f"    avg cum_delay  = {success_stats['cum_delay'] / c:.2f} ms, "
                    f"avg cum_jitter = {success_stats['cum_jitter'] / c:.2f} ms"
                )
                print(
                    f"    avg e2e_loss   = {success_stats['e2e_loss'] / c:.3f}, "
                    f"avg hops      = {success_stats['hops'] / c:.2f}"
                )
            else:
                print("  Success QoS (reached_goal_qos_ok): no episodes yet.")

            if fail_stats["count"] > 0:
                c = fail_stats["count"]
                print("  Fail QoS / khác:")
                print(
                    f"    avg cum_delay  = {fail_stats['cum_delay'] / c:.2f} ms, "
                    f"avg cum_jitter = {fail_stats['cum_jitter'] / c:.2f} ms"
                )
                print(
                    f"    avg e2e_loss   = {fail_stats['e2e_loss'] / c:.3f}, "
                    f"avg hops      = {fail_stats['hops'] / c:.2f}"
                )
            print("-" * 60)

    return agent

def rollout_dqn_episode(
    env: UAVRoutingEnvTorch,
    agent: DQNAgent,
    device: str = "cpu",
    epsilon_eval: float = 0.05,
    max_steps: int | None = None,
):
    """
    Chạy 1 episode với epsilon nhỏ (epsilon_eval) để evaluation.
    Khác với train:
      - Không update network.
      - Không dùng replay buffer.
    Trả về:
      - path_node_ids (hoặc None nếu không tới goal),
      - total_dist,
      - total_delay.
    """
    device = torch.device(device)

    if max_steps is None:
        max_steps = env.max_hops + 5

    # Lưu epsilon cũ để khôi phục
    old_eps = agent.epsilon
    agent.epsilon = epsilon_eval

    state = env.reset()
    total_dist = 0.0
    total_delay = 0.0
    path_indices = [env.start_idx]

    visited = set([env.start_idx])
    done = False

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # dùng select_action của agent (epsilon-greedy với epsilon_eval)
        action_idx = agent.select_action(state, valid_actions=valid_actions)

        # tránh vòng lặp đơn giản
        if action_idx in visited and action_idx != env.goal_idx:
            pruned = [a for a in valid_actions if a not in visited]
            if pruned:
                action_idx = agent.select_action(state, valid_actions=pruned)

        key = (env.current_idx, action_idx)
        if key not in env.edge_dict:
            break

        dist, delay_ms, bw, jitter_ms, loss = env.edge_dict[key]
        total_dist += dist
        total_delay += delay_ms

        next_state, reward, done, info = env.step(action_idx)
        path_indices.append(action_idx)
        visited.add(action_idx)
        state = next_state

        if done:
            break

    # Khôi phục epsilon
    agent.epsilon = old_eps

    if env.current_idx != env.goal_idx:
        return None, None, None

    path_node_ids = env.decode_path(path_indices)
    return path_node_ids, total_dist, total_delay

def extract_greedy_path_dqn(
    env: UAVRoutingEnvTorch,
    agent: DQNAgent,
    device: str = "cpu",
    num_trials: int = 10,
    epsilon_eval: float = 0.05,
    max_steps: int | None = None,
    verbose: bool = False,
):
    """
    Extract the best path found by running multiple evaluation episodes
    """
    best_path = None
    best_dist = None
    best_delay = None

    for i in range(num_trials):
        path, dist, delay = rollout_dqn_episode(
            env,
            agent,
            device=device,
            # epsilon_eval=epsilon_eval,
            max_steps=max_steps,
        )

        if path is None:
            continue

        if best_path is None or delay < best_delay:
            best_path = path
            best_dist = dist
            best_delay = delay

        if verbose:
            print(f"[EVAL TRIAL {i+1}/{num_trials}] path={path}, dist={dist}, delay={delay} ms")

    if best_path is None and verbose:
        print("[EVAL] No path found to BS2 in num_trials.")

    return best_path, best_dist, best_delay
