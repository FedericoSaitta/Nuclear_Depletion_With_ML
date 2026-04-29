import torch.nn as nn
import torch.nn.functional as F
from ML.models.model_helper import get_activation
import torch

class Deep_Neural_Network(nn.Module):
  def __init__(self, n_inputs, n_outputs, hidden_layers, dropout_prob, activation, output_activation, residual):
    super(Deep_Neural_Network, self).__init__()
    self.residual = residual

    layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
    self.layers = nn.ModuleList([
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ])
    self.dropout = nn.Dropout(p=dropout_prob)

    self.activation_fn = get_activation(activation)
    self.output_activation_fn = get_activation(output_activation)

  def forward(self, x):
    for layer in self.layers[:-1]:
      residual = x  # store input for skip connection
      x = layer(x)
      x = self.activation_fn(x)
      x = self.dropout(x)

      # Only add residual if dimensions match and flag is True
      if self.residual and x.shape == residual.shape:
        x = x + residual

    y = self.layers[-1](x)
    y = self.output_activation_fn(y)
    return y


# ─── ODE Function ────────────────────────────────────────────────────────────

class ODEFuncForced(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.nfe = 0
    self.t_points = None
    self.forcing_profiles = None

    n_input = len(cfg.dataset.inputs)
    n_target = len(cfg.dataset.targets)

    self.net = Deep_Neural_Network(
      n_inputs=n_input + n_target,
      n_outputs=n_target,
      hidden_layers=cfg.model.layers,
      dropout_prob=cfg.model.dropout_probability,
      activation=cfg.model.activation,
      output_activation=cfg.model.output_activation,
      residual=cfg.model.residual_connections,
    )

  def set_forcing(self, t_points, forcing_profiles):
    self.t_points = t_points
    self.forcing_profiles = forcing_profiles  # (batch, steps, n_input)

  def _interpolate_forcing(self, t):
    """Piecewise-constant (zero-order hold) forcing interpolation.
    
    Returns the forcing value at the left endpoint of whichever interval
    t falls into, i.e. the value is held constant until the next grid point.
    This is correct for power profiles that are sampled independently at
    each timestep (step-function behaviour).
    """
    t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
    idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
    idx = idx.clamp(0, len(self.t_points) - 2)

    return self.forcing_profiles[:, idx, :]   # (batch, n_input)

  def forward(self, t, y):
    self.nfe += 1
    forcing = self._interpolate_forcing(t)      # (batch, n_input)
    combined = torch.cat([forcing, y], dim=-1)   # (batch, n_input + n_target)
    return self.net(combined)
  
# ─── Matrix ODE Function ─────────────────────────────────────────────────────
 
class ODEFuncMatrix(nn.Module):
  """ODE function with a constrained depletion matrix A(t).

  dy/dt = A(forcing(t), y(t)) @ y(t)

  Constraints (all configurable):
    - matrix_zero_entries:           forced to zero (sparsity).
    - Diagonal entries:              -softplus output (loss).
    - Off-diagonal entries:          +softplus output (production).
    - matrix_equal_opposite_pairs:   [primary, linked] with A[linked] = -A[primary]
                                     (mass conservation for single-channel transitions).
    - matrix_constant_pairs:         subset of equal-opposite pairs whose magnitude
                                     is a learnable scalar (e.g. beta-decay rates).

  The network only outputs entries that are NOT (a) zeroed, (b) linked, or
  (c) constants — so capacity is not wasted on derived quantities.
  """

  def __init__(self, cfg):
    super().__init__()
    self.nfe = 0
    self.t_points = None
    self.forcing_profiles = None

    n_input = len(cfg.dataset.inputs)
    self.n_target = len(cfg.dataset.targets)
    n_net_input = n_input + self.n_target

    # ── Sparsity mask ──
    sparsity_mask = torch.ones(self.n_target, self.n_target, dtype=torch.bool)
    zero_entries = getattr(cfg.model, 'matrix_zero_entries', None) or []
    for pair in zero_entries:
      sparsity_mask[int(pair[0]), int(pair[1])] = False
    self.register_buffer('sparsity_mask', sparsity_mask)

    # ── Equal-opposite pairs: (linked_pos) -> (primary_pos) ──
    eo_pairs = getattr(cfg.model, 'matrix_equal_opposite_pairs', None) or []
    linked_to_primary = {}
    for pair in eo_pairs:
      p = (int(pair[0][0]), int(pair[0][1]))
      l = (int(pair[1][0]), int(pair[1][1]))
      if not sparsity_mask[p]:
        raise ValueError(f"Equal-opposite primary {p} is in matrix_zero_entries.")
      if not sparsity_mask[l]:
        raise ValueError(f"Equal-opposite linked {l} is in matrix_zero_entries.")
      if (p[0] == p[1]) == (l[0] == l[1]):
        raise ValueError(
          f"Equal-opposite pair {p}<->{l}: one entry must be diagonal "
          f"and the other off-diagonal."
        )
      linked_to_primary[l] = p

    # ── Constant pairs ──
    const_pairs = getattr(cfg.model, 'matrix_constant_pairs', None) or []
    constant_primaries = set()
    for pair in const_pairs:
      p = (int(pair[0][0]), int(pair[0][1]))
      l = (int(pair[1][0]), int(pair[1][1]))
      if linked_to_primary.get(l) != p:
        raise ValueError(
          f"Constant pair {p}<->{l} must also appear in matrix_equal_opposite_pairs."
        )
      constant_primaries.add(p)

    # ── Classify each active entry: net / constant-primary / linked ──
    active_rows, active_cols = torch.where(sparsity_mask)

    net_rows, net_cols, net_is_diag = [], [], []
    const_rows, const_cols = [], []
    linked_rows, linked_cols = [], []
    linked_primary_rows, linked_primary_cols = [], []  # where to read primary from

    for k in range(active_rows.shape[0]):
      i, j = int(active_rows[k]), int(active_cols[k])
      pos = (i, j)
      if pos in linked_to_primary:
        continue  # handled in second pass
      if pos in constant_primaries:
        const_rows.append(i); const_cols.append(j)
      else:
        net_rows.append(i); net_cols.append(j); net_is_diag.append(i == j)

    for k in range(active_rows.shape[0]):
      i, j = int(active_rows[k]), int(active_cols[k])
      pos = (i, j)
      if pos not in linked_to_primary:
        continue
      pr, pc = linked_to_primary[pos]
      linked_rows.append(i); linked_cols.append(j)
      linked_primary_rows.append(pr); linked_primary_cols.append(pc)

    # ── Register everything ──
    L = torch.long
    self.register_buffer('net_rows',    torch.tensor(net_rows,    dtype=L))
    self.register_buffer('net_cols',    torch.tensor(net_cols,    dtype=L))
    self.register_buffer('net_is_diag', torch.tensor(net_is_diag, dtype=torch.bool))
    self.n_net = len(net_rows)

    self.register_buffer('const_rows', torch.tensor(const_rows, dtype=L))
    self.register_buffer('const_cols', torch.tensor(const_cols, dtype=L))
    self.n_const = len(const_rows)
    if self.n_const > 0:
      self.const_raw = nn.Parameter(torch.zeros(self.n_const))
      self.register_buffer('const_is_diag',
                           (self.const_rows == self.const_cols))
    else:
      self.register_parameter('const_raw', None)

    self.register_buffer('linked_rows',         torch.tensor(linked_rows,         dtype=L))
    self.register_buffer('linked_cols',         torch.tensor(linked_cols,         dtype=L))
    self.register_buffer('linked_primary_rows', torch.tensor(linked_primary_rows, dtype=L))
    self.register_buffer('linked_primary_cols', torch.tensor(linked_primary_cols, dtype=L))
    self.n_linked = len(linked_rows)

    self.register_buffer('ranges', torch.ones(self.n_target))
  
    # Network outputs ONLY the entries that need it
    self.net = Deep_Neural_Network(
      n_inputs=n_net_input,
      n_outputs=max(self.n_net, 1),  # guard against degenerate n_net=0
      hidden_layers=cfg.model.layers,
      dropout_prob=cfg.model.dropout_probability,
      activation=cfg.model.activation,
      output_activation='none',
      residual=cfg.model.residual_connections,
    )



  def set_ranges(self, ranges):
      """Per-target physical ranges (max - min from MinMax scaler)."""
      if not isinstance(ranges, torch.Tensor):
          ranges = torch.tensor(ranges, dtype=torch.float32)
      self.ranges = ranges.to(self.ranges.device).to(self.ranges.dtype)

  def set_forcing(self, t_points, forcing_profiles):
    self.t_points = t_points
    self.forcing_profiles = forcing_profiles  # (batch, steps, n_input)

  def _interpolate_forcing(self, t):
    """Piecewise-constant (zero-order hold) forcing interpolation."""
    t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
    idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
    idx = idx.clamp(0, len(self.t_points) - 2)
    return self.forcing_profiles[:, idx, :]   # (batch, n_input)

  def _build_matrix(self, forcing, y):
    """Assemble A in three stages: network -> constants -> linked = -primary."""
    batch = y.shape[0]
    A = y.new_zeros(batch, self.n_target, self.n_target)

    # 1) Network entries (sign by diagonality)
    if self.n_net > 0:
      net_input = torch.cat([forcing, y], dim=-1)
      net_raw = self.net(net_input)[:, :self.n_net]   # (batch, n_net)
      net_vals = torch.where(
        self.net_is_diag,
        -F.softplus(net_raw),
         F.softplus(net_raw),
      )
      A[:, self.net_rows, self.net_cols] = net_vals

    # 2) Constant entries (learnable scalars, broadcast over batch)
    if self.n_const > 0:
      const_vals = torch.where(
        self.const_is_diag,
        -F.softplus(self.const_raw),
         F.softplus(self.const_raw),
      )                                              # (n_const,)
      A[:, self.const_rows, self.const_cols] = \
        const_vals.unsqueeze(0).expand(batch, -1)

    # 3) Linked entries: equal-opposite in PHYSICAL space.
    #    A_phys[linked] = -A_phys[primary]  =>
    #    A_scaled[linked] = -A_scaled[primary] * (r[p_row] * r[l_col]) / (r[l_row] * r[p_col])
    if self.n_linked > 0:
        primary_vals = A[:, self.linked_primary_rows, self.linked_primary_cols]
        ratio = (
            self.ranges[self.linked_primary_rows] * self.ranges[self.linked_cols]
        ) / (
            self.ranges[self.linked_rows] * self.ranges[self.linked_primary_cols]
        )
        A[:, self.linked_rows, self.linked_cols] = -primary_vals * ratio

    return A

  def forward(self, t, y):
    self.nfe += 1
    forcing = self._interpolate_forcing(t)            # (batch, n_input)
    A = self._build_matrix(forcing, y)                # (batch, n_target, n_target)
    return torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)  # (batch, n_target)