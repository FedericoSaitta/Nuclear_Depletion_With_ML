import torch
import torch.nn as nn
import torch.nn.functional as F

from nuclear_surrogates.models.model_helper import get_activation


class Deep_Neural_Network(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        hidden_layers,
        dropout_prob,
        activation,
        output_activation,
        residual,
    ):
        super().__init__()
        self.residual = residual

        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(
                    layer_sizes[:-1], layer_sizes[1:], strict=False
                )
            ]
        )
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
        return self.output_activation_fn(y)


# ─── ODE Function ────────────────────────────────────────────────────────────


class ForcedODEFunc(nn.Module):
    """Base for ODE right-hand sides driven by an external forcing profile.

    Holds the forcing grid and the zero-order-hold lookup shared by every
    concrete right-hand side; subclasses only implement `forward`.
    """

    def __init__(self):
        super().__init__()
        self.nfe = 0
        self.t_points = None
        self.forcing_profiles = None

    def set_forcing(self, t_points, forcing_profiles):
        self.t_points = t_points
        self.forcing_profiles = forcing_profiles  # (batch, steps, n_input)

    def forcing_index(self, t):
        """Index of the forcing interval containing *t*.

        Also needed by the Jacobian analysis, which must attribute a gradient to
        the same interval the forward pass actually read.
        """
        t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
        idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
        return idx.clamp(0, len(self.t_points) - 2)

    def _interpolate_forcing(self, t):
        """Piecewise-constant (zero-order hold) forcing interpolation.

        Returns the forcing at the left endpoint of t's interval, held constant
        until the next grid point — correct for power profiles sampled
        independently at each timestep.
        """
        return self.forcing_profiles[:, self.forcing_index(t), :]  # (batch, n_input)


class ODEFuncForced(ForcedODEFunc):
    """Black-box right-hand side: dy/dt = net(forcing(t), y)."""

    def __init__(self, cfg):
        super().__init__()

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

    def forward(self, t, y):
        self.nfe += 1
        forcing = self._interpolate_forcing(t)  # (batch, n_input)
        combined = torch.cat([forcing, y], dim=-1)  # (batch, n_input + n_target)
        return self.net(combined)


# ─── Matrix ODE Function ─────────────────────────────────────────────────────


class ODEFuncMatrix(ForcedODEFunc):
    """ODE function with a constrained depletion matrix A(t).

    dy/dt = A(forcing(t), y(t)) @ y(t)

    Constraints:
      - matrix_zero_entries: forced to zero (sparsity).
      - Diagonal entries:    -softplus output (loss).
      - Off-diagonal:        +softplus output (production).

    The network only outputs active (non-zero) entries, so no capacity is
    wasted on entries that are immediately masked out.
    """

    def __init__(self, cfg):
        super().__init__()

        n_input = len(cfg.dataset.inputs)
        self.n_target = len(cfg.dataset.targets)
        n_net_input = n_input + self.n_target  # forcing + isotope concentrations

        # ── Sparsity mask ──
        sparsity_mask = torch.ones(self.n_target, self.n_target, dtype=torch.bool)
        zero_entries = getattr(cfg.model, "matrix_zero_entries", None) or []
        for pair in zero_entries:
            sparsity_mask[int(pair[0]), int(pair[1])] = False
        self.register_buffer("sparsity_mask", sparsity_mask)

        # ── Active entries (row, col) and which of them are diagonal ──
        active_rows, active_cols = torch.where(sparsity_mask)
        self.register_buffer("active_rows", active_rows)
        self.register_buffer("active_cols", active_cols)
        self.register_buffer("active_is_diag", active_rows == active_cols)
        self.n_active = int(active_rows.shape[0])

        # Network outputs only the active entries
        self.net = Deep_Neural_Network(
            n_inputs=n_net_input,
            n_outputs=self.n_active,
            hidden_layers=cfg.model.layers,
            dropout_prob=cfg.model.dropout_probability,
            activation=cfg.model.activation,
            output_activation="none",  # We apply our own constraints
            residual=cfg.model.residual_connections,
        )

    def _build_matrix(self, forcing, y):
        """Build the constrained depletion matrix from forcing and state inputs.

        Returns: (batch, n_target, n_target) matrix with negative diagonal,
                 positive off-diagonal entries, and zeros elsewhere.
        """
        net_input = torch.cat([forcing, y], dim=-1)  # (batch, n_input + n_target)
        raw = self.net(net_input)  # (batch, n_active)
        batch = raw.shape[0]

        # Diagonal: -softplus (loss terms), Off-diagonal: +softplus (production)
        constrained = torch.where(
            self.active_is_diag,
            -F.softplus(raw),
            F.softplus(raw),
        )

        A = raw.new_zeros(batch, self.n_target, self.n_target)
        A[:, self.active_rows, self.active_cols] = constrained

        return A

    def forward(self, t, y):
        self.nfe += 1
        forcing = self._interpolate_forcing(t)  # (batch, n_input)
        A = self._build_matrix(forcing, y)  # (batch, n_target, n_target)
        return torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)  # (batch, n_target)
