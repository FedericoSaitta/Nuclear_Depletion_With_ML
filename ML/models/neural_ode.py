# ML/models/node_model.py
import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

from loguru import logger
import lightning as L
from torchdiffeq import odeint

from ML.utils import plot
from ML.utils import metrics
from ML.models.model_architectures import ODEFuncForced
from ML.models.model_helper import get_loss_fn
import ML.datamodule.data_scalers as data_scaler

# ─── Lightning Module ────────────────────────────────────────────────────────

class NODE_Model(L.LightningModule):
  def __init__(self, config_object):
    super().__init__()
    self.save_hyperparameters(OmegaConf.to_container(config_object, resolve=True))
    self.cfg = config_object

    # Model
    self.func = ODEFuncForced(config_object)

    # Training Config
    self.loss_fn = get_loss_fn(config_object.train.loss)

    # Optimizer Config
    self.rtol = getattr(config_object.train, 'rtol', 1e-5)
    self.atol = getattr(config_object.train, 'atol', 1e-7)

    # Results directory
    self.result_dir = f'results/{config_object.model.name}/'
    os.makedirs(self.result_dir, exist_ok=True)

    # Collect losses for plotting
    self._train_losses = []
    self._val_losses = []
    self._train_nfes = []

    # Collect data for test-level metrics
    self._test_preds = []
    self._test_trues = []
    self._test_input_trajs = []

  def setup(self, stage=None):
    """Called after datamodule.setup — grab t_span and feature layout."""
    dm = self.trainer.datamodule
    self.t_span = dm.t_span.to(self.device)
    self.n_input_features = dm.n_input_features
    self.n_target_features = dm.n_target_features

  def _odeint(self, y0, t_span):
      """Central odeint call — solver configured from yaml."""
      options = {}
      if self.cfg.train.solver == 'rk4':
          step_size = getattr(self.cfg.train, 'step_size', None)
          if step_size:
              options['step_size'] = step_size
      return odeint(
          self.func, y0, t_span,
          method=self.cfg.train.solver,
          rtol=self.rtol, atol=self.atol,
          options=options if options else None,
      )

  def _forward_batch(self, batch):
    """
    Shared forward pass for train/val/test.

    Trajectory layout: [forcing_features..., target_features...]
    batch[0] is (batch_size, steps, n_input_features + n_target_features)
    
    The ODE integrator feeds the FULL state vector y (all isotopes) back
    into the network at every solver step. This means all isotopes
    inform all future predictions — the coupling is automatic.

    Returns: target_pred (batch_size, steps, n_target), target_true (batch_size, steps, n_target)
    """
    trajectories = batch[0]

    n_in = self.n_input_features
    
    forcing_profiles = trajectories[:, :, :n_in]
    target_true = trajectories[:, :, n_in:]
    y0 = target_true[:, 0, :]  # All isotope concentrations at t=0

    t_span = self.t_span.to(trajectories.device)
    self.func.set_forcing(t_span, forcing_profiles)

    # odeint integrates the full state vector [isotope_1, isotope_2, ...]
    # so at each internal solver step, the network receives ALL current
    # isotope predictions to compute ALL derivatives.
    target_pred = self._odeint(y0, t_span)
    target_pred = target_pred.permute(1, 0, 2)

    return target_pred, target_true

  # ─── Unscaling helpers ─────────────────────────────────────────────

  def _unscale_targets(self, scaled_2d):
    """Unscale target values. Input shape: (N, n_target)."""
    target_scaler = self.trainer.datamodule.target_scaler
    return data_scaler.inverse_transformer(target_scaler, scaled_2d)

  def _unscale_inputs(self, scaled_2d):
    """Unscale input values. Input shape: (N, n_input)."""
    input_scaler = self.trainer.datamodule.input_scaler
    return data_scaler.inverse_transformer(input_scaler, scaled_2d)

  # ── Training ─────────────────────────────────────────────────────────

  def training_step(self, batch, batch_idx):
    self.func.nfe = 0
    target_pred, target_true = self._forward_batch(batch)

    loss = self.loss_fn(target_pred, target_true)

    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log('nfe', float(self.func.nfe), on_step=False, on_epoch=True)
    return loss

  def on_train_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["train_loss"].item()
    self._train_losses.append(epoch_loss)

    epoch_nfe = self.trainer.callback_metrics["nfe"].item()
    self._train_nfes.append(epoch_nfe)

    lr = self.optimizers().param_groups[0]['lr']
    self.log('lr', lr, on_epoch=True, prog_bar=True)

  def on_train_end(self):
    plot.plot_losses(self._train_losses, self._val_losses, self.result_dir, nfes=self._train_nfes)
    logger.info("Training complete.")

  # ── Validation ───────────────────────────────────────────────────────

  def validation_step(self, batch, batch_idx):
    target_pred, target_true = self._forward_batch(batch)
    loss = self.loss_fn(target_pred, target_true)
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  def on_validation_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["val_loss"].item()
    self._val_losses.append(epoch_loss)

  # ── Test ─────────────────────────────────────────────────────────────

  def test_step(self, batch, batch_idx):
    target_pred, target_true = self._forward_batch(batch)
    trajectories = batch[0]
    n_in = self.n_input_features

    loss = self.loss_fn(target_pred, target_true)
    self.log('test_loss', loss, on_step=False, on_epoch=True)

    self._test_preds.append(target_pred.cpu().numpy())
    self._test_trues.append(target_true.cpu().numpy())
    self._test_input_trajs.append(trajectories[:, :, :n_in].cpu().numpy())

    return loss

  def on_test_epoch_end(self):
    datamodule = self.trainer.datamodule
    
    all_preds_scaled = np.concatenate(self._test_preds, axis=0)        # (num_runs, steps, n_target)
    all_trues_scaled = np.concatenate(self._test_trues, axis=0)        # (num_runs, steps, n_target)
    all_inputs_scaled = np.concatenate(self._test_input_trajs, axis=0) # (num_runs, steps, n_input)

    num_runs, steps, n_target = all_preds_scaled.shape
    n_input = all_inputs_scaled.shape[2]

    # ── Unscale using separate scalers ──
    ar_preds_unscaled = self._unscale_targets(all_preds_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    trues_unscaled = self._unscale_targets(all_trues_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    inputs_unscaled = self._unscale_inputs(all_inputs_scaled.reshape(-1, n_input)).reshape(num_runs, steps, n_input)

    # ── Teacher-forced predictions (single-step NODE) ──
    logger.info("Computing teacher-forced (single-step) predictions...")
    tf_preds_scaled = self._teacher_forced_predictions(all_inputs_scaled, all_trues_scaled)
    tf_preds_unscaled = self._unscale_targets(tf_preds_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)

    target_names = list(datamodule.target.keys())
    logger.info(f"Test set: {num_runs} trajectories, {steps} steps, {n_target} targets")
    
    # 1. Overall metrics + per-target plots
    flat_ar_preds = ar_preds_unscaled.reshape(-1, n_target)
    flat_trues = trues_unscaled.reshape(-1, n_target)
    
    mae_per_output = metrics.mae(flat_trues, flat_ar_preds)
    rmse_per_output = metrics.rmse(flat_trues, flat_ar_preds)
    r2_per_output = metrics.r2(flat_trues, flat_ar_preds)
    
    self.log('Mean Absolute Error (avg)', float(mae_per_output.mean()))
    self.log('Root Mean Squared Error (avg)', float(rmse_per_output.mean()))
    self.log('R-squared coefficient (avg)', float(r2_per_output.mean()))
    
    logger.info(f"TEST SET — UNSCALED metrics (Autoregressive):")
    logger.info(f"  R² (avg):   {r2_per_output.mean():.6f}")
    logger.info(f"  RMSE (avg): {rmse_per_output.mean():.6f}")
    logger.info(f"  MAE (avg):  {mae_per_output.mean():.6f}")

    # 2. Per-target: predictions vs actuals + residuals
    per_target_metrics = []
    for idx, target_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(output_dir, exist_ok=True)
      
      y_true_flat = flat_trues[:, idx]
      y_pred_flat = flat_ar_preds[:, idx]
      
      plot.plot_predictions_vs_actuals(
        y_true_flat, y_pred_flat,
        mae_per_output[idx], rmse_per_output[idx], r2_per_output[idx],
        output_dir
      )
      plot.plot_residuals_combined(y_true_flat, y_pred_flat, output_dir)
      
      per_target_metrics.append({
        'name': target_name,
        'mae': float(mae_per_output[idx]),
        'rmse': float(rmse_per_output[idx]),
        'r2': float(r2_per_output[idx])
      })

    # 3. MARE comparison (Teacher-Forcing vs Autoregressive)
    self._compute_mare_comparison(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled,
      target_names, per_target_metrics
    )

    # 4. Jacobian sensitivity analysis (replaces permutation feature importance)
    forcing_names = [
      key for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])
    ]
    self._compute_jacobian_analysis(
      all_inputs_scaled, all_trues_scaled, target_names, forcing_names
    )
    
    # 5. Prediction comparison plots (TF vs AR)
    self._plot_prediction_comparisons(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled, target_names
    )

    # 6. Error growth (MAE, MALE)
    self._plot_error_growth(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled, target_names
    )

    # 7. Trajectory-specific plots — loop over ALL targets
    t_np = self.t_span.cpu().numpy()
    num_to_plot = min(5, num_runs)
    for target_idx, target_name in enumerate(target_names):
      target_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(target_dir, exist_ok=True)

      for i in range(num_to_plot):
        plot.plot_node_trajectory(
          t_np,
          ar_preds_unscaled[i, :, target_idx],
          trues_unscaled[i, :, target_idx],
          inputs_unscaled[i, :, 0],  # Power (first input) as reference
          title=f'{target_name} — Test Trajectory {i+1}',
          save_path=os.path.join(target_dir, f'test_traj_{i+1}.png'),
        )

      plot.plot_node_trajectory_summary(
        t_np,
        ar_preds_unscaled[:, :, target_idx],
        trues_unscaled[:, :, target_idx],
        title=f'{target_name} — All Test Trajectories ({num_runs} runs)',
        save_path=os.path.join(target_dir, f'test_all_trajectories.png'),
      )

    # 8. Log to database
    if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'update_final_results'):
      test_metrics = {
        'mae_avg': float(mae_per_output.mean()),
        'rmse_avg': float(rmse_per_output.mean()),
        'r2_avg': float(r2_per_output.mean()),
        'per_target': per_target_metrics
      }
      self.trainer.logger.update_final_results(
        train_losses=self._train_losses,
        val_losses=self._val_losses,
        val_r2_scores=[],
        val_mae_scores=[],
        test_metrics=test_metrics
      )

    # Cleanup
    self._test_preds.clear()
    self._test_trues.clear()
    self._test_input_trajs.clear()

  # ─── Teacher-Forced Predictions ──────────────────────────────────────

  def _teacher_forced_predictions(self, all_inputs_scaled, all_trues_scaled):
    """
    Single-step NODE predictions: at each timestep t, use ground truth y(t) 
    as initial condition and integrate one dt forward to predict y(t+1).
    This is analogous to the DNN's teacher-forcing mode.
    
    y(t) is the FULL state vector (all isotopes), so the network uses
    ground truth for ALL isotopes at each step.
    
    Returns: (num_runs, steps, n_target) array of scaled predictions.
    """
    num_runs, steps, n_target = all_trues_scaled.shape
    tf_preds = np.zeros_like(all_trues_scaled)
    
    # First timestep: prediction = ground truth (no prior step to predict from)
    tf_preds[:, 0, :] = all_trues_scaled[:, 0, :]
    
    device = self.device
    t_span = self.t_span.to(device)
    
    # Process in batches to avoid OOM
    batch_size = 64
    
    for batch_start in range(0, num_runs, batch_size):
      batch_end = min(batch_start + batch_size, num_runs)
      
      inputs_batch = torch.tensor(
        all_inputs_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      trues_batch = torch.tensor(
        all_trues_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      
      forcing_profiles = inputs_batch  # (batch, steps, n_input)
      self.func.set_forcing(t_span, forcing_profiles)
      
      # Single-step integration for each timestep
      for t in range(steps - 1):
        y_t = trues_batch[:, t, :]  # Ground truth ALL isotopes at t
        t_short = t_span[t:t+2]     # Integrate from t to t+1
        
        with torch.no_grad():
          pred_t1 = self._odeint(y_t, t_short)[-1]  # Take last timepoint = t+1
        
        tf_preds[batch_start:batch_end, t+1, :] = pred_t1.cpu().numpy()
    
    return tf_preds

  # ─── MARE Comparison ─────────────────────────────────────────────────

  def _compute_mare_comparison(self, trues_unscaled, ar_preds_unscaled, 
                               tf_preds_unscaled, target_names, per_target_metrics=None):
    """Compare MARE between teacher-forcing and autoregressive (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("MARE: Teacher-Forcing vs Autoregressive")
    logger.info(f"{'='*20}")
    
    for idx, target_name in enumerate(target_names):
      # Flatten across runs and timesteps
      ar_gt = trues_unscaled[:, :, idx].flatten()
      ar_pred = ar_preds_unscaled[:, :, idx].flatten()
      tf_gt = trues_unscaled[:, :, idx].flatten()
      tf_pred = tf_preds_unscaled[:, :, idx].flatten()
      
      mare_ar = metrics.mare(ar_gt, ar_pred)
      mare_tf = metrics.mare(tf_gt, tf_pred)
      
      self.log(f'{target_name}/MARE_TeacherForcing', float(mare_tf))
      self.log(f'{target_name}/MARE_Autoregressive', float(mare_ar))
      
      logger.info(f"  {target_name}: MARE(TF)={mare_tf:.6f}, MARE(AR)={mare_ar:.6f}")
      
      if per_target_metrics is not None:
        per_target_metrics[idx]['mare_tf'] = float(mare_tf)
        per_target_metrics[idx]['mare_ar'] = float(mare_ar)

  # ─── Jacobian Sensitivity Analysis ───────────────────────────────────
  def compute_state_jacobian(self, t_eval, y_eval, forcing_eval, t_idx):
      with torch.inference_mode(False):
          y_eval = y_eval.clone().requires_grad_(True)
          
          was_training = self.func.training
          self.func.train()
          self.func.set_forcing(t_eval, forcing_eval)
          
          dydt = self.func(t_eval[t_idx], y_eval)  # correct time
          
          jacobians = []
          for i in range(dydt.shape[-1]):
              grad = torch.autograd.grad(
                  dydt[:, i].sum(), y_eval,
                  retain_graph=True, create_graph=False
              )[0]
              jacobians.append(grad)
          
          self.func.train(was_training)
          return torch.stack(jacobians, dim=1)

  def compute_forcing_jacobian(self, t_eval, y_eval, forcing_eval, t_idx):
      with torch.inference_mode(False):
          forcing_eval = forcing_eval.clone().requires_grad_(True)
          
          was_training = self.func.training
          self.func.train()
          self.func.set_forcing(t_eval, forcing_eval)
          
          dydt = self.func(t_eval[t_idx], y_eval.clone().detach())
          
          # Find which forcing index the interpolation actually used
          with torch.no_grad():
              t_clamped = t_eval[t_idx].clamp(t_eval[0], t_eval[-1])
              interp_idx = (torch.searchsorted(t_eval, t_clamped.unsqueeze(0)).squeeze() - 1).clamp(0, len(t_eval) - 2)
          
          jacobians = []
          for i in range(dydt.shape[-1]):
              grad = torch.autograd.grad(
                  dydt[:, i].sum(), forcing_eval,
                  retain_graph=True, create_graph=False
              )[0]
              jacobians.append(grad[:, interp_idx, :])  # ← match the interpolation index
          
          self.func.train(was_training)
          return torch.stack(jacobians, dim=1)
      
      
  def _compute_jacobian_analysis(self, all_inputs_scaled, all_trues_scaled,
                                 target_names, forcing_names):
    """
    Evaluate state and forcing Jacobians across all test trajectories and
    timesteps to build a unified sensitivity picture.
    
    Produces:
      - Mean |∂f/∂y| heatmap: which state variables drive which derivatives
      - Mean |∂f/∂u| heatmap: which forcing inputs drive which derivatives
      - Combined sensitivity bar chart per target
      - Jacobian evolution over time for each target
    """
    logger.info(f"\n{'='*20}")
    logger.info("JACOBIAN SENSITIVITY ANALYSIS")
    logger.info(f"{'='*20}")
    
    num_runs, steps, n_input = all_inputs_scaled.shape
    n_target = all_trues_scaled.shape[2]
    device = self.device
    t_span = self.t_span.to(device)
    
    # Sample a subset of runs and timesteps to keep computation tractable
    max_runs = min(20, num_runs)
    # Sample evenly spaced timesteps (skip first and last)
    timestep_indices = np.linspace(1, steps - 2, min(20, steps - 2), dtype=int)
    
    logger.info(f"Evaluating Jacobians: {max_runs} runs × {len(timestep_indices)} timesteps")
    
    # Accumulators: (n_timesteps_sampled, n_target, n_target) and (n_timesteps_sampled, n_target, n_input)
    all_state_jacs = []   # list of (n_target, n_target) per (run, timestep)
    all_forcing_jacs = [] # list of (n_target, n_input) per (run, timestep)
    
    # For time-resolved analysis: {timestep_idx: list of jacobians}
    state_jacs_by_time = {t: [] for t in timestep_indices}
    forcing_jacs_by_time = {t: [] for t in timestep_indices}
    
    batch_size = 32
    
    for batch_start in range(0, max_runs, batch_size):
      batch_end = min(batch_start + batch_size, max_runs)
      
      inputs_batch = torch.tensor(
        all_inputs_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      trues_batch = torch.tensor(
        all_trues_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      
      for t_idx in timestep_indices:
        y_t = trues_batch[:, t_idx, :]  # (batch, n_target)
        
        # State Jacobian: ∂f/∂y
        state_jac = self.compute_state_jacobian(t_span, y_t, inputs_batch, int(t_idx))
        abs_state_jac = state_jac.abs().cpu().numpy()  # (batch, n_target, n_target)
        
        # Forcing Jacobian: ∂f/∂u
        forcing_jac = self.compute_forcing_jacobian(t_span, y_t, inputs_batch, int(t_idx))
        abs_forcing_jac = forcing_jac.abs().cpu().numpy()  # (batch, n_target, n_input)
        
        # Collect per-sample Jacobians
        for b in range(abs_state_jac.shape[0]):
          all_state_jacs.append(abs_state_jac[b])
          all_forcing_jacs.append(abs_forcing_jac[b])
          state_jacs_by_time[t_idx].append(abs_state_jac[b])
          forcing_jacs_by_time[t_idx].append(abs_forcing_jac[b])
    
    # ── Aggregate: mean absolute Jacobians ──
    mean_state_jac = np.mean(all_state_jacs, axis=0)    # (n_target, n_target)
    mean_forcing_jac = np.mean(all_forcing_jacs, axis=0) # (n_target, n_input)
    std_state_jac = np.std(all_state_jacs, axis=0)
    std_forcing_jac = np.std(all_forcing_jacs, axis=0)
    
    # ── Log summary ──
    logger.info(f"\nMean |∂f/∂y| (state sensitivities):")
    for i, t_name in enumerate(target_names):
      for j, s_name in enumerate(target_names):
        logger.info(f"  ∂(d{t_name}/dt)/∂{s_name}: {mean_state_jac[i,j]:.6f} ± {std_state_jac[i,j]:.6f}")
    
    logger.info(f"\nMean |∂f/∂u| (forcing sensitivities):")
    for i, t_name in enumerate(target_names):
      for j, f_name in enumerate(forcing_names):
        logger.info(f"  ∂(d{t_name}/dt)/∂{f_name}: {mean_forcing_jac[i,j]:.6f} ± {std_forcing_jac[i,j]:.6f}")
    
    # ── Plot 1: State Jacobian heatmap ──
    self._plot_jacobian_heatmap(
      mean_state_jac, target_names, target_names,
      title='State Sensitivity: Mean |∂f/∂y|',
      xlabel='State variable (y_j)',
      ylabel='Derivative (dy_i/dt)',
      save_path=os.path.join(self.result_dir, 'jacobian_state_heatmap.png')
    )
    
    # ── Plot 2: Forcing Jacobian heatmap ──
    self._plot_jacobian_heatmap(
      mean_forcing_jac, forcing_names, target_names,
      title='Forcing Sensitivity: Mean |∂f/∂u|',
      xlabel='Forcing input (u_j)',
      ylabel='Derivative (dy_i/dt)',
      save_path=os.path.join(self.result_dir, 'jacobian_forcing_heatmap.png')
    )
    
    # ── Plot 3: Combined sensitivity bar chart per target ──
    all_names = target_names + forcing_names
    combined_jac = np.concatenate([mean_state_jac, mean_forcing_jac], axis=1)  # (n_target, n_target + n_input)
    combined_std = np.concatenate([std_state_jac, std_forcing_jac], axis=1)
    
    for idx, t_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, t_name)
      os.makedirs(output_dir, exist_ok=True)
      
      self._plot_combined_sensitivity(
        combined_jac[idx], combined_std[idx], all_names,
        target_names, forcing_names,
        title=f'Sensitivity of d{t_name}/dt',
        save_path=os.path.join(output_dir, 'jacobian_combined_sensitivity.png')
      )
    
    # ── Plot 4: Jacobian evolution over time per target ──
    sorted_timesteps = sorted(timestep_indices)
    time_fractions = self.t_span.cpu().numpy()[sorted_timesteps]
    
    for idx, t_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, t_name)
      
      # State sensitivities over time
      state_over_time = np.array([
        np.mean([j[idx, :] for j in state_jacs_by_time[t]], axis=0)
        for t in sorted_timesteps
      ])  # (n_timesteps, n_target)
      
      # Forcing sensitivities over time
      forcing_over_time = np.array([
        np.mean([j[idx, :] for j in forcing_jacs_by_time[t]], axis=0)
        for t in sorted_timesteps
      ])  # (n_timesteps, n_input)
      
      self._plot_jacobian_over_time(
        time_fractions, state_over_time, forcing_over_time,
        target_names, forcing_names,
        title=f'Sensitivity evolution: d{t_name}/dt',
        save_path=os.path.join(output_dir, 'jacobian_time_evolution.png')
      )
    
    logger.info(f"\n  ✓ Jacobian analysis plots saved to: {self.result_dir}")

  def _plot_jacobian_heatmap(self, matrix, col_names, row_names, title, xlabel, ylabel, save_path):
    """Plot a heatmap of a Jacobian matrix."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(max(8, len(col_names) * 1.2), max(6, len(row_names) * 0.8)))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean absolute sensitivity')
    
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Annotate cells with values
    for i in range(len(row_names)):
      for j in range(len(col_names)):
        val = matrix[i, j]
        color = 'white' if val > matrix.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

  def _plot_combined_sensitivity(self, sensitivities, stds, all_names,
                                 target_names, forcing_names, title, save_path):
    """Bar chart showing sensitivity of one derivative to all state + forcing variables."""
    import matplotlib.pyplot as plt
    
    n_target = len(target_names)
    n_forcing = len(forcing_names)
    
    colors = ['#2196F3'] * n_target + ['#FF9800'] * n_forcing  # Blue for state, orange for forcing
    
    sorted_idx = np.argsort(sensitivities)[::-1]
    sorted_sens = sensitivities[sorted_idx]
    sorted_stds = stds[sorted_idx]
    sorted_names = [all_names[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(max(8, len(all_names) * 0.6), 5))
    bars = ax.bar(range(len(sorted_sens)), sorted_sens, yerr=sorted_stds,
                  color=sorted_colors, capsize=3, edgecolor='gray', linewidth=0.5)
    
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean |∂f/∂·|')
    ax.set_title(title)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
      Patch(facecolor='#2196F3', label='State (∂f/∂y)'),
      Patch(facecolor='#FF9800', label='Forcing (∂f/∂u)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

  def _plot_jacobian_over_time(self, time_fractions, state_over_time, forcing_over_time,
                               target_names, forcing_names, title, save_path):
    """Line plot showing how sensitivities evolve over the trajectory."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # State sensitivities
    for j, name in enumerate(target_names):
      ax1.plot(time_fractions, state_over_time[:, j], label=name, linewidth=1.5)
    ax1.set_xlabel('Normalised time')
    ax1.set_ylabel('Mean |∂f/∂y_j|')
    ax1.set_title('State sensitivities')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Forcing sensitivities
    for j, name in enumerate(forcing_names):
      ax2.plot(time_fractions, forcing_over_time[:, j], label=name, linewidth=1.5)
    ax2.set_xlabel('Normalised time')
    ax2.set_ylabel('Mean |∂f/∂u_j|')
    ax2.set_title('Forcing sensitivities')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

  # ─── Batch Forward (for other analyses) ──────────────────────────────

  def _batch_forward_numpy(self, inputs_scaled, trues_scaled):
    """
    Run NODE forward pass on numpy arrays in batches.
    Returns: (num_runs, steps, n_target) numpy array of scaled predictions.
    """
    num_runs = inputs_scaled.shape[0]
    n_target = trues_scaled.shape[2]
    steps = inputs_scaled.shape[1]
    all_preds = np.zeros((num_runs, steps, n_target))
    
    device = self.device
    t_span = self.t_span.to(device)
    batch_size = 64
    
    with torch.no_grad():
      for batch_start in range(0, num_runs, batch_size):
        batch_end = min(batch_start + batch_size, num_runs)
        
        inputs_batch = torch.tensor(
          inputs_scaled[batch_start:batch_end], dtype=torch.float32, device=device
        )
        trues_batch = torch.tensor(
          trues_scaled[batch_start:batch_end], dtype=torch.float32, device=device
        )
        
        forcing_profiles = inputs_batch  # (batch, steps, n_input)
        y0 = trues_batch[:, 0, :]
        
        self.func.set_forcing(t_span, forcing_profiles)
        
        pred = self._odeint(y0, t_span).permute(1, 0, 2)
        
        all_preds[batch_start:batch_end] = pred.cpu().numpy()
    
    return all_preds

  # ─── Prediction Comparisons ──────────────────────────────────────────

  def _plot_prediction_comparisons(self, trues_unscaled, ar_preds_unscaled, 
                                   tf_preds_unscaled, target_names):
    """Plot teacher-forcing vs autoregressive predictions (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nPlotting prediction comparison for: {target_name}")
      
      # Use first run for comparison plot
      ground_truth = trues_unscaled[0, :, idx]
      teacher_forced_preds = tf_preds_unscaled[0, :, idx]
      autoregressive_preds = ar_preds_unscaled[0, :, idx]
      
      output_dir = os.path.join(self.result_dir, target_name)
      plot.plot_prediction_comparison(
        ground_truth, teacher_forced_preds,
        autoregressive_preds, target_name, output_dir
      )
      
      logger.info(f"  ✓ Comparison plot saved to: {output_dir}")

  # ─── Error Growth ───────────────────────────────────────────────────

  def _plot_error_growth(self, trues_unscaled, ar_preds_unscaled, 
                         tf_preds_unscaled, target_names):
    """Calculate and plot how prediction error grows over time (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("ERROR GROWTH ANALYSIS (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    num_runs = trues_unscaled.shape[0]
    logger.info(f"Analyzing error growth across {num_runs} runs")
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nAnalyzing error growth for: {target_name}")
      
      # Extract per-target data: (num_runs, steps)
      gt = trues_unscaled[:, :, idx]
      ar = ar_preds_unscaled[:, :, idx]
      tf = tf_preds_unscaled[:, :, idx]
      
      # MAE per run per timestep
      tf_mae_errors = np.abs(tf - gt)       # (num_runs, steps)
      ar_mae_errors = np.abs(ar - gt)
      
      # MALE per run per timestep
      epsilon = 1e-20
      tf_male_errors = np.abs(
        np.log10(np.abs(tf) + epsilon) - np.log10(np.abs(gt) + epsilon)
      )
      ar_male_errors = np.abs(
        np.log10(np.abs(ar) + epsilon) - np.log10(np.abs(gt) + epsilon)
      )
      
      # Statistics across runs for each timestep
      avg_tf_mae = np.mean(tf_mae_errors, axis=0)
      avg_ar_mae = np.mean(ar_mae_errors, axis=0)
      std_tf_mae = np.std(tf_mae_errors, axis=0)
      std_ar_mae = np.std(ar_mae_errors, axis=0)
      
      avg_tf_male = np.mean(tf_male_errors, axis=0)
      avg_ar_male = np.mean(ar_male_errors, axis=0)
      std_tf_male = np.std(tf_male_errors, axis=0)
      std_ar_male = np.std(ar_male_errors, axis=0)
      
      # Log final MALE values
      self.log(f'{target_name}/Final MALE (AR)', float(avg_ar_male[-1]))
      self.log(f'{target_name}/Final MALE (TF)', float(avg_tf_male[-1]))
      
      output_dir = os.path.join(self.result_dir, target_name)
      
      # Plot MAE over time
      plot.plot_error_growth_metric(
        avg_tf_mae, avg_ar_mae,
        std_tf_mae, std_ar_mae,
        target_name, output_dir, num_runs,
        metric_name='MAE',
        ylabel='Mean Absolute Error',
        skip_first_n=0,
        tf_errors_all=tf_mae_errors,
        ar_errors_all=ar_mae_errors
      )
      
      # Plot MALE over time
      plot.plot_error_growth_metric(
        avg_tf_male, avg_ar_male,
        std_tf_male, std_ar_male,
        target_name, output_dir, num_runs,
        metric_name='MALE',
        ylabel='Mean Absolute Log Error',
        skip_first_n=0,
        tf_errors_all=tf_male_errors,
        ar_errors_all=ar_male_errors
      )
      
      logger.info(f"  ✓ Error growth plots saved to: {output_dir}")

  # ── Predict ──────────────────────────────────────────────────────────

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    target_pred, target_true = self._forward_batch(batch)
    return {
        'pred': target_pred.squeeze(-1).cpu(),
        'true': target_true.squeeze(-1).cpu(),
    }

  # ── Optimizer ────────────────────────────────────────────────────────
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self.parameters(),
      lr=self.cfg.train.learning_rate,
      weight_decay=self.cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5,
      patience=self.cfg.train.lr_scheduler_patience,
    )

    logger.info(f"Optimizer: AdamW | LR: {self.cfg.train.learning_rate} | Weight Decay: {self.cfg.train.weight_decay}")
    logger.info(f"Scheduler: ReduceLROnPlateau | Patience: {self.cfg.train.lr_scheduler_patience} | Factor: 0.5")

    return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]