# This file is used to write functions to help plot the training variables for the Machine Learning Models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

def plot_correlation_matrix(X, Y, col_index_map, target_name='Target', save_dir=None, name='correlation_matrix'):
  # Ensure Y is 1D
  if Y.ndim > 1 and Y.shape[1] == 1: Y = Y.flatten()
  
  # Get sorted feature names
  feature_list = sorted(col_index_map.items(), key=lambda x: x[1])
  feature_names = [name for name, _ in feature_list]
  
  # Combine X and Y into single array
  data_combined = np.column_stack([X, Y])
  all_names = feature_names + [f'{target_name}(t+1)']
  
  # Calculate correlation matrix
  corr_matrix = np.corrcoef(data_combined.T)
  
  # Create figure
  fig, ax = plt.subplots(figsize=(12, 10))
  
  # Create heatmap
  im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
  
  # Set ticks and labels
  ax.set_xticks(range(len(all_names)))
  ax.set_yticks(range(len(all_names)))
  ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
  ax.set_yticklabels(all_names, fontsize=9)
  
  # Add colorbar
  cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=11)
  
  # Add correlation values to cells
  for i in range(len(all_names)):
    for j in range(len(all_names)):
      corr_val = corr_matrix[i, j]
      # Choose text color based on background
      text_color = 'white' if abs(corr_val) > 0.5 else 'black'
      # Only show values for off-diagonal or if significant
      if i != j or abs(corr_val) > 0.01:
        ax.text(j, i, f'{corr_val:.2f}', 
                ha='center', va='center', 
                color=text_color, fontsize=7)
  
  # Highlight the target row and column
  target_idx = len(all_names) - 1

  ax.set_title(f'Correlation Matrix: All Features and {target_name}(t+1)', 
                fontsize=14, fontweight='bold', pad=20)
  
  plt.tight_layout()
  
  # Save plot
  filepath = os.path.join(save_dir, f'{name}.png')
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  plt.close()
  logger.info(f"Correlation matrix saved to: {filepath}")
  

def plot_data_distributions(X, col_index_map, save_dir=None, name='Raw_Data'):
  # Get all feature names sorted by index
  feature_list = sorted(col_index_map.items(), key=lambda x: x[1])
  feature_names = [name for name, _ in feature_list]
  
  # Calculate grid dimensions
  n_features = len(feature_names)
  n_cols = 4
  n_rows = int(np.ceil(n_features / n_cols))
  
  # Create figure - handle single feature case
  if n_features == 1:
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))
      axes = [ax]  # Wrap single axis in list
  else:
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
      axes = axes.flatten()
  
  # Plot histogram for each feature
  for idx, (feature_name, col_idx) in enumerate(feature_list):
      ax = axes[idx]
      
      # Handle 1D or 2D array
      if X.ndim == 1:
          data = X
      else:
          data = X[:, col_idx]
      
      # Create histogram
      n, bins, patches = ax.hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
      
      # Color code
      for patch in patches:
          patch.set_facecolor('steelblue')
      
      ax.set_title(feature_name)
      ax.set_xlabel('Value')
      ax.set_ylabel('Frequency')
      ax.grid(True, alpha=0.3)
      
      # Add statistics
      mean_val = np.mean(data)
      std_val = np.std(data)
      ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3e}')
      ax.legend(fontsize=8)
  
  # Hide unused subplots
  for idx in range(n_features, len(axes)):
      axes[idx].axis('off')
  
  plt.suptitle(f'{name} - Feature Distributions', 
                fontsize=16, fontweight='bold', y=1.0)
  plt.tight_layout()
  
  filepath = os.path.join(save_dir, name + '_distribution_features.png')
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  plt.close()
  logger.info(f"Feature distributions saved to: {filepath}")

def plot_losses(train_losses, val_losses, save_dir, nfes=None):
  eps = 1e-10
  log_train_losses = np.log10(np.array(train_losses) + eps)
  log_val_losses = np.log10(np.array(val_losses) + eps)

  fig, ax_loss = plt.subplots(figsize=(10, 6))

  ax_loss.plot(log_train_losses, label='Training Loss (log10)', linewidth=2, color='#2E86AB')
  ax_loss.plot(log_val_losses, label='Validation Loss (log10)', linewidth=2, color='#A23B72')
  ax_loss.set_xlabel('Epoch', fontsize=12)
  ax_loss.set_ylabel('Log10(Loss)', fontsize=12)
  ax_loss.grid(True, alpha=0.3)

  if nfes and len(nfes) == len(train_losses):
    ax_nfe = ax_loss.twinx()
    ax_nfe.plot(nfes, label='NFE', linewidth=1.5, color='#FF9800',
                linestyle='--', alpha=0.8)
    ax_nfe.set_ylabel('Function Evaluations (NFE)', fontsize=12, color='#FF9800')
    ax_nfe.tick_params(axis='y', labelcolor='#FF9800')

    # Combine legends from both axes
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_nfe, labels_nfe = ax_nfe.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_nfe, labels_loss + labels_nfe,
                   fontsize=10, loc='upper right')
  else:
    ax_loss.legend(fontsize=10)

  ax_loss.set_title('Training and Validation Loss Over Time', fontsize=14)
  plt.tight_layout()

  filepath = os.path.join(save_dir, 'training_loss_log.png')
  fig.savefig(filepath, dpi=300)
  plt.close(fig)

  logger.info(f"Logarithmic Train and Validation Losses saved to: {filepath}")


def plot_predictions_vs_actuals(actuals, predictions, mae, rmse, r2, plots_folder):
  plt.figure(figsize=(10, 8))
  plt.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
  
  # Perfect prediction line
  min_val = min(actuals.min(), predictions.min())
  max_val = max(actuals.max(), predictions.max())
  plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
  
  plt.xlabel('Actual Values', fontsize=12)
  plt.ylabel('Predicted Values', fontsize=12)
  plt.title(f'Predictions vs Actual Values\nR² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  filepath = os.path.join(plots_folder, 'predictions_vs_actual.png')
  plt.savefig(filepath, dpi=300)
  plt.close()
  logger.info(f"Model predictions saved to: {filepath}")


def plot_residuals_combined(actuals, predictions, plots_folder):
    residuals = actuals - predictions
    
    # ==========================================
    # PLOT 1: Linear Scale (Original)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Residual scatter
    axes[0].scatter(predictions, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4e}, Std: {residuals.std():.4e}', 
                      fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(plots_folder, 'residuals_combined.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Model prediction residuals (linear scale) saved to: {filepath}")
    
    # ==========================================
    # PLOT 2: Log-Log Scale (ABSOLUTE VALUES)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Use ABSOLUTE values for log plotting
    abs_predictions = np.abs(predictions)
    abs_residuals = np.abs(residuals)
    
    # Filter only zeros (log can't handle zero)
    mask_nonzero_pred = abs_predictions > 0
    mask_nonzero_res = abs_residuals > 0
    mask_combined = mask_nonzero_pred & mask_nonzero_res
    
    if np.sum(mask_combined) > 0:
        # Left: Residual scatter (log-log) with ABSOLUTE VALUES
        # Color by whether original prediction was positive or negative
        mask_pos_pred = (predictions > 0) & mask_combined
        mask_neg_pred = (predictions < 0) & mask_combined
        
        n_pos = np.sum(mask_pos_pred)
        n_neg = np.sum(mask_neg_pred)
        
        if n_pos > 0:
            axes[0].scatter(abs_predictions[mask_pos_pred], 
                           abs_residuals[mask_pos_pred], 
                           alpha=0.5, s=20, edgecolors='k', linewidth=0.5, 
                           color='blue', 
                           label=f'Positive predictions ({n_pos} pts)')
        
        if n_neg > 0:
            axes[0].scatter(abs_predictions[mask_neg_pred], 
                           abs_residuals[mask_neg_pred], 
                           alpha=0.5, s=20, edgecolors='k', linewidth=0.5, 
                           color='red', 
                           label=f'Negative predictions ({n_neg} pts)')
        
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('|Predicted Values| (log scale)', fontsize=12)
        axes[0].set_ylabel('|Residuals| (log scale)', fontsize=12)
        axes[0].set_title('Residual Plot (Log-Log Scale, Absolute Values)', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, which='both', alpha=0.3)
        
        # Add reference line: |residual| = |prediction| (100% error)
        pred_range = [abs_predictions[mask_combined].min(), 
                      abs_predictions[mask_combined].max()]
        axes[0].plot(pred_range, pred_range, 'k--', alpha=0.3, linewidth=1, 
                    label='100% error line')
        axes[0].legend(fontsize=10)
        
        # Right: Residual distribution (log scale)
        abs_residuals_valid = abs_residuals[mask_nonzero_res]
        
        if len(abs_residuals_valid) > 0:
            min_res = np.min(abs_residuals_valid)
            max_res = np.max(abs_residuals_valid)
            log_bins = np.logspace(np.log10(min_res), np.log10(max_res), 50)
            
            counts, bins, patches = axes[1].hist(abs_residuals_valid, bins=log_bins, 
                                                  edgecolor='black', alpha=0.7, 
                                                  color='steelblue')
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_xlabel('|Residuals| (log scale)', fontsize=12)
            axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
            
            median_abs_res = np.median(abs_residuals_valid)
            mean_abs_res = np.mean(abs_residuals_valid)
            axes[1].set_title(f'|Residual| Distribution (Log Scale)\n'
                            f'Median: {median_abs_res:.4e}, Mean: {mean_abs_res:.4e}', 
                            fontsize=14)
            axes[1].grid(True, which='both', alpha=0.3)
            
        else:
            axes[1].text(0.5, 0.5, 'No non-zero residuals to plot', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'Insufficient data for log-log plot\n(need non-zero predictions and residuals)', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Insufficient data for log-log plot', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    filepath_log = os.path.join(plots_folder, 'residuals_combined_loglog.png')
    plt.savefig(filepath_log, dpi=300)
    plt.close()
    logger.info(f"Model prediction residuals (log-log scale) saved to: {filepath_log}")

def plot_feature_importance(importance_means, importance_stds, feature_names, baseline, plots_folder, metric_name, n_top=20):
  n_features = len(importance_means)
  
  # Sort features by importance
  indices = np.argsort(importance_means)[::-1]
  
  # Plot top N features (or all if less than N)
  n_top = min(n_top, n_features)
  top_indices = indices[:n_top]
  
  plt.figure(figsize=(12, 8))
  colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_top))
  bars = plt.barh(range(n_top), importance_means[top_indices], 
                  xerr=importance_stds[top_indices], 
                  align='center', alpha=0.8, edgecolor='black',
                  color=colors)
  
  # Add exact values next to each bar
  for i, (idx, bar) in enumerate(zip(top_indices, bars)):
    value = importance_means[idx]
    std = importance_stds[idx]
    # Position text at the end of the bar
    plt.text(value + std + 0.003, i, f'{value:.4f}±{std:.4f}', va='center', fontsize=11, fontweight='bold')
  
  # Extend x-axis by 25% to fit text
  current_xlim = plt.xlim()
  plt.xlim(current_xlim[0], current_xlim[1] * 1.25)
  
  plt.yticks(range(n_top), [feature_names[i] for i in top_indices])
  plt.xlabel(f'Permutation Importance {metric_name}', fontsize=12)
  plt.ylabel('Features', fontsize=12)
  plt.title(f'Top {n_top} Most Important Features\nBaseline {metric_name} = {baseline:.4f}', fontsize=14)
  plt.gca().invert_yaxis()
  plt.grid(True, alpha=0.3, axis='x')
  plt.tight_layout()
  filepath = os.path.join(plots_folder, f'{metric_name}_importance.png')
  plt.savefig(filepath, dpi=300)
  plt.close()
  logger.info(f"{metric_name} Importances plot saved to: {filepath}")


def plot_prediction_comparison(ground_truth, teacher_forced_preds, autoregressive_preds, target_name, result_dir):
  # Calculate MARE for both prediction methods
  mare_teacher = np.mean(np.abs(ground_truth - teacher_forced_preds) / np.max(ground_truth))
  mare_autoregressive = np.mean(np.abs(ground_truth - autoregressive_preds) / np.max(ground_truth))
  
  # Create figure with two rows: main plot and residuals
  fig = plt.figure(figsize=(16, 9))
  
  # Create grid: 2 rows with height ratio 3:1 (main plot and residuals)
  gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
  
  # Main predictions plot
  ax1 = fig.add_subplot(gs[0])
  time_steps = np.arange(len(ground_truth)) + 1 # Shift them by one to the right as t=0 isnt predicted
  
  # Plot ground truth as solid line
  ax1.plot(time_steps, ground_truth, 'k-', label='Ground Truth', 
           linewidth=2.5, alpha=0.9, zorder=1)
  
  # Teacher-forced with square markers
  ax1.plot(time_steps, teacher_forced_preds, 'b-', 
           linewidth=2, alpha=0.8, zorder=2)
  ax1.plot(time_steps, teacher_forced_preds, 
           's', color='blue', markersize=5, markerfacecolor='blue', 
           markeredgecolor='white', markeredgewidth=0.5,
           label=f'Teacher-Forced (MARE: {mare_teacher:.4f})', zorder=3)
  
  # Autoregressive with triangle markers
  ax1.plot(time_steps, autoregressive_preds, 'r-', 
           linewidth=2, alpha=0.8, zorder=2)
  ax1.plot(time_steps, autoregressive_preds, 
           '^', color='red', markersize=5, markerfacecolor='red', 
           markeredgecolor='white', markeredgewidth=0.5,
           label=f'Autoregressive (MARE: {mare_autoregressive:.4f})', zorder=3)
  
  ax1.set_ylabel(f'{target_name}', fontsize=14, fontweight='bold')
  ax1.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black')
  ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
  ax1.set_title(f'{target_name} - Prediction Comparison', fontsize=16, fontweight='bold', pad=15)
  ax1.tick_params(labelbottom=False, labelsize=11)
  
  # Residuals plot (both methods)
  ax2 = fig.add_subplot(gs[1], sharex=ax1)
  residuals_teacher = ground_truth - teacher_forced_preds
  residuals_autoregressive = ground_truth - autoregressive_preds

  # Plot residuals with thicker lines and markers
  ax2.plot(time_steps, residuals_teacher, 'b-', linewidth=1.8, alpha=0.8, zorder=2)
  ax2.plot(time_steps, residuals_teacher, 
           's', color='blue', markersize=5, markerfacecolor='blue', 
           markeredgecolor='white', markeredgewidth=0.5, 
           label='Teacher-Forced', zorder=3)
  
  ax2.plot(time_steps, residuals_autoregressive, 'r-', linewidth=1.8, alpha=0.8, zorder=2)
  ax2.plot(time_steps, residuals_autoregressive, 
           '^', color='red', markersize=6, markerfacecolor='red', 
           markeredgecolor='white', markeredgewidth=0.5,
           label='Autoregressive', zorder=3)
  
  ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
  ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
  ax2.set_xlabel('Time Steps', fontsize=14, fontweight='bold')
  ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
  ax2.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='black')
  ax2.tick_params(labelsize=11)
  
  plt.savefig(os.path.join(result_dir, f'{target_name}_prediction_comparison.png'), dpi=300, bbox_inches='tight')
  plt.close()

def plot_error_growth_metric(avg_tf_error, avg_ar_error, std_tf_error, std_ar_error,
                             target_name, output_dir, num_runs, 
                             metric_name='MALE', ylabel='Error', skip_first_n=0,
                             tf_errors_all=None, ar_errors_all=None):  # ← Add raw data
  """Plot error growth over time for a specific metric, with both linear and log scales."""
  os.makedirs(output_dir, exist_ok=True)
  
  # Skip first N steps if requested
  if skip_first_n > 0:
      time_steps = np.arange(skip_first_n, len(avg_tf_error))
      avg_tf_error = avg_tf_error[skip_first_n:]
      avg_ar_error = avg_ar_error[skip_first_n:]
      std_tf_error = std_tf_error[skip_first_n:]
      std_ar_error = std_ar_error[skip_first_n:]
      if tf_errors_all is not None:
          tf_errors_all = tf_errors_all[:, skip_first_n:]
          ar_errors_all = ar_errors_all[:, skip_first_n:]
  else:
      time_steps = np.arange(len(avg_tf_error))
  
  # Create both linear and log scale versions
  for scale_type in ['linear', 'log']:
      fig, ax = plt.subplots(figsize=(12, 7))
      
      # Plot mean errors
      ax.plot(time_steps, avg_tf_error, label='Teacher-Forcing (mean)', 
              color='#2E86AB', linewidth=2.5, alpha=0.9, zorder=5)
      ax.plot(time_steps, avg_ar_error, label='Autoregressive (mean)', 
              color='#A23B72', linewidth=2.5, alpha=0.9, zorder=5)
      
      # Add shaded regions
      if scale_type == 'linear':
          # LINEAR SCALE: Use mean ± std (arithmetic)
          ax.fill_between(time_steps, 
                          avg_tf_error - std_tf_error,
                          avg_tf_error + std_tf_error,
                          alpha=0.2, color='#2E86AB', label='TF ±1σ')
          ax.fill_between(time_steps,
                          avg_ar_error - std_ar_error,
                          avg_ar_error + std_ar_error,
                          alpha=0.2, color='#A23B72', label='AR ±1σ')
      else:
          # LOG SCALE: Use percentiles (proper error propagation)
          if tf_errors_all is not None and ar_errors_all is not None:
              # Compute 16th and 84th percentiles (equivalent to ±1σ for normal distribution)
              tf_lower = np.percentile(tf_errors_all, 16, axis=0)
              tf_upper = np.percentile(tf_errors_all, 84, axis=0)
              ar_lower = np.percentile(ar_errors_all, 16, axis=0)
              ar_upper = np.percentile(ar_errors_all, 84, axis=0)
              
              ax.fill_between(time_steps, tf_lower, tf_upper,
                              alpha=0.2, color='#2E86AB', label='TF 16-84%ile')
              ax.fill_between(time_steps, ar_lower, ar_upper,
                              alpha=0.2, color='#A23B72', label='AR 16-84%ile')
          else:
              # Fallback: Use multiplicative factors (geometric std)
              # This is approximate but better than arithmetic on log scale
              tf_factor = np.exp(std_tf_error / (avg_tf_error + 1e-20))
              ar_factor = np.exp(std_ar_error / (avg_ar_error + 1e-20))
              
              ax.fill_between(time_steps,
                              avg_tf_error / tf_factor,
                              avg_tf_error * tf_factor,
                              alpha=0.2, color='#2E86AB', label='TF approx. ±1σ')
              ax.fill_between(time_steps,
                              avg_ar_error / ar_factor,
                              avg_ar_error * ar_factor,
                              alpha=0.2, color='#A23B72', label='AR approx. ±1σ')
      
      ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
      
      # Set scale and labels
      if scale_type == 'log':
          ax.set_yscale('log')
          ax.set_ylabel(f'{ylabel} (log scale)', fontsize=13, fontweight='bold')
          title_suffix = ' (Log Scale)'
      else:
          ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
          title_suffix = ''
      
      ax.set_title(f'{metric_name} Growth Over Time: {target_name}{title_suffix}\n(Averaged over {num_runs} runs)', 
                    fontsize=14, fontweight='bold', pad=15)
      ax.legend(fontsize=10, loc='best', framealpha=0.9)
      ax.grid(True, alpha=0.3, linestyle='--')
      
      plt.tight_layout()
      
      filename = os.path.join(output_dir, f'{target_name}_{metric_name}_growth_{scale_type}.png')
      plt.savefig(filename, dpi=300, bbox_inches='tight')
      plt.close()
      
      print(f"{metric_name} growth plot ({scale_type} scale) saved: {filename}")


# ─── Plotting Trajectories ────────────────────────────

def plot_node_trajectory(t, u238_pred, u238_true, power, title, save_path):
    """Three-panel plot: power forcing, U238 prediction vs truth, residual."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 8), height_ratios=[1, 2, 1],
        sharex=True, gridspec_kw={'hspace': 0.1},
    )

    ax1.plot(t, power, color='tab:orange', linewidth=1.0)
    ax1.set_ylabel('Power (scaled)')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, u238_true, label='Truth', linewidth=1.5)
    ax2.plot(t, u238_pred, label='Prediction', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('U238 (scaled)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    residuals = u238_pred - u238_true
    ax3.plot(t, residuals, color='tab:red', linewidth=0.8)
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_ylabel('Residual')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)

    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"NODE trajectory plot saved to: {save_path}")


def plot_node_trajectory_summary(t, all_preds, all_trues, title, save_path):
    """Overlay multiple trajectory predictions with mean absolute residual."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.05},
    )

    for i in range(len(all_preds)):
        ax1.plot(t, all_trues[i], color='tab:blue', alpha=0.3, linewidth=0.5,
                 label='Truth' if i == 0 else None)
        ax1.plot(t, all_preds[i], color='tab:red', alpha=0.3, linewidth=0.5,
                 label='Prediction' if i == 0 else None)

    ax1.set_ylabel('U238 (scaled)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    all_residuals = np.array(all_preds) - np.array(all_trues)
    mean_abs_res = np.mean(np.abs(all_residuals), axis=0)
    ax2.plot(t, mean_abs_res, color='tab:red', linewidth=1.0)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Mean |Residual|', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"NODE trajectory summary saved to: {save_path}")
