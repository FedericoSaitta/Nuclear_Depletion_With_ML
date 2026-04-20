# === Standard imports ===
import os
import torch
import torch.nn as nn
import numpy as np

# === External libraries ===
from loguru import logger
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error

from ML.utils import plot
from ML.utils import metrics
import ML.datamodule.data_scalers as data_scaler
from ML.models.model_architectures import Deep_Neural_Network
from ML.models.model_helper import get_loss_fn


class DNN_Model(L.LightningModule):
  def __init__(self, config_object):
    super().__init__()
    self.cfg = config_object

    # Model architecture
    self.n_inputs = len(config_object.dataset.inputs)
    self.n_outputs = len(config_object.dataset.targets)
    self.dropout_prob = config_object.model.dropout_probability
    self.NN_layers = config_object.model.layers
    self.activation = config_object.model.activation
    self.output_activation = config_object.model.output_activation
    self.residual_connections = config_object.model.residual_connections

    # Training configuration
    self.loss_fn = get_loss_fn(config_object.train.loss)

    ## If residual connections are turned on but layer inputs and outputs dont match they wont be used
    self.residual_map = [(in_size == out_size) for in_size, out_size in zip(self.NN_layers[:-1], self.NN_layers[1:])]
    if self.residual_connections and not any(self.residual_map):
      logger.error("Residual connections requested, but no adjacent layers have matching input/output dimensions.")
    
    # Keep track of losses
    self._init_tracking_variables()

    # Results
    self.result_dir = 'results/' + config_object.model.name + '/'

    
    # Creating the model
    self.model = Deep_Neural_Network(self.n_inputs, self.n_outputs, self.NN_layers, self.dropout_prob, self.activation, self.output_activation, self.residual_connections)
    
  # Gets called for each train batch
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
 
    loss = self.loss_fn(y_hat, y)
    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True) # doesnt get logged in sql databse on purpose
    return loss

  # Gets called for each train epoch
  def on_train_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["train_loss"].item()
    self.train_losses.append(epoch_loss)
  
  # Gets called when training ends
  def on_train_end(self):
    plot.plot_losses(self.train_losses, self.val_losses, self.result_dir)

  # Gets called for each validation batch
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    
    loss = self.loss_fn(y_hat, y)
    self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    
    # Unscale predictions and targets
    target_scaler = self.trainer.datamodule.target_scaler
    y_unscaled = data_scaler.inverse_transformer(target_scaler, y.cpu().numpy())
    y_hat_unscaled = data_scaler.inverse_transformer(target_scaler, y_hat.detach().cpu().numpy())
    
    # Store for manual R² calculation
    self.val_preds_epoch.append(y_hat_unscaled)
    self.val_targets_epoch.append(y_unscaled)
    
    return loss

  def on_validation_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["val_loss"].item()
    self.val_losses.append(epoch_loss)
    
    # Get predictions and targets
    all_preds = np.concatenate(self.val_preds_epoch, axis=0)
    all_targets = np.concatenate(self.val_targets_epoch, axis=0)
    
    # R² averaged per output
    r2 = r2_score(all_targets, all_preds, multioutput='uniform_average')
    self.log("val_r2", r2, prog_bar=True)
    self.val_r2_scores.append(r2)
    
    # MAE averaged per output
    mae = mean_absolute_error(all_targets, all_preds, multioutput='uniform_average')
    self.log("val_mae", mae, prog_bar=True)
    self.val_mae_scores.append(mae)
    
    # Reset for next epoch
    self.val_preds_epoch = []
    self.val_targets_epoch = []
    
  # Gets for each predict batch
  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    x, y = batch
    preds = self.model(x)

    # Move to CPU
    y_cpu = y.cpu().numpy()
    preds_cpu = preds.cpu().numpy()

    target_scaler = self.trainer.datamodule.target_scaler
    labels = data_scaler.inverse_transformer(target_scaler, y_cpu)
    predictions = data_scaler.inverse_transformer(target_scaler, preds_cpu)

    return {
      "labels": torch.from_numpy(labels), "predictions": torch.from_numpy(predictions) 
      }
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    
    # Store scaled inputs and targets
    self.test_inputs.append(x.cpu().numpy())
    self.test_targets_scaled.append(y.cpu().numpy())
    
    # Get predictions (already unscaled)
    res = self.predict_step(batch, batch_idx)
    self.test_predictions.append(res["predictions"].cpu().numpy())
    self.test_labels.append(res["labels"].cpu().numpy())


  def on_test_epoch_end(self):
    datamodule = self.trainer.datamodule
    
    # Prepare data
    test_data = self._prepare_test_data()
    y_true_test = test_data['labels']
    y_pred_test = test_data['predictions']
    
    target_names = list(datamodule.target.keys())
    logger.info(f"Test set shape - True: {y_true_test.shape}, Pred: {y_pred_test.shape}")
    
    # 1. Compute overall metrics
    mae_arr, rmse_arr, r2_arr = self._compute_and_log_overall_metrics(
      y_true_test, y_pred_test, target_names
    )
    
    # 2. Process each output
    per_target_metrics = []
    for idx, target_name in enumerate(target_names):
      self._process_single_output(
        idx, target_name,
        y_true_test[:, idx], y_pred_test[:, idx],
        mae_arr[idx], rmse_arr[idx], r2_arr[idx]
      )
      
      per_target_metrics.append({
        'name': target_name,
        'mae': float(mae_arr[idx]),
        'rmse': float(rmse_arr[idx]),
        'r2': float(r2_arr[idx])
      })
    
    # 3. MARE comparison
    ar_preds_dict, ar_truth_dict, steps_per_run = self._compute_mare_comparison(
      test_data['inputs'], test_data['targets_scaled'],
      y_true_test, y_pred_test, target_names, datamodule,
      per_target_metrics  # Pass to add MARE values
    )
    
    # 4. Feature importance
    self._compute_feature_importance(
      target_names, datamodule, datamodule.test_dataloader()
    )
    
    # 5. Prediction comparisons
    self._plot_prediction_comparisons(
        y_pred_test, ar_preds_dict, ar_truth_dict, 
        target_names, steps_per_run, datamodule.delta_conc,
        test_data['inputs'], datamodule
    )

    self._plot_error_growth(
        y_pred_test, ar_preds_dict, ar_truth_dict, 
        target_names, steps_per_run, datamodule.delta_conc,
        test_data['inputs'], datamodule)
    
    # 6. Log everything to database as a SINGLE ROW
    if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'update_final_results'):
      test_metrics = {
          'mae_avg': float(mae_arr.mean()),
          'rmse_avg': float(rmse_arr.mean()),
          'r2_avg': float(r2_arr.mean()),
          'per_target': per_target_metrics
      }
      
      self.trainer.logger.update_final_results(
          train_losses=self.train_losses,
          val_losses=self.val_losses,
          val_r2_scores=self.val_r2_scores,
          val_mae_scores=self.val_mae_scores,
          test_metrics=test_metrics
      )
        
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

  def _prepare_test_data(self):
    """Consolidate test data from batches."""
    return {
      'inputs': np.concatenate(self.test_inputs, axis=0),
      'targets_scaled': np.concatenate(self.test_targets_scaled, axis=0),
      'predictions': np.concatenate(self.test_predictions, axis=0),
      'labels': np.concatenate(self.test_labels, axis=0)
    }

  def _compute_and_log_overall_metrics(self, y_true, y_pred, target_names):
    """Compute and log averaged metrics across all outputs."""
    mae_per_output = metrics.mae(y_true, y_pred)
    rmse_per_output = metrics.rmse(y_true, y_pred)
    r2_per_output = metrics.r2(y_true, y_pred)
    
    self.log('Mean Absolute Error (avg)', float(mae_per_output.mean()))
    self.log('Root Mean Squared Error (avg)', float(rmse_per_output.mean()))
    self.log('R-squared coefficient (avg)', float(r2_per_output.mean()))
    
    return mae_per_output, rmse_per_output, r2_per_output
  
  def _process_single_output(self, idx, target_name, y_true, y_pred, mae_output, rmse_output, r2_output, steps_per_run=100):
    """Process and visualize results for a single output."""
    output_dir = os.path.join(self.result_dir, target_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot.plot_predictions_vs_actuals(y_true, y_pred, mae_output, rmse_output, r2_output, output_dir)
    plot.plot_residuals_combined(y_true, y_pred, output_dir, steps_per_run=steps_per_run)
    
    return output_dir
  
  def _compute_mare_comparison(self, X_test, Y_test, y_true_test, y_pred_test, target_names, datamodule, per_target_metrics=None):
    """Compare MARE between teacher-forcing and autoregressive modes."""
    logger.info(f"\n{'='*20}")
    logger.info("MARE: Teacher-Forcing vs Autoregressive")
    logger.info(f"{'='*20}")
    
    # Autoregressive predictions (returns deltas if delta_conc=True)
    steps_per_run = 100
    ar_predictions_dict, ar_ground_truth_dict = metrics.model_autoregress(
        self.model, X_test, Y_test, 
        datamodule.input_scaler, datamodule.target_scaler, 
        steps_per_run, datamodule.col_index_map, datamodule.target_index_map, 
        delta_conc=datamodule.delta_conc 
    )
    
    # Compute MARE for each target
    for idx, target_name in enumerate(target_names):
        # Get teacher-forced predictions and ground truth (deltas if delta_conc=True)
        Y_test_original = data_scaler.inverse_transformer(datamodule.target_scaler, Y_test)
        tf_gt_raw = Y_test_original[:, idx] if Y_test_original.ndim > 1 else Y_test_original
        tf_pred_raw = y_pred_test[:, idx] if y_pred_test.ndim > 1 else y_pred_test
        
        # Get autoregressive predictions and ground truth (deltas if delta_conc=True)
        ar_pred_raw = ar_predictions_dict[target_name]
        ar_gt_raw = ar_ground_truth_dict[target_name]
        
        if datamodule.delta_conc:
            # Convert deltas to absolute concentrations for MARE calculation
            
            # For teacher-forcing: need initial concentrations for each run
            n_runs = len(X_test) // steps_per_run
            tf_gt_absolute = []
            tf_pred_absolute = []
            
            for run in range(n_runs):
                run_start = run * steps_per_run
                run_end = run_start + steps_per_run
                
                # Get initial concentration from first timestep of this run
                X_first_unscaled = data_scaler.inverse_transformer(
                    datamodule.input_scaler, 
                    X_test[run_start].reshape(1, -1)
                )[0]
                
                if target_name in datamodule.col_index_map:
                    initial_conc = X_first_unscaled[datamodule.col_index_map[target_name]]
                    
                    # Convert deltas to absolute for this run
                    run_gt_deltas = tf_gt_raw[run_start:run_end]
                    run_pred_deltas = tf_pred_raw[run_start:run_end]
                    
                    run_gt_absolute = initial_conc + np.cumsum(run_gt_deltas)
                    run_pred_absolute = initial_conc + np.cumsum(run_pred_deltas)
                    
                    tf_gt_absolute.extend(run_gt_absolute)
                    tf_pred_absolute.extend(run_pred_absolute)
                else:
                    logger.warning(
                        f"{target_name} not in inputs - cannot convert to absolute. "
                        "Using deltas for MARE calculation."
                    )
                    tf_gt_absolute.extend(tf_gt_raw[run_start:run_end])
                    tf_pred_absolute.extend(tf_pred_raw[run_start:run_end])
            
            # Convert autoregressive deltas to absolute
            ar_gt_absolute = []
            ar_pred_absolute = []
            
            for run in range(n_runs):
                run_start = run * steps_per_run
                run_end = run_start + steps_per_run
                
                # Get initial concentration for this run
                X_first_unscaled = data_scaler.inverse_transformer(
                    datamodule.input_scaler, 
                    X_test[run_start].reshape(1, -1)
                )[0]
                
                if target_name in datamodule.col_index_map:
                    initial_conc = X_first_unscaled[datamodule.col_index_map[target_name]]
                    
                    # Convert deltas to absolute for this run
                    run_gt_deltas = ar_gt_raw[run_start:run_end]
                    run_pred_deltas = ar_pred_raw[run_start:run_end]
                    
                    run_gt_absolute = initial_conc + np.cumsum(run_gt_deltas)
                    run_pred_absolute = initial_conc + np.cumsum(run_pred_deltas)
                    
                    ar_gt_absolute.extend(run_gt_absolute)
                    ar_pred_absolute.extend(run_pred_absolute)
                else:
                    ar_gt_absolute.extend(ar_gt_raw[run_start:run_end])
                    ar_pred_absolute.extend(ar_pred_raw[run_start:run_end])
            
            # Convert to arrays
            tf_gt = np.array(tf_gt_absolute)
            tf_pred = np.array(tf_pred_absolute)
            ar_gt = np.array(ar_gt_absolute)
            ar_pred = np.array(ar_pred_absolute)
        else:
            # Already absolute concentrations
            tf_gt = tf_gt_raw
            tf_pred = tf_pred_raw
            ar_gt = ar_gt_raw
            ar_pred = ar_pred_raw

        # Calculate MARE on absolute concentrations
        mare_tf = metrics.mare(tf_gt, tf_pred)
        mare_ar = metrics.mare(ar_gt, ar_pred)
        
        self.log(f'{target_name}/MARE_TeacherForcing', float(mare_tf))
        self.log(f'{target_name}/MARE_Autoregressive', float(mare_ar))
        
        # ADD MARE VALUES TO per_target_metrics FOR DATABASE LOGGING
        if per_target_metrics is not None:
            per_target_metrics[idx]['mare_tf'] = float(mare_tf)
            per_target_metrics[idx]['mare_ar'] = float(mare_ar)
    
    return ar_predictions_dict, ar_ground_truth_dict, steps_per_run

  def _compute_feature_importance(self, target_names, datamodule, loader):
    """Compute and plot feature importance for each output."""
    logger.info(f"\n{'='*20}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*20}")
    
    feature_names = [
      key for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])
    ]
    
    for idx, target_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, target_name)
      logger.info(f"\nComputing feature importance for: {target_name}")
      
      # R²-based importance
      self._compute_and_plot_importance(
        loader, idx, feature_names, output_dir, 
        metric_config={'name': 'r2', 'direction': 'increasing'},
        metric_name='r2_score'
      )
      
      # MSE-based importance
      self._compute_and_plot_importance(
        loader, idx, feature_names, output_dir,
        metric_config={'name': 'mse', 'direction': 'decreasing'},
        metric_name='mse_score'
      )
      
      logger.info(f"  ✓ Feature importance plots saved to: {output_dir}")

  def _compute_and_plot_importance(self, loader, output_idx, feature_names, 
                                output_dir, metric_config, metric_name):
    """Helper to compute and plot a single importance metric."""
    logger.info(f"  Computing {metric_name}-based feature importance...")
    importance_means, importance_stds, baseline = metrics.calculate_feature_importance(
      self.model, loader, self.device, 
      n_repeats=5, metric=metric_config, output_idx=output_idx
    )
    plot.plot_feature_importance(
      importance_means, importance_stds, feature_names, 
      baseline, output_dir, metric_name, n_top=20
    )

  def _plot_prediction_comparisons(self, y_pred_test, ar_predictions_dict, 
                                  ar_ground_truth_dict, target_names, 
                                  steps_per_run, delta_conc, X_test, datamodule):
    """Plot teacher-forcing vs autoregressive predictions."""
    logger.info(f"\n{'='*20}")
    logger.info("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    first_run_slice = slice(0, steps_per_run)
    
    for idx, target_name in enumerate(target_names):
        logger.info(f"\nPlotting prediction comparison for: {target_name}")
        
        # Get data for first run (these are deltas if delta_conc=True)
        ground_truth_deltas = ar_ground_truth_dict[target_name][first_run_slice]
        autoregressive_deltas = ar_predictions_dict[target_name][first_run_slice]
        teacher_forced_deltas = (
            y_pred_test[first_run_slice, idx] 
            if y_pred_test.ndim > 1 
            else y_pred_test[first_run_slice]
        )
        
        if delta_conc:
            # Convert deltas to absolute concentrations for plotting
            # Get initial concentration from first timestep
            X_first_unscaled = data_scaler.inverse_transformer(
                datamodule.input_scaler, 
                X_test[0].reshape(1, -1)
            )[0]
            
            if target_name in datamodule.col_index_map:
                initial_conc = X_first_unscaled[datamodule.col_index_map[target_name]]
                
                # Convert to absolute using cumsum
                ground_truth = initial_conc + np.cumsum(ground_truth_deltas)
                autoregressive_preds = initial_conc + np.cumsum(autoregressive_deltas)
                teacher_forced_preds = initial_conc + np.cumsum(teacher_forced_deltas)
            else:
                logger.warning(
                    f"{target_name} not in inputs - cannot convert to absolute. "
                    "Plotting deltas instead."
                )
                ground_truth = ground_truth_deltas
                autoregressive_preds = autoregressive_deltas
                teacher_forced_preds = teacher_forced_deltas
        else:
            # Already absolute concentrations
            ground_truth = ground_truth_deltas
            autoregressive_preds = autoregressive_deltas
            teacher_forced_preds = teacher_forced_deltas
        
        # Save plot
        output_dir = os.path.join(self.result_dir, target_name)
        plot.plot_prediction_comparison(
            ground_truth, teacher_forced_preds, 
            autoregressive_preds, target_name, output_dir
        )
        
        logger.info(f"  ✓ Comparison plot saved to: {output_dir}")
        
  def _plot_error_growth(self, y_pred_test, ar_predictions_dict, 
                       ar_ground_truth_dict, target_names, 
                       steps_per_run, delta_conc, X_test, datamodule):
    """Calculate and plot how prediction error grows over time using multiple metrics."""
    logger.info(f"\n{'='*20}")
    logger.info("ERROR GROWTH ANALYSIS (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    num_runs = len(y_pred_test) // steps_per_run
    logger.info(f"Analyzing error growth across {num_runs} runs")
    
    for idx, target_name in enumerate(target_names):
        logger.info(f"\nAnalyzing error growth for: {target_name}")
        
        # Initialize arrays to store errors for each run
        tf_mae_errors = []
        ar_mae_errors = []
        tf_male_errors = []
        ar_male_errors = []
        tf_rmse_errors = []
        ar_rmse_errors = []
        
        for run_idx in range(num_runs):
            start_idx = run_idx * steps_per_run
            end_idx = start_idx + steps_per_run
            run_slice = slice(start_idx, end_idx)
            
            # Get data for this run
            ground_truth_deltas = ar_ground_truth_dict[target_name][run_slice]
            autoregressive_deltas = ar_predictions_dict[target_name][run_slice]
            teacher_forced_deltas = (
                y_pred_test[run_slice, idx] 
                if y_pred_test.ndim > 1 
                else y_pred_test[run_slice]
            )
            
            if delta_conc:
                X_first_unscaled = data_scaler.inverse_transformer(
                    datamodule.input_scaler, 
                    X_test[start_idx].reshape(1, -1)
                )[0]
                
                if target_name in datamodule.col_index_map:
                    initial_conc = X_first_unscaled[datamodule.col_index_map[target_name]]
                    
                    ground_truth = initial_conc + np.cumsum(ground_truth_deltas)
                    autoregressive_preds = initial_conc + np.cumsum(autoregressive_deltas)
                    teacher_forced_preds = initial_conc + np.cumsum(teacher_forced_deltas)
                else:
                    logger.warning(
                        f"{target_name} not in inputs - cannot convert to absolute. "
                        "Using deltas for error calculation."
                    )
                    ground_truth = ground_truth_deltas
                    autoregressive_preds = autoregressive_deltas
                    teacher_forced_preds = teacher_forced_deltas
            else:
                ground_truth = ground_truth_deltas
                autoregressive_preds = autoregressive_deltas
                teacher_forced_preds = teacher_forced_deltas
            
            # Calculate MAE (Mean Absolute Error)
            tf_mae = np.abs(teacher_forced_preds - ground_truth)
            ar_mae = np.abs(autoregressive_preds - ground_truth)
            tf_mae_errors.append(tf_mae)
            ar_mae_errors.append(ar_mae)
            
            # Calculate MALE (Mean Absolute Log Error)
            epsilon = 1e-20
            tf_male = np.abs(np.log10(np.abs(teacher_forced_preds) + epsilon) - 
                            np.log10(np.abs(ground_truth) + epsilon))
            ar_male = np.abs(np.log10(np.abs(autoregressive_preds) + epsilon) - 
                            np.log10(np.abs(ground_truth) + epsilon))
            tf_male_errors.append(tf_male)
            ar_male_errors.append(ar_male)
            
            # Calculate RMSE components (will take sqrt of mean later)
            tf_rmse = (teacher_forced_preds - ground_truth) ** 2
            ar_rmse = (autoregressive_preds - ground_truth) ** 2
            tf_rmse_errors.append(tf_rmse)
            ar_rmse_errors.append(ar_rmse)
        
        # Convert to arrays (shape: num_runs x steps_per_run)
        tf_mae_errors = np.array(tf_mae_errors)
        ar_mae_errors = np.array(ar_mae_errors)
        tf_male_errors = np.array(tf_male_errors)
        ar_male_errors = np.array(ar_male_errors)
        tf_rmse_errors = np.array(tf_rmse_errors)
        ar_rmse_errors = np.array(ar_rmse_errors)
        
        # Calculate statistics across runs for each timestep
        # MAE
        avg_tf_mae = np.mean(tf_mae_errors, axis=0)
        avg_ar_mae = np.mean(ar_mae_errors, axis=0)
        std_tf_mae = np.std(tf_mae_errors, axis=0)
        std_ar_mae = np.std(ar_mae_errors, axis=0)
        
        # MALE
        avg_tf_male = np.mean(tf_male_errors, axis=0)
        avg_ar_male = np.mean(ar_male_errors, axis=0)
        std_tf_male = np.std(tf_male_errors, axis=0)
        std_ar_male = np.std(ar_male_errors, axis=0)
        
        # Logging the final MALE value
        self.log(str(target_name) + '/Final MALE (AR)', float(avg_ar_male[-1]))
        self.log(str(target_name) + '/Final MALE (TF)', float(avg_tf_male[-1]))
        
        # Save plots
        output_dir = os.path.join(self.result_dir, target_name)
        
        # Plot 1: MAE over time
        plot.plot_error_growth_metric(
            avg_tf_mae, avg_ar_mae,
            std_tf_mae, std_ar_mae,
            target_name, output_dir, num_runs,
            metric_name='MAE',
            ylabel='Mean Absolute Error',
            skip_first_n=0,
            tf_errors_all=tf_mae_errors,  # ← Pass raw data
            ar_errors_all=ar_mae_errors   # ← Pass raw data
        )

        # Plot 2: MALE over time
        plot.plot_error_growth_metric(
            avg_tf_male, avg_ar_male,
            std_tf_male, std_ar_male,
            target_name, output_dir, num_runs,
            metric_name='MALE',
            ylabel='Mean Absolute Log Error',
            skip_first_n=0,
            tf_errors_all=tf_male_errors,  # ← Pass raw data
            ar_errors_all=ar_male_errors   # ← Pass raw data
        )

        logger.info(f"  ✓ Error growth plots saved to: {output_dir}")

  def _init_tracking_variables(self):
    """Initialize lists for tracking losses and test results."""
    self.train_losses = []
    self.val_losses = []
    self.val_r2_scores = []
    self.val_mae_scores = []
    self.test_predictions = []
    self.test_labels = []
    self.test_inputs = []
    self.test_targets_scaled = []
    # For manual R² calculation
    self.val_preds_epoch = []
    self.val_targets_epoch = []