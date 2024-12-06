import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

def compute_aci_bounds(
    observations: pd.Series,
    predictions: pd.Series,
    calibration_end: pd.Timestamp,
    alpha: float = 0.1
) -> Tuple[List[float], List[float]]:
    """
    Compute Adaptive Conformal Inference bounds for time series predictions.
    
    Args:
        observations: Series of observed values
        predictions: Series of predicted values
        calibration_end: End timestamp of calibration period
        alpha: Significance level (default 0.1 for 90% confidence)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    # Initialize calibration window
    calibration_mask = observations.index < calibration_end
    window_size = calibration_mask.sum()
    
    # Initialize residuals with calibration period
    initial_residuals = np.abs(
        observations[calibration_mask] - predictions[calibration_mask]
    )
    residuals = list(initial_residuals)
    
    # Store results
    lower_bounds = []
    upper_bounds = []
    
    # Process test period
    test_indices = observations.index >= calibration_end
    
    for idx in observations[test_indices].index:
        # Calculate new residual
        new_residual = np.abs(observations[idx] - predictions[idx])
        
        # Update residuals array
        residuals.append(new_residual)
        if len(residuals) > window_size:
            residuals.pop(0)
            
        # Compute quantile
        quantile_idx = np.ceil((window_size + 1) * (1 - alpha)) / window_size
        q_alpha = np.quantile(residuals, quantile_idx)
        
        # Generate prediction intervals
        current_pred = predictions[idx]
        lower_bounds.append(current_pred - q_alpha)
        upper_bounds.append(current_pred + q_alpha)
    
    # Add NaNs for calibration period
    lower_bounds = [np.nan] * window_size + lower_bounds
    upper_bounds = [np.nan] * window_size + upper_bounds
    
    return lower_bounds, upper_bounds


def plot_predictions_with_bounds(
    time_index: pd.DatetimeIndex,
    observations: pd.Series,
    predictions: pd.Series,
    lower_bounds: List[float],
    upper_bounds: List[float],
    calibration_end: pd.Timestamp,
    title: str = "Adaptive Conformal Prediction Intervals",
    plot_start: pd.Timestamp = None,
    plot_end: pd.Timestamp = None
) -> None:
    """
    Plot the predictions with conformal prediction bounds for a specified time period.
    
    Args:
        time_index: DatetimeIndex for x-axis
        observations: Series of observed values
        predictions: Series of predicted values
        lower_bounds: List of lower bound values
        upper_bounds: List of upper bound values
        calibration_end: End timestamp of calibration period
        title: Plot title
        plot_start: Start timestamp for plotting (optional)
        plot_end: End timestamp for plotting (optional)
    """
    plt.figure(figsize=(15, 8))
    
    # Filter data if plot range is specified
    if plot_start is not None and plot_end is not None:
        mask = (time_index >= plot_start) & (time_index <= plot_end)
        plot_time = time_index[mask]
        plot_obs = observations[mask]
        plot_pred = predictions[mask]
        plot_lower = np.array(lower_bounds)[mask]
        plot_upper = np.array(upper_bounds)[mask]
    else:
        plot_time = time_index
        plot_obs = observations
        plot_pred = predictions
        plot_lower = lower_bounds
        plot_upper = upper_bounds
    
    # Plot observations and predictions
    plt.plot(plot_time, plot_obs, label="Observations", color='blue')
    plt.plot(plot_time, plot_pred, label="Predictions", color='red')
    
    # Plot confidence intervals
    plt.fill_between(plot_time, plot_lower, plot_upper, 
                    color='gray', alpha=0.3, 
                    label="Conformal Prediction Interval")
    
    # Add vertical line for calibration/test split if it's within the plot range
    if (plot_start is None or calibration_end >= plot_start) and \
       (plot_end is None or calibration_end <= plot_end):
        plt.axvline(x=calibration_end, color='k', linestyle='--', 
                   label='Calibration/Test Split')
    
    plt.xlabel("Time")
    plt.ylabel("Water Level (cm)")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    
    # Set x-axis limits if specified
    if plot_start is not None and plot_end is not None:
        plt.xlim(plot_start, plot_end)
    
    plt.show()


if __name__ == "__main__":
    # Load data
    work_dir = Path(r'c:\Users\deng_jg\work\09conformal_prediction\LobithNN_conformal')
    experiment_dir = work_dir / 'data' / 'experiment'
    pred_fn = experiment_dir / "main_stations_min_maxau_rees_PRED.csv"
    
    pred = pd.read_csv(pred_fn, parse_dates=['time'], index_col=['time', 'lead_time'])
    
    # Extract lead_time=1 predictions and adjust timestamps
    lead_time_1 = pred.xs(1, level='lead_time').copy()
    lead_time_1.index = lead_time_1.index + pd.Timedelta(hours=1)  # Adjust timestamps
    
    # Define calibration period as 1 year from start of data
    start_time = lead_time_1.index.min()
    calibration_end = start_time + pd.DateOffset(years=1)
    
    # Compute ACI bounds
    lower_bounds, upper_bounds = compute_aci_bounds(
        observations=lead_time_1['obs'],
        predictions=lead_time_1['sim'],
        calibration_end=calibration_end
    )
    
    # Define plotting period (adjust these dates as needed)
    plot_start = pd.Timestamp('2020-02-07')
    plot_end = pd.Timestamp('2020-02-07')
    
    # Plot results
    plot_predictions_with_bounds(
        time_index=lead_time_1.index,
        observations=lead_time_1['obs'],
        predictions=lead_time_1['sim'],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        calibration_end=calibration_end,
        plot_start=plot_start,
        plot_end=plot_end,
        title=f"ACI Prediction Intervals ({plot_start.date()} to {plot_end.date()})"
    )
    
    
    

    
    
