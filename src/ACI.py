import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ACI
def compute_aci_bounds(
    observations: pd.Series,
    predictions: pd.Series,
    calibration_end: pd.Timestamp,
    alpha: float = 0.1,
    gamma: float = 1e-3
) -> Tuple[List[float], List[float]]:
    """
    Compute Adaptive Conformal Inference bounds for time series predictions.
    
    Args:
        observations: Series of observed values
        predictions: Series of predicted values
        calibration_end: End timestamp of calibration period
        alpha: Significance level (default 0.1 for 90% confidence)
        gamma: Update rate for alpha (default 1e-3)
        
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
    
    # Initialize alpha_t for the desired confidence level
    alpha_t = alpha
    
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
        quantile_idx = np.ceil((window_size + 1) * (1 - alpha_t)) / window_size
        quantile_idx = np.clip(quantile_idx, 0, 1) # clip quantile_idx to be between 0 and 1
        assert quantile_idx >= 0 and quantile_idx <= 1, f'Quantile value should be between 0 and 1, but is {quantile_idx}'
        q_alpha_t = np.quantile(residuals, quantile_idx)
        assert not np.isnan(q_alpha_t), f'Quantile value should not be NaN'
        
        # Check if the observation is bounded
        is_lower_bounded = observations[idx] > predictions[idx] - q_alpha_t
        is_upper_bounded = observations[idx] < predictions[idx] + q_alpha_t
        is_not_bounded = not (is_lower_bounded and is_upper_bounded)
        
        # Adjust alpha_t dynamically relative to desired coverage
        alpha_t += gamma * (alpha - is_not_bounded)
        alpha_t = np.clip(alpha_t, 0, 1)  # Ensure alpha_t stays in [0, 1]
        
        # Generate prediction intervals
        current_pred = predictions[idx]
        lower_bounds.append(current_pred - q_alpha_t)
        upper_bounds.append(current_pred + q_alpha_t)
    
    # Add NaNs for calibration period
    lower_bounds = [np.nan] * window_size + lower_bounds
    upper_bounds = [np.nan] * window_size + upper_bounds
    
    return lower_bounds, upper_bounds


# metrics
def PICP(
    y_true: pd.Series, 
    lower: List[float], 
    upper: List[float]
) -> float:
    """Prediction Interval Coverage Probability
    
    Args:
        y_true: Series of true values
        lower: List of lower bound values
        upper: List of upper bound values
        
    Returns:
        Tuple of (PICP for each sample, mean PICP)
    """
    # Calculate PICP for each timestep
    in_interval = np.logical_and(y_true >= lower, y_true <= upper)
    interval_PICP = np.mean(in_interval)
    
    return interval_PICP

def PIAW(
    lower: List[float], 
    upper: List[float]
) -> float:
    """Prediction Interval Average Width
    
    Args:
        y_true: Series of true values
        lower: List of lower bound values
        upper: List of upper bound values
        
    Returns:
        Tuple of (PIAW for each timestep, mean PIAW)
    """
    # Calculate PIAW for each timestep
    widths = np.array(upper) - np.array(lower)
    mean_width = np.mean(widths)
    
    return mean_width

def _winkler(
    y_true: float, 
    lower: float, 
    upper: float, 
    alpha: float
) -> float:
    """Winkler Score
    Source:https://www.kaggle.com/datasets/carlmcbrideellis/winkler-interval-score-metric
    Args:
        y_true: True value
        lower: Lower bound of the prediction interval
        upper: Upper bound of the prediction interval
        alpha: Significance level (e.g., 0.1 for 90% confidence interval)
        
    Returns:
        Winkler score for the given prediction interval
    """
    # Ensure inputs are valid
    assert not np.isnan(y_true), "y_true contains NaN value(s)"
    assert not np.isinf(y_true), "y_true contains inf value(s)"
    assert not np.isnan(lower), "lower interval value contains NaN value(s)"
    assert not np.isinf(lower), "lower interval value contains inf value(s)"
    assert not np.isnan(upper), "upper interval value contains NaN value(s)"
    assert not np.isinf(upper), "upper interval value contains inf value(s)"
    assert 0 < alpha <= 1, f"alpha should be (0,1]. Found: {alpha}"

    # Calculate Winkler score
    score = np.abs(upper - lower)
    if y_true < lower:
        score += (2 / alpha) * (lower - y_true)
    elif y_true > upper:
        score += (2 / alpha) * (y_true - upper)
    
    return score

def evaluate(
    y_true: pd.Series, 
    lower: List[float], 
    upper: List[float], 
    alpha: float
) -> dict[str, float]:
    picp = PICP(y_true, lower, upper)
    piaw = PIAW(lower, upper)
    winkler = np.vectorize(_winkler)  # vectorize the winkler function
    winkler_scores = winkler(y_true, lower, upper, alpha)
    return {
        "PICP": picp,
        "PIAW": piaw,
        "mean_Winkler": np.mean(winkler_scores)
    }


# plotting
def plot_predictions_with_bounds(
    time_index: pd.DatetimeIndex,
    observations: pd.Series,
    predictions: pd.Series,
    lower_bounds: List[float],
    upper_bounds: List[float],
    calibration_end: pd.Timestamp,
    title: str = "Adaptive Conformal Prediction Intervals",
    plot_start: pd.Timestamp = None,
    plot_end: pd.Timestamp = None,
    save_path: str = None,
    save_format: str = 'html'
) -> go.Figure:
    """
    Plot the predictions with conformal prediction bounds using Plotly.
    
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
        save_path: Path where to save the figure (optional)
        save_format: Format to save the figure ('html' or 'png', 'jpg', 'pdf', 'svg')
    
    Returns:
        Plotly figure object
    """
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

    # Create figure
    fig = go.Figure()

    # Add confidence interval as a filled area
    fig.add_trace(
        go.Scatter(
            x=plot_time,
            y=plot_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_time,
            y=plot_lower,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(128,128,128,0.3)',
            fill='tonexty',
            name='Conformal Prediction Interval',
            showlegend=True
        )
    )

    # Add observations
    fig.add_trace(
        go.Scatter(
            x=plot_time,
            y=plot_obs,
            mode='lines',
            name='Observations',
            line=dict(color='blue'),
            showlegend=True
        )
    )

    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=plot_time,
            y=plot_pred,
            mode='lines',
            name='Predictions',
            line=dict(color='red'),
            showlegend=True
        )
    )

    # Add vertical line for calibration/test split if it's within the plot range
    if (plot_start is None or calibration_end >= plot_start) and \
       (plot_end is None or calibration_end <= plot_end):
        fig.add_shape(
            type="line",
            x0=calibration_end,
            x1=calibration_end,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=calibration_end,
            y=1,
            yref="paper",
            text="Calibration/Test Split",
            showarrow=False,
            yanchor="bottom"
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Discharge (m3/s)",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )

    # Set x-axis range if specified
    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])

    # Save the figure if save_path is provided
    if save_path:
        if save_format.lower() == 'html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)

    # Return the figure object
    return fig


if __name__ == "__main__":
    # Load data
    work_dir = Path(r'c:\Users\deng_jg\work\09conformal_prediction\LobithNN_conformal')
    experiment_dir = work_dir / 'data' / 'experiment'
    pred_fn = experiment_dir / "main_stations_min_maxau_rees_PRED.csv"
    visual_dir = work_dir / 'data' / 'visualization'
    visual_dir.mkdir(exist_ok=True, parents=True)  # Create visualization directory if it doesn't exist
    
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
        calibration_end=calibration_end,
        gamma=3e-3
    )
    print("finish computing ACI bounds")

    # evaluation
    evaluation = evaluate(
        y_true=lead_time_1['obs'],
        lower=lower_bounds,
        upper=upper_bounds,
        alpha=0.1
    )
    print(evaluation)

    # Define plotting period (adjust these dates as needed)
    # plot_start = pd.Timestamp('2021-07-16') # lead_time_1.index.min()
    # plot_end = pd.Timestamp('2021-07-19') # lead_time_1.index.max()

    plot_start = pd.Timestamp('2020-02-07')
    plot_end = pd.Timestamp('2020-02-08')
    
    # Plot results
    fig = plot_predictions_with_bounds(
        time_index=lead_time_1.index,
        observations=lead_time_1['obs'],
        predictions=lead_time_1['sim'],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        calibration_end=calibration_end,
        plot_start=plot_start,
        plot_end=plot_end,
        title=f"ACI Prediction Intervals ({plot_start.date()} to {plot_end.date()})",
        save_path=visual_dir / f"aci_predictions_{plot_start.date()}_{plot_end.date()}_gamma.html",  # Save as interactive HTML
        # Or save as static image:
        # save_path="aci_predictions.png", save_format="png"
    )

    
    
    

    
    
