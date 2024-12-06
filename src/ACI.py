import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        calibration_end=calibration_end
    )
    print("finish computing ACI bounds")
    
    # Define plotting period (adjust these dates as needed)
    plot_start = lead_time_1.index.min() # pd.Timestamp('2021-07-16')
    plot_end = lead_time_1.index.max() # pd.Timestamp('2021-07-19')
    
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
        save_path=visual_dir / f"aci_predictions_{plot_start.date()}_{plot_end.date()}.html",  # Save as interactive HTML
        # Or save as static image:
        # save_path="aci_predictions.png", save_format="png"
    )
    
    
    

    
    
