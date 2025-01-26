import streamlit as st
import numpy as np
from monte_carlo_portfolio import run_simulation, AssetClass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_plotly_figures(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_investment):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,  # Change to single column layout for mobile
        subplot_titles=(
            'Portfolio Value Over Time',
            'Asset Class Returns',
            'Portfolio Returns',
            'Annual Withdrawals'
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    years = np.arange(results.shape[1])
    
    # Portfolio value plot (top, spanning both columns)
    for i in range(min(100, len(results))):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=results[i],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.1)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Add median and percentiles
    median = np.median(results, axis=0)
    percentile_10 = np.percentile(results, 10, axis=0)
    percentile_90 = np.percentile(results, 90, axis=0)

    fig.add_trace(
        go.Scatter(
            x=years,
            y=median,
            name='Median',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=percentile_10,
            name='10th Percentile',
            line=dict(color='green', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=percentile_90,
            name='90th Percentile',
            line=dict(color='blue', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Withdrawal progression (top right)
    fig.add_trace(
        go.Scatter(
            x=years[1:],
            y=withdrawals,
            name='Annual Withdrawal',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Withdrawal: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Asset class return distributions (middle left)
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, asset in enumerate(asset_classes):
        fig.add_trace(
            go.Histogram(
                x=asset_returns[asset.name].flatten(),
                name=asset.name,
                nbinsx=50,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                hovertemplate='Return: %{x:.1%}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

    # Asset class returns Pareto (middle right)
    for i, asset in enumerate(asset_classes):
        returns = asset_returns[asset.name].flatten()
        counts, bins = np.histogram(returns, bins=50)
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / cumsum[-1] * 100
        
        fig.add_trace(
            go.Scatter(
                x=bins[:-1],
                y=cumsum_norm,
                name=f'{asset.name} Cumulative',
                line=dict(color=colors[i % len(colors)]),
                hovertemplate='Return: %{x:.1%}<br>Cumulative: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )

    # Portfolio return distribution (bottom left)
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns.flatten(),
            name='Portfolio Returns',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Return: %{x:.1%}<br>Count: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Portfolio returns Pareto (bottom right)
    counts, bins = np.histogram(portfolio_returns.flatten(), bins=50)
    cumsum = np.cumsum(counts)
    cumsum_norm = cumsum / cumsum[-1] * 100
    
    fig.add_trace(
        go.Scatter(
            x=bins[:-1],
            y=cumsum_norm,
            name='Portfolio Cumulative',
            line=dict(color='blue'),
            hovertemplate='Return: %{x:.1%}<br>Cumulative: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add 80% reference line for Pareto charts
    fig.add_hline(y=80, line=dict(color="red", width=1, dash="dash"),
                  row=3, col=1)

    # Update layout for better mobile viewing
    fig.update_layout(
        height=1000,  # Adjusted for mobile scrolling
        showlegend=True,
        title_text="Portfolio Monte Carlo Simulation",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=100, b=20)  # Tighter margins
    )

    # Make fonts more readable on mobile
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=16)
    )

    return fig

def run_convergence_analysis(initial_investment, years, initial_withdrawal, inflation_rate, asset_classes):
    """Run multiple simulations with increasing sample sizes to show convergence"""
    sample_sizes = [1000, 3000, 5000, 7000, 10000]
    convergence_results = []
    
    # Run simulations for each sample size
    for n_sims in sample_sizes:
        results, portfolio_returns, withdrawals, _, _ = run_simulation(
            initial_investment=initial_investment,
            years=years,
            num_simulations=n_sims,
            initial_withdrawal=initial_withdrawal,
            inflation_rate=inflation_rate,
            asset_classes=asset_classes
        )
        
        final_values = results[:, -1]
        convergence_results.append({
            'n_sims': n_sims,
            'median': np.median(final_values),
            'percentile_10': np.percentile(final_values, 10),
            'percentile_90': np.percentile(final_values, 90),
            'risk_of_depletion': np.mean(final_values < withdrawals[-1]) * 100
        })
    
    return convergence_results

def plot_convergence(convergence_results):
    """Create a single convergence analysis plot with all metrics"""
    # Create figure
    fig = go.Figure()
    
    # Extract data
    n_sims = [r['n_sims'] for r in convergence_results]
    medians = [r['median'] for r in convergence_results]
    p90 = [r['percentile_90'] for r in convergence_results]
    p10 = [r['percentile_10'] for r in convergence_results]
    risk = [r['risk_of_depletion'] for r in convergence_results]
    
    # Calculate relative changes
    def calc_relative_change(values):
        changes = []
        for i in range(1, len(values)):
            changes.append(abs((values[i] - values[i-1]) / values[i-1]) * 100)
        return changes
    
    median_changes = calc_relative_change(medians)
    p90_changes = calc_relative_change(p90)
    p10_changes = calc_relative_change(p10)
    risk_changes = calc_relative_change(risk)
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=medians,
            name='Median',
            line=dict(color='red'),
            hovertemplate='Sims: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p90,
            name='90th Percentile',
            line=dict(color='blue'),
            hovertemplate='Sims: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p10,
            name='10th Percentile',
            line=dict(color='green'),
            hovertemplate='Sims: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        )
    )
    
    # Add risk of depletion on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=risk,
            name='Risk of Depletion',
            line=dict(color='orange', dash='dash'),
            yaxis='y2',
            hovertemplate='Sims: %{x}<br>Risk: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Update layout for mobile
    fig.update_layout(
        height=500,
        title_text="Convergence Analysis",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(size=14)
    )
    
    return fig, median_changes, p90_changes, p10_changes, risk_changes

def main():
    st.set_page_config(
        page_title="Portfolio Monte Carlo Simulation", 
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with collapsed sidebar on mobile
    )
    
    st.title("Portfolio Monte Carlo Simulation")
    
    # Create a placeholder for the summary at the top
    summary_placeholder = st.empty()
    
    # Create sidebar for inputs
    st.sidebar.header("Simulation Parameters")
    
    # Basic parameters
    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=100000,
        max_value=100000000,
        value=7000000,
        step=100000,
        format="%d"
    )
    
    initial_withdrawal = st.sidebar.number_input(
        "Initial Annual Withdrawal ($)",
        min_value=10000,
        max_value=1000000,
        value=300000,
        step=10000,
        format="%d"
    )
    
    # Asset Allocation Section
    st.sidebar.header("Asset Allocation")
    
    # Initialize allocation variables in session state if they don't exist
    if 'stocks_allocation' not in st.session_state:
        st.session_state.stocks_allocation = 50
    if 'bonds_allocation' not in st.session_state:
        st.session_state.bonds_allocation = 15
    if 'alts_allocation' not in st.session_state:
        st.session_state.alts_allocation = 15
    if 'private_allocation' not in st.session_state:
        st.session_state.private_allocation = 15
    if 'cash_allocation' not in st.session_state:
        st.session_state.cash_allocation = 5
    
    # Create expandable section for each asset class
    with st.sidebar.expander(f"Stocks ({st.session_state.stocks_allocation}%)", expanded=True):
        stocks_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.stocks_allocation,
            step=5,
            key="stocks_alloc"
        )
        st.session_state.stocks_allocation = stocks_allocation
        
        stocks_return = st.number_input("Expected Return (%)", -10, 30, 10, step=1, key="stocks_return")
        stocks_std = st.number_input("Standard Deviation (%)", 0, 30, 16, step=1, key="stocks_std")
        stocks_min = st.number_input("Minimum Return (%)", -50, 0, -25, step=1, key="stocks_min")
        stocks_max = st.number_input("Maximum Return (%)", 0, 50, 35, step=1, key="stocks_max")
    
    with st.sidebar.expander(f"Bonds/MMF ({st.session_state.bonds_allocation}%)"):
        bonds_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.bonds_allocation,
            step=5,
            key="bonds_alloc"
        )
        st.session_state.bonds_allocation = bonds_allocation
        
        bonds_return = st.number_input("Expected Return (%)", -5, 15, 4, step=1, key="bonds_return")
        bonds_std = st.number_input("Standard Deviation (%)", 0, 20, 4, step=1, key="bonds_std")
        bonds_min = st.number_input("Minimum Return (%)", -10, 5, -1, step=1, key="bonds_min")
        bonds_max = st.number_input("Maximum Return (%)", 0, 20, 10, step=1, key="bonds_max")
    
    with st.sidebar.expander(f"Alternatives ({st.session_state.alts_allocation}%)"):
        alts_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.alts_allocation,
            step=5,
            key="alts_alloc"
        )
        st.session_state.alts_allocation = alts_allocation
        
        alts_return = st.number_input("Expected Return (%)", -5, 25, 11, step=1, key="alts_return")
        alts_std = st.number_input("Standard Deviation (%)", 0, 25, 12, step=1, key="alts_std")
        alts_min = st.number_input("Minimum Return (%)", -30, 0, -15, step=1, key="alts_min")
        alts_max = st.number_input("Maximum Return (%)", 0, 40, 30, step=1, key="alts_max")
    
    with st.sidebar.expander(f"Private Placements ({st.session_state.private_allocation}%)"):
        private_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.private_allocation,
            step=5,
            key="private_alloc"
        )
        st.session_state.private_allocation = private_allocation
        
        private_return = st.number_input("Expected Return (%)", 0, 40, 20, step=1, key="private_return")
        private_std = st.number_input("Standard Deviation (%)", 0, 40, 20, step=1, key="private_std")
        private_min = st.number_input("Minimum Return (%)", -50, 0, -30, step=1, key="private_min")
        private_max = st.number_input("Maximum Return (%)", 0, 60, 40, step=1, key="private_max")
    
    with st.sidebar.expander(f"Cash ({st.session_state.cash_allocation}%)"):
        cash_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.cash_allocation,
            step=5,
            key="cash_alloc"
        )
        st.session_state.cash_allocation = cash_allocation
        
        cash_return = st.number_input("Expected Return (%)", 0, 10, 2, step=1, key="cash_return")
        cash_std = st.number_input("Standard Deviation (%)", 0, 5, 1, step=1, key="cash_std")
        cash_min = st.number_input("Minimum Return (%)", 0, 5, 1, step=1, key="cash_min")
        cash_max = st.number_input("Maximum Return (%)", 0, 10, 4, step=1, key="cash_max")
    
    # Check if allocations sum to 100%
    total_allocation = (stocks_allocation + bonds_allocation + alts_allocation + 
                       private_allocation + cash_allocation)
    
    if total_allocation != 100:
        st.sidebar.error(f"Total allocation must equal 100% (currently {total_allocation}%)")
    
    # Simulation parameters
    st.sidebar.header("Simulation Settings")
    
    years = st.sidebar.slider(
        "Simulation Years",
        min_value=10,
        max_value=50,
        value=30
    )
    
    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=10000,
        value=5000
    )
    
    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    ) / 100
    
    # Run simulation button
    if st.sidebar.button("Run Simulation") and total_allocation == 100:
        with st.spinner("Running simulations and convergence analysis..."):
            # Create asset classes with user inputs
            asset_classes = [
                AssetClass("Stocks", stocks_return/100, stocks_std/100, 
                          stocks_min/100, stocks_max/100, stocks_allocation/100),
                AssetClass("Bonds/MMF", bonds_return/100, bonds_std/100,
                          bonds_min/100, bonds_max/100, bonds_allocation/100),
                AssetClass("Alternatives", alts_return/100, alts_std/100,
                          alts_min/100, alts_max/100, alts_allocation/100),
                AssetClass("Private Placements", private_return/100, private_std/100,
                          private_min/100, private_max/100, private_allocation/100),
                AssetClass("Cash", cash_return/100, cash_std/100,
                          cash_min/100, cash_max/100, cash_allocation/100)
            ]
            
            # Run simulation first to get the results
            results, portfolio_returns, withdrawals, asset_returns, _ = run_simulation(
                initial_investment=initial_investment,
                years=years,
                num_simulations=num_simulations,
                initial_withdrawal=initial_withdrawal,
                inflation_rate=inflation_rate,
                asset_classes=asset_classes
            )
            
            # Calculate statistics
            final_values = results[:, -1]
            risk_of_depletion = np.mean(final_values < withdrawals[-1]) * 100
            
            # Calculate year of depletion (first year portfolio hits zero)
            years_of_depletion = []
            for sim in results:
                depleted_years = np.where(sim <= 0)[0]
                if len(depleted_years) > 0:
                    years_of_depletion.append(depleted_years[0])
            
            # Calculate median year of depletion if it occurs
            if years_of_depletion:
                median_year_of_depletion = int(np.median(years_of_depletion))
                depletion_text = f"Year {median_year_of_depletion}"
            else:
                depletion_text = "Never"
            
            # Calculate year of escape (first year hitting $10MM)
            escape_threshold = 10_000_000
            years_of_escape = []
            for sim in results:
                escape_years = np.where(sim >= escape_threshold)[0]
                if len(escape_years) > 0:
                    years_of_escape.append(escape_years[0])
            
            # Calculate median year of escape if it occurs
            if years_of_escape:
                median_year_of_escape = int(np.median(years_of_escape))
                escape_text = f"Year {median_year_of_escape}"
                escape_probability = (len(years_of_escape) / len(results)) * 100
                escape_delta = f"{escape_probability:.1f}% probability"
            else:
                escape_text = "Never"
                escape_delta = "0% probability"
            
            # Display summary at the top using the placeholder
            with summary_placeholder.container():
                st.header("Simulation Summary")
                
                # Use full width for mobile
                st.metric(
                    "Median Portfolio Value",
                    f"${np.median(final_values):,.0f}",
                    f"Initial: ${initial_investment:,.0f}"
                )
                st.metric(
                    "Risk of Depletion",
                    f"{risk_of_depletion:.1f}%"
                )
                st.metric(
                    "Median Year of Depletion",
                    depletion_text,
                    f"{len(years_of_depletion) / len(results) * 100:.1f}% of simulations"
                )
                st.metric(
                    "Median Year of Escape ($10MM)",
                    escape_text,
                    escape_delta
                )
                
                # Asset allocation table with horizontal scroll
                st.subheader("Asset Allocation")
                allocation_data = {
                    "Asset": [asset.name for asset in asset_classes],
                    "Alloc": [f"{asset.allocation:.1%}" for asset in asset_classes],
                    "Return": [f"{asset.mean_return:.1%}" for asset in asset_classes],
                    "StdDev": [f"{asset.std_dev:.1%}" for asset in asset_classes],
                    "Min": [f"{asset.min_return:.1%}" for asset in asset_classes],
                    "Max": [f"{asset.max_return:.1%}" for asset in asset_classes]
                }
                st.dataframe(
                    allocation_data, 
                    hide_index=True,
                    use_container_width=True
                )
                
                st.divider()
            
            # Create and display main plot
            fig = create_plotly_figures(
                results,
                portfolio_returns,
                withdrawals,
                asset_returns,
                asset_classes,
                initial_investment
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add space between main simulation and convergence analysis
            st.markdown("---")
            
            # Run and display convergence analysis at the bottom
            convergence_results = run_convergence_analysis(
                initial_investment=initial_investment,
                years=years,
                initial_withdrawal=initial_withdrawal,
                inflation_rate=inflation_rate,
                asset_classes=asset_classes
            )
            
            st.header("Convergence Analysis")
            
            conv_col1, conv_col2 = st.columns([2, 1])
            
            with conv_col1:
                conv_fig, median_changes, p90_changes, p10_changes, risk_changes = plot_convergence(convergence_results)
                st.plotly_chart(conv_fig, use_container_width=True)
            
            with conv_col2:
                st.markdown("### Relative Changes")
                changes_df = pd.DataFrame({
                    'Iteration': [f"{convergence_results[i]['n_sims']:,} → {convergence_results[i+1]['n_sims']:,}" 
                                 for i in range(len(median_changes))],
                    'Median': [f"{change:.2f}%" for change in median_changes],
                    '90th': [f"{change:.2f}%" for change in p90_changes],
                    '10th': [f"{change:.2f}%" for change in p10_changes],
                    'Risk': [f"{change:.2f}%" for change in risk_changes]
                })
                
                st.dataframe(
                    changes_df,
                    hide_index=True,
                    column_config={
                        "Iteration": "Simulations",
                        "Median": "Median Value",
                        "90th": "90th Percentile",
                        "10th": "10th Percentile",
                        "Risk": "Risk of Depletion"
                    }
                )

if __name__ == "__main__":
    main() 