import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from monte_carlo_portfolio import run_simulation, AssetClass, run_convergence_analysis
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
from reportlab.lib.utils import ImageReader
import plotly.io as pio

def create_plotly_figures(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_investment):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Asset Class Returns Distribution',
            'Portfolio Return Distribution',
            'Annual Withdrawals Over Time'
        ),
        vertical_spacing=0.12,
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

    # Withdrawal progression
    fig.add_trace(
        go.Scatter(
            x=years[1:],
            y=withdrawals,
            name='Annual Withdrawal',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Withdrawal: $%{y:,.0f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Asset class return distributions
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
    returns_hist, returns_bins = np.histogram(portfolio_returns.flatten(), bins=50)
    most_common_idx = np.argmax(returns_hist)
    most_common_range = f"{returns_bins[most_common_idx]:.1%} to {returns_bins[most_common_idx + 1]:.1%}"
    most_common_frequency = returns_hist[most_common_idx] / len(portfolio_returns.flatten()) * 100
    
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns.flatten() * 100,  # Convert to percentage
            name='Portfolio Returns',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Add annotation for most common range
    fig.add_annotation(
        text=f"Most Common: {most_common_range}<br>({most_common_frequency:.1f}% of returns)",
        xref="x3", yref="paper",
        x=returns_bins[most_common_idx] * 100,
        y=1,
        showarrow=True,
        arrowhead=1,
        row=3, col=1
    )

    # Portfolio returns Pareto (bottom right)
    counts, bins = np.histogram(portfolio_returns.flatten() * 100, bins=50)  # Convert to percentage
    cumsum = np.cumsum(counts)
    cumsum_norm = cumsum / cumsum[-1] * 100
    
    fig.add_trace(
        go.Scatter(
            x=bins[:-1],
            y=cumsum_norm,
            name='Portfolio Cumulative',
            line=dict(color='blue'),
            hovertemplate='Return: %{x:.1f}%<br>Cumulative: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )

    # Update axis labels to show percentages
    fig.update_xaxes(
        title_text="Return Rate (%)", 
        tickformat=".1f",  # Show as percentage with 1 decimal
        row=3, col=1
    )

    # Add 80% reference line for Pareto charts
    fig.add_hline(y=80, line=dict(color="red", width=1, dash="dash"),
                  row=3, col=1)

    # Update layout for better mobile viewing and legend readability
    fig.update_layout(
        height=1200,  # Increased height
        showlegend=True,
        title_text="Portfolio Monte Carlo Simulation",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            groupclick="toggleitem",
            itemsizing="constant",
            tracegroupgap=5  # Add gap between legend groups
        ),
        margin=dict(l=20, r=20, t=120, b=20),  # Increased top margin for legend
        font=dict(size=14)
    )

    # Update subplot titles font size and position
    for i, annotation in enumerate(fig.layout.annotations):
        annotation.update(
            font=dict(size=16),
            y=annotation.y + 0.02  # Move titles up slightly
        )

    # Group legends by subplot with proper naming
    for trace in fig.data:
        # Skip if trace name is None
        if trace.name is None:
            continue
            
        # Handle different types of traces
        if 'Cumulative' in trace.name:
            trace.showlegend = False
        elif trace.name in ['Median', '10th Percentile', '90th Percentile']:
            trace.legendgroup = "portfolio_value"
            trace.legendgrouptitle = dict(text="Portfolio Value")
        elif 'Annual Withdrawal' in trace.name:
            trace.legendgroup = "withdrawals"
            trace.legendgrouptitle = dict(text="Withdrawals")
        elif 'Portfolio Returns' in trace.name:
            trace.legendgroup = "portfolio_returns"
            trace.legendgrouptitle = dict(text="Portfolio Returns")
        elif any(asset.name in trace.name for asset in asset_classes):
            trace.legendgroup = "assets"
            trace.legendgrouptitle = dict(text="Asset Classes")

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
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
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
    
    # Add traces for portfolio values
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=medians,
            name='Median',
            line=dict(color='red'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p90,
            name='90th Percentile',
            line=dict(color='blue'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p10,
            name='10th Percentile',
            line=dict(color='green'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add risk of depletion on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=risk,
            name='Risk of Depletion',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Sims: %{x:,}<br>Risk: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout for better legend readability
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
            x=0.5,
            font=dict(size=12),
            itemsizing="constant"
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(size=14)
    )

    # Group convergence metrics in legend
    for trace in fig.data:
        if trace.name in ['Median', '90th Percentile', '10th Percentile']:
            trace.legendgroup = "portfolio"
            trace.legendgrouptitle = dict(text="Portfolio Values")
        elif trace.name == 'Risk of Depletion':
            trace.legendgroup = "risk"
            trace.legendgrouptitle = dict(text="Risk Metrics")

    # Update axes with larger font
    fig.update_xaxes(title_text="Number of Simulations", 
                    tickformat=",", 
                    title_font=dict(size=14))
    fig.update_yaxes(title_text="Portfolio Value ($)", 
                    tickformat="$,.0f", 
                    secondary_y=False,
                    title_font=dict(size=14))
    fig.update_yaxes(title_text="Risk of Depletion (%)", 
                    tickformat=".1f", 
                    secondary_y=True,
                    title_font=dict(size=14))
    
    return fig, median_changes, p90_changes, p10_changes, risk_changes

def generate_pdf_report(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_investment, years, convergence_results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Portfolio Simulation Report", title_style))
    story.append(Spacer(1, 12))
    
    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    final_values = results[:, -1]
    risk_of_depletion = np.mean(final_values < withdrawals[-1]) * 100
    
    metrics_data = [
        ["Initial Investment", f"${initial_investment:,.0f}"],
        ["Median Final Value", f"${np.median(final_values):,.0f}"],
        ["Risk of Depletion", f"{risk_of_depletion:.1f}%"],
        ["Median Annual Return", f"{np.median(portfolio_returns.flatten()) * 100:.1f}%"],
        ["Time Horizon", f"{years} years"]
    ]
    
    t = Table(metrics_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Asset Allocation
    story.append(Paragraph("Asset Allocation", styles['Heading2']))
    allocation_data = [["Asset Class", "Allocation", "Expected Return", "Std Dev"]]
    for asset in asset_classes:
        allocation_data.append([
            asset.name,
            f"{asset.allocation:.1%}",
            f"{asset.mean_return:.1%}",
            f"{asset.std_dev:.1%}"
        ])
    
    t = Table(allocation_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Convergence Analysis
    story.append(Paragraph("Convergence Analysis", styles['Heading2']))
    convergence_data = [["Simulations", "Median Change", "90th %ile Change", "10th %ile Change"]]
    for i in range(len(convergence_results) - 1):
        convergence_data.append([
            f"{convergence_results[i]['n_sims']:,} → {convergence_results[i+1]['n_sims']:,}",
            f"{abs((convergence_results[i+1]['median'] - convergence_results[i]['median']) / convergence_results[i]['median'] * 100):.2f}%",
            f"{abs((convergence_results[i+1]['percentile_90'] - convergence_results[i]['percentile_90']) / convergence_results[i]['percentile_90'] * 100):.2f}%",
            f"{abs((convergence_results[i+1]['percentile_10'] - convergence_results[i]['percentile_10']) / convergence_results[i]['percentile_10'] * 100):.2f}%"
        ])
    
    t = Table(convergence_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    
    # Add graphs section
    story.append(Spacer(1, 30))
    story.append(Paragraph("Portfolio Analysis Graphs", styles['Heading2']))
    
    try:
        # Create main simulation figure
        main_fig = create_plotly_figures(
            results, portfolio_returns, withdrawals, 
            asset_returns, asset_classes, initial_investment
        )
        
        # Create convergence figure
        conv_fig, _, _, _, _ = plot_convergence(convergence_results)
        
        # Save figures as images
        for fig, title in [(main_fig, "Portfolio Simulation Results"), 
                          (conv_fig, "Convergence Analysis")]:
            # Add title for each graph
            story.append(Spacer(1, 20))
            story.append(Paragraph(title, styles['Heading3']))
            story.append(Spacer(1, 10))
            
            # Convert Plotly figure to image with white background
            img_bytes = pio.to_image(
                fig, 
                format="png", 
                width=700, 
                height=500,
                scale=2.0,  # Higher resolution
                engine='kaleido'
            )
            img_buffer = io.BytesIO(img_bytes)
            
            # Add image to PDF
            img = ImageReader(img_buffer)
            story.append(Table(
                [[img]], 
                colWidths=[7*inch],
                style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]
            ))
            story.append(Spacer(1, 20))
    
    except Exception as e:
        # If there's an error with the graphs, add an error message
        story.append(Paragraph(f"Error generating graphs: {str(e)}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(
        page_title="Portfolio Monte Carlo Simulation", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Create header row with just the title
    header_col1, header_col2, header_col3 = st.columns([2, 4, 2])
    
    with header_col2:
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
        st.session_state.stocks_allocation = 30
    if 'bonds_allocation' not in st.session_state:
        st.session_state.bonds_allocation = 20
    if 'alts_allocation' not in st.session_state:
        st.session_state.alts_allocation = 25
    if 'private_allocation' not in st.session_state:
        st.session_state.private_allocation = 20
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
        stocks_max = st.number_input("Maximum Return (%)", 0, 50, 25, step=1, key="stocks_max")
    
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
    
    # Store the "Run Simulation" button state
    run_clicked = st.sidebar.button("Run Simulation")
    
    # Run simulation if button is clicked OR if this is the first load (no previous runs)
    if ('simulation_results' not in st.session_state) or (run_clicked and total_allocation == 100):
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
            
            # Run simulation and store results in session state
            results, portfolio_returns, withdrawals, asset_returns, _ = run_simulation(
                initial_investment=initial_investment,
                years=years,
                num_simulations=num_simulations,
                initial_withdrawal=initial_withdrawal,
                inflation_rate=inflation_rate,
                asset_classes=asset_classes
            )
            
            # Run convergence analysis
            convergence_results = run_convergence_analysis(
                initial_investment=initial_investment,
                years=years,
                initial_withdrawal=initial_withdrawal,
                inflation_rate=inflation_rate,
                asset_classes=asset_classes
            )
            
            # Store results in session state
            st.session_state.simulation_results = {
                'results': results,
                'portfolio_returns': portfolio_returns,
                'withdrawals': withdrawals,
                'asset_returns': asset_returns,
                'asset_classes': asset_classes,
                'initial_investment': initial_investment,
                'convergence_results': convergence_results  # Add convergence results to session state
            }
    
    # If we have simulation results, display them
    if 'simulation_results' in st.session_state:
        # Extract results from session state
        sim_data = st.session_state.simulation_results
        results = sim_data['results']
        portfolio_returns = sim_data['portfolio_returns']
        withdrawals = sim_data['withdrawals']
        asset_returns = sim_data['asset_returns']
        asset_classes = sim_data['asset_classes']
        initial_investment = sim_data['initial_investment']
        
        # Calculate and display statistics
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Base Metrics")
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
                    "Median Annual Return",
                    f"{np.median(portfolio_returns.flatten()) * 100:.1f}%"
                )
            
            with col2:
                st.subheader("Extreme Scenarios")
                st.metric(
                    "5th Percentile",
                    f"${np.percentile(final_values, 5):,.0f}",
                    f"{((np.percentile(final_values, 5) / initial_investment) ** (1/years) - 1) * 100:.1f}% CAGR"
                )
                st.metric(
                    "95th Percentile",
                    f"${np.percentile(final_values, 95):,.0f}",
                    f"{((np.percentile(final_values, 95) / initial_investment) ** (1/years) - 1) * 100:.1f}% CAGR"
                )
            
            with col3:
                st.subheader("Key Events")
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
        conv_fig, median_changes, p90_changes, p10_changes, risk_changes = plot_convergence(sim_data['convergence_results'])
        st.header("Convergence Analysis")
        
        # Display convergence plot
        st.plotly_chart(conv_fig, use_container_width=True)
        
        # Display changes table
        st.markdown("### Simulation Convergence")
        changes_df = pd.DataFrame({
            'Simulations': [f"{sim_data['convergence_results'][i]['n_sims']:,} → {sim_data['convergence_results'][i+1]['n_sims']:,}" 
                         for i in range(len(median_changes))],
            'Median Change': [f"{change:.2f}%" for change in median_changes],
            '90th Percentile': [f"{change:.2f}%" for change in p90_changes],
            '10th Percentile': [f"{change:.2f}%" for change in p10_changes],
            'Risk Change': [f"{change:.2f}%" for change in risk_changes]
        })
        
        st.dataframe(
            changes_df,
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    main() 