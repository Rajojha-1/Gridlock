import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import PRIMARY_NAVY, ACCENT_YELLOW, VIOLATION_CLASSES

def plot_violation_breakdown(df):
    """
    Generates a Plotly Bar Chart showing the breakdown of detected violations.
    Only includes actual violations (defined in config.VIOLATION_CLASSES).
    """
    # Filter for actual violations
    violation_df = df[df['Violation Type'].isin(VIOLATION_CLASSES)]
    
    if violation_df.empty:
        # Return an empty placeholder figure with a nice message
        fig = go.Figure()
        fig.add_annotation(
            text="No violations detected to display in chart.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(
            title="Violation Type Breakdown",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Calculate counts
    counts = violation_df['Violation Type'].value_counts().reset_index()
    counts.columns = ['Violation Type', 'Count']

    # Draw vertical bar chart
    fig = px.bar(
        counts,
        x='Violation Type',
        y='Count',
        labels={'Violation Type': 'Violation Category', 'Count': 'Number of Incidents'},
        color_discrete_sequence=['#6366f1']  # Premium Indigo
    )
    
    # Premium layout styling
    fig.update_traces(
        marker_line_color='rgba(99, 102, 241, 0.5)',
        marker_line_width=1.5,
        opacity=0.9,
        texttemplate='%{y}',
        textposition='outside'
    )
    
    fig.update_layout(
        title=dict(
            text="Violation Category Analysis",
            font=dict(family="Outfit, sans-serif", size=18, color="#ffffff")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=40, r=20),
        xaxis=dict(
            showgrid=False,
            title_font=dict(family="Inter, sans-serif", size=12, color="#94a3b8"),
            tickfont=dict(family="Inter, sans-serif", size=11, color="#cbd5e1")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
            title_font=dict(family="Inter, sans-serif", size=12, color="#94a3b8"),
            tickfont=dict(family="Inter, sans-serif", size=11, color="#cbd5e1")
        ),
        hovermode="x unified"
    )
    
    return fig

def plot_confidence_distribution(df):
    """
    Generates a Plotly Histogram displaying the confidence score distribution of all detections.
    """
    if df.empty:
        # Return an empty placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text="No detections available to show distribution.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(
            title="Confidence Score Distribution",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Draw Histogram
    fig = px.histogram(
        df,
        x='Confidence %',
        nbins=10,
        labels={'Confidence %': 'Confidence (Percent)', 'count': 'Frequency'},
        color_discrete_sequence=['#fbbf24']  # Premium Gold/Amber
    )
    
    # Premium layout styling
    fig.update_traces(
        marker_line_color='rgba(251, 191, 36, 0.5)',
        marker_line_width=1.0,
        opacity=0.85
    )
    
    fig.update_layout(
        title=dict(
            text="Model Confidence Distribution",
            font=dict(family="Outfit, sans-serif", size=18, color="#ffffff")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=40, r=20),
        xaxis=dict(
            showgrid=False,
            title_font=dict(family="Inter, sans-serif", size=12, color="#94a3b8"),
            tickfont=dict(family="Inter, sans-serif", size=11, color="#cbd5e1")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
            title_font=dict(family="Inter, sans-serif", size=12, color="#94a3b8"),
            tickfont=dict(family="Inter, sans-serif", size=11, color="#cbd5e1")
        )
    )
    
    return fig
