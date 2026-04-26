import os
import sys

def run():
    print("\n--- [9/9] Interactive Plotly Dash Dashboard ---")
    print("Dashboard placeholder. To run, use: python -m dash_app")
    
    dash_app_code = """
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

# Load data safely
DATA_PATH = os.path.join(os.path.dirname(__file__), 'output', 'youtube_cleaned.pkl')
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found. Please run the data pipeline first.")
    exit(1)

print("Loading data for dashboard (This may take a moment)...")
df = pd.read_pickle(DATA_PATH)

# For performance, if the dataset is massive, sample it for the heavy interactive plots
if len(df) > 10000:
    df_sample = df.sample(10000, random_state=42)
else:
    df_sample = df

app = dash.Dash(__name__, title="YouTube NLP Dashboard")

# Define the Layout
app.layout = html.Div(style={'font-family': 'Segoe UI, Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f9f9f9'}, children=[
    html.H1(children='📺 YouTube NLP Pipeline Dashboard', style={'textAlign': 'center', 'color': '#ff0000'}),
    
    html.Div([
        html.H3(f"Total Videos Analyzed: {len(df):,} | Total Views Tracked: {df['view_count'].sum():,}", 
                style={'textAlign': 'center', 'color': '#333', 'fontWeight': 'normal'})
    ]),
    
    html.Hr(),
    
    # Top Row: Pie Chart and Histogram
    html.Div([
        html.Div([
            html.H4("🌍 Language Distribution (Top 10)", style={'textAlign': 'center'}),
            dcc.Graph(
                id='lang-pie',
                figure=px.pie(df['detected_lang'].value_counts().reset_index().head(10), 
                              values='count', names='detected_lang', hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            )
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'padding': '10px'}),
        
        html.Div([
            html.H4("🕒 Upload Frequency by Hour", style={'textAlign': 'center'}),
            dcc.Graph(
                id='hour-hist',
                figure=px.histogram(df, x='upload_hour', nbins=24,
                                   color_discrete_sequence=['#1f77b4']).update_layout(bargap=0.1) if 'upload_hour' in df.columns else px.scatter(title="No upload_hour data available")
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'backgroundColor': 'white', 'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'padding': '10px'})
    ], style={'paddingBottom': '20px'}),
    
    html.Hr(),
    
    # Bottom Row: Interactive Scatter Plot
    html.Div([
        html.H4("📈 Engagement: Views vs. Likes (Log Scale)", style={'textAlign': 'center'}),
        html.Div([
            html.Label("Filter by Language:"),
            dcc.Dropdown(
                id='lang-filter',
                options=[{'label': 'All Languages', 'value': 'ALL'}] + [{'label': l, 'value': l} for l in df_sample['detected_lang'].value_counts().head(10).index],
                value='ALL',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'margin': 'auto', 'paddingBottom': '20px'}),
        
        dcc.Graph(id='scatter-views-likes')
    ], style={'backgroundColor': 'white', 'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'padding': '20px'})
])

# Callbacks for interactivity
@app.callback(
    Output('scatter-views-likes', 'figure'),
    [Input('lang-filter', 'value')]
)
def update_scatter(selected_lang):
    dff = df_sample
    if selected_lang != 'ALL':
        dff = dff[dff['detected_lang'] == selected_lang]
    
    fig = px.scatter(
        dff, x='view_count', y='like_count', 
        hover_data=['title'], color='detected_lang',
        log_x=True, log_y=True,
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(transition_duration=500)
    return fig

if __name__ == '__main__':
    # Run the server on port 8050
    app.run(debug=True, port=8050)
"""
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dash_app.py'), 'w', encoding='utf-8') as f:
        f.write(dash_app_code.strip())
    print("Generated fully functional dash_app.py for interactive exploration.")

if __name__ == "__main__":
    run()
