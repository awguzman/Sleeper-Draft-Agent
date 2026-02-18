"""
This module defines the Dash application for the Sleeper Draft Agent dashboard.

It provides a web-based user interface to:
- Connect to a live Sleeper mock draft.
- View the current state of the draft (status, rosters).
- See the top available players as viewed by the agent.
- Receive a real-time draft recommendation from the trained PPO model when it's the user's turn.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import sys
import os
import glob

# Add project root to path to allow importing from `app` and `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.sleeper import SleeperDraftManager
from src import config

# --- Helper Functions ---
def get_available_models(models_dir="../src/models"):
    """Scans the models directory for .pth files and returns them as dropdown options."""
    if not os.path.isdir(models_dir):
        return []
    
    # Find all .pth files
    model_paths = glob.glob(os.path.join(models_dir, "*.pth"))
    
    # Format for dropdown: {'label': 'filename.pth', 'value': 'path/to/filename.pth'}
    options = [{'label': os.path.basename(p), 'value': p} for p in model_paths]
    
    return options

# --- Global State ---
manager: SleeperDraftManager = None
user_slot: int = None

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Sleeper Draft Agent"

# --- Markdown Content ---
explanation_text = """
The agent is a **Reinforcement Learning (PPO) model** trained to draft a fantasy football team. It learns by simulating 
tens of thousands of drafts and optimizing its drafting strategy for total team value.

The agent considers two main types of information:

1.  **Player Value (ECR and VOR):** It evaluates available players based on their **Value** (a normalized Expert 
    Consensus Rank from FantasyPros.com) and **VOR** (Value Over Replacement) which measures a player's impact relative 
    to a replacement-level player at their position.
2.  **Your Roster & Opponent Rosters:** It understands what positions you and your opponents have filled, and what 
    positions are still needed. This helps it identify positional scarcity, avoid over-drafting, and anticipate opponent
    picks.

The agent's recommendation is a learned balance of these factors, aiming to maximize your team's overall strength given 
the current draft context. It should not replace the users own judgement.
"""

# --- Layout ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Sleeper Draft Agent", className="text-center my-4"), width=12)
    ]),

    # Control Panel for connecting to a draft
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Sleeper Draft ID"),
                    dbc.Input(id="input-draft-id", placeholder="Enter Draft ID...", type="text"),
                ], width=4),
                dbc.Col([
                    dbc.Label("Your Draft Slot (1-12)"),
                    dbc.Input(id="input-user-slot", placeholder="1", type="number", min=1, max=12, value=1),
                ], width=2),
                dbc.Col([
                    dbc.Label("Select Model"),
                    dcc.Dropdown(
                        id='model-dropdown', 
                        options=get_available_models(), 
                        placeholder="Select a model...",
                        style={'color': 'black'}
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label(""),
                    dbc.Button("Connect", id="btn-connect", color="primary", className="w-100"),
                ], width=2, className="d-flex align-items-end"),
            ]),
            html.Div(id="connection-status", className="mt-2 text-info")
        ])
    ], className="mb-4"),

    # Main Content Area (hidden until a successful connection is made)
    html.Div(id="main-content", style={"display": "none"}, children=[
        
        # Interval component to trigger updates periodically
        dcc.Interval(id="interval-component", interval=10000, n_intervals=0), # Polls every 10 seconds

        # Top Row: Live status and the agent's recommendation
        dbc.Row([
            # Status Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Status"),
                    dbc.CardBody([
                        html.H4(id="status-pick-info", className="card-title"),
                        html.Hr(),
                        html.H5(id="status-on-clock", className="text-warning"),
                    ])
                ], className="h-100")
            ], width=4),

            # Recommendation Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Agent Recommendation"),
                    dbc.CardBody([
                        html.Div(id="rec-content", className="text-center")
                    ])
                ], className="h-100 border-success")
            ], width=8),
        ], className="mb-4"),

        # Bottom Row: Data tables for context
        dbc.Row([
            # Agent's Window (Top Available Players)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"Top {config.N_PLAYERS_WINDOW} Available"),
                    dbc.CardBody([
                        html.Div(id="table-available")
                    ])
                ])
            ], width=6),

            # User's Current Roster
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Your Roster"),
                    dbc.CardBody([
                        html.Div(id="table-roster")
                    ])
                ])
            ], width=6),
        ]),
        
        # Explanation Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("How the Agent Works"),
                    dbc.CardBody([
                        dcc.Markdown(explanation_text, className="text-muted small")
                    ])
                ], className="mt-4")
            ], width=12)
        ])
    ])
], fluid=True)

# --- Callbacks ---

@app.callback(
    [Output("connection-status", "children"),
     Output("main-content", "style")],
    [Input("btn-connect", "n_clicks")],
    [State("input-draft-id", "value"),
     State("input-user-slot", "value"),
     State("model-dropdown", "value")]
)
def connect_draft(n_clicks, draft_id, slot, model_path):
    """
    Callback triggered by the 'Connect' button.
    Initializes the SleeperDraftManager, sets the user's slot, and shows the main content area.
    
    :param n_clicks: Number of times the connect button has been clicked.
    :param draft_id: The ID of the Sleeper draft entered by the user.
    :param slot: The user's draft slot (1-12) entered by the user.
    :param model_path: The path to the selected model file from the dropdown.
    :return: A tuple containing the connection status message/alert and the style for the main content div.
    """
    global manager, user_slot
    
    if not n_clicks:
        # Initial load, do nothing
        return "", {"display": "none"}
    
    if not draft_id:
        return dbc.Alert("Please enter a Draft ID.", color="danger"), {"display": "none"}
    
    if not model_path:
        return dbc.Alert("Please select a model.", color="danger"), {"display": "none"}
    
    try:
        if not os.path.exists(model_path):
             return dbc.Alert(f"Error: Model not found at {model_path}", color="danger"), {"display": "none"}

        # Initialize the backend manager
        manager = SleeperDraftManager(draft_id, model_path)
        user_slot = int(slot)
        
        # Perform an initial state update
        manager.update_state()
        
        # Show the main content and a success message
        return dbc.Alert(f"Successfully connected to draft {draft_id}!", color="success"), {"display": "block"}
    except Exception as e:
        # Display error if connection or initialization fails
        return dbc.Alert(f"Connection failed: {str(e)}", color="danger"), {"display": "none"}

@app.callback(
    [Output("status-pick-info", "children"),
     Output("status-on-clock", "children"),
     Output("rec-content", "children"),
     Output("table-available", "children"),
     Output("table-roster", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    """
    Callback triggered periodically by the dcc.Interval component.
    This is the main update loop of the dashboard, responsible for:
    1. Polling the Sleeper API for the latest draft state.
    2. Calculating the current draft status (round, pick number, who is on the clock).
    3. Getting a draft recommendation from the agent for the current team on the clock.
    4. Generating and updating all the UI components (status, recommendation, tables).
    
    :param n: The number of times the interval has fired.
    :return: A tuple of updated children for the respective Output components.
    """
    global manager, user_slot
    
    # If manager is not initialized (not connected yet), do nothing
    if not manager:
        return dash.no_update
    
    # 1. Poll the API for the latest draft state
    manager.update_state()
    
    # 2. Calculate current draft status (round, pick, on-the-clock team)
    next_pick_num = manager.current_pick_no + 1
    current_round = ((next_pick_num - 1) // config.NUM_TEAMS) + 1
    pick_in_round = (next_pick_num - 1) % config.NUM_TEAMS
    
    # Determine who is on the clock using snake draft logic
    if current_round % 2 == 1:
        on_clock_idx = pick_in_round
    else:
        on_clock_idx = config.NUM_TEAMS - 1 - pick_in_round
        
    on_clock_slot = on_clock_idx + 1 # Convert to 1-indexed slot for display
    is_user_turn = (on_clock_slot == user_slot)
    
    status_text = f"Round {current_round} • Pick {next_pick_num} (Overall)"
    clock_text = f"On Clock: Team {on_clock_slot}" + (" (YOU)" if is_user_turn else "")
    
    # 3. Get a recommendation from the agent for the current team on the clock
    rec = manager.get_recommendation(on_clock_idx)
    
    # Format the recommendation for display
    rec_display = html.Div([
        html.H2(rec['Player'], className="display-4"),
        html.H4(f"{rec['Pos']} • {rec['Team']}"),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H5(f"ECR: {rec['ECR']:.2f}"), width=4),
            dbc.Col(html.H5(f"Value: {rec['Value']:.2f}"), width=4),
            dbc.Col(html.H5(f"VOR: {rec['VOR']:.2f}"), width=4),
        ])
    ])
    
    # 4. Generate data tables for display
    # Top Available Players Table
    top_n = manager.available_players.head(config.N_PLAYERS_WINDOW)
    df_avail = top_n.select(['Player', 'Pos', 'Team', 'ECR', 'Value', 'VOR']).to_pandas()
    table_avail = dbc.Table.from_dataframe(df_avail, striped=True, bordered=True, hover=True, size='sm', className="text-light")
    
    # User's Current Roster Table
    user_roster = manager.rosters[user_slot - 1] # Convert user_slot to 0-indexed team_idx
    roster_list = []
    for pos, players in user_roster.items():
        for p in players:
            roster_list.append({'Pos': pos, 'Player': p['Player'], 'Pick': p['PickNum']})
            
    if roster_list:
        df_roster = pd.DataFrame(roster_list)
        table_roster = dbc.Table.from_dataframe(df_roster, striped=True, bordered=True, hover=True, size='sm', className="text-light")
    else:
        table_roster = html.P("No players drafted yet.")

    # Return all updated components
    return status_text, clock_text, rec_display, table_avail, table_roster

if __name__ == '__main__':
    # Run the Dash application
    app.run(debug=False)
