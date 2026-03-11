"""
This module defines the Dash application for the Sleeper Draft Agent dashboard.

It acts as the front end, providing the user with a full on AI-based decision system.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import sys
import os
import functools

# Add project root to path to allow importing from `app` and `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.sleeper import SleeperDraftManager, get_draft_metadata

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Sleeper Draft Agent"

# --- Memoized Function for Manager Creation ---
@functools.lru_cache(maxsize=10)
def load_draft_manager(draft_id, model_path):
    """
    Creates and returns a SleeperDraftManager instance.
    This function is memoized to ensure that the expensive model loading
    and object creation only happens once per draft_id/model_path combination.
    """
    print(f"--- Cache: Creating new manager for draft {draft_id} ---")
    return SleeperDraftManager(draft_id, model_path)

# --- Markdown Content ---
explanation_text = """
The agent is a **Reinforcement Learning (PPO) model** trained to draft a fantasy football team. It learns by simulating 
thousands of drafts and optimizing its drafting strategy for total team value.

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
    # Client-side storage for session data
    dcc.Store(id='session-data', storage_type='session'),

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
                ], width=6),
                dbc.Col([
                    dbc.Label("Your Draft Slot (1-12)"),
                    dbc.Input(id="input-user-slot", placeholder="1", type="number", min=1, max=12, value=1),
                ], width=3),
                dbc.Col([
                    dbc.Label(""),
                    dbc.Button("Connect", id="btn-connect", color="primary", className="w-100"),
                ], width=3, className="d-flex align-items-end"),
            ]),
            html.Div(id="connection-status", className="mt-2 text-info")
        ])
    ], className="mb-4"),

    # Main Content Area (hidden until a successful connection is made)
    html.Div(id="main-content", style={"display": "none"}, children=[
        
        # Interval component to trigger updates periodically
        dcc.Interval(id="interval-component", interval=10000, n_intervals=0, disabled=True), # Initially disabled

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
                        html.Div(id="rec-content", className="text-center"),
                        html.Hr(),
                        html.Div(id="rec-alternatives")
                    ])
                ], className="h-100 border-success")
            ], width=8),
        ], className="mb-4"),

        # Bottom Row: Data tables for context
        dbc.Row([
            # Agent's Window (Top Available Players)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"Top 25 Players Available"),
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
    [Output("session-data", "data"),
     Output("connection-status", "children"),
     Output("main-content", "style"),
     Output("interval-component", "disabled")],
    [Input("btn-connect", "n_clicks")],
    [State("input-draft-id", "value"),
     State("input-user-slot", "value")]
)
def connect_draft(n_clicks, draft_id, slot):
    """
    Callback triggered by the 'Connect' button.

    Validates the draft, finds the model, and saves session data to the dcc.Store.
    """
    if not n_clicks:
        # Initial load, do nothing
        return dash.no_update, "", {"display": "none"}, True
    
    if not draft_id:
        return dash.no_update, dbc.Alert("Please enter a Draft ID.", color="danger"), {"display": "none"}, True
    
    # Try to auto-detect model from Draft Metadata
    metadata = get_draft_metadata(draft_id)
    
    if not metadata:
        return dash.no_update, dbc.Alert("Could not fetch draft data from Sleeper. Please check the Draft ID.", color="danger"), {"display": "none"}, True

    num_teams = metadata['num_teams']
    num_rounds = metadata['num_rounds']
    roster_slots = metadata['roster_slots']
    auto_model_name = f"draft_agent_{num_teams}team_{num_rounds}rounds_{roster_slots['QB']}QB_{roster_slots['RB']}RB_{roster_slots['WR']}WR_{roster_slots['TE']}TE_{roster_slots['K']}K_{roster_slots['DST']}DST.pth"
    auto_model_path = os.path.join("..", "src", "models", auto_model_name)

    if not os.path.exists(auto_model_path):
        roster_str = ", ".join([f"'{k}': {v}" for k, v in roster_slots.items() if v > 0])
        error_msg = (
            f"Model not found for this draft configuration ({num_teams} teams, {num_rounds} rounds, Starters: {roster_slots['QB']} QB, {roster_slots['RB']} RB, {roster_slots['WR']} WR, {roster_slots['TE']} TE, {roster_slots['K']} K, {roster_slots['DST']} DST).\n"
            f"Please train a model for this draft scenario as described in the ReadMe.\n"
            f"\nBe sure to use the following settings in src/config.py:\n"
            f"\nNUM_TEAMS = {num_teams},\n"
            f"\nNUM_ROUNDS = {num_rounds},\n"
            f"\nROSTER_SLOTS = {{{roster_str}}}"
        )
        return dash.no_update, dbc.Alert(dcc.Markdown(error_msg), color="danger"), {"display": "none"}, True

    # Store the necessary data in the session.
    session_data = {
        'draft_id': draft_id,
        'user_slot': int(slot),
        'model_path': auto_model_path,
        'num_teams': num_teams
    }
    
    success_msg = dbc.Alert(f"Successfully connected to draft! Loaded draft agent configured for {num_teams} teams, {num_rounds} rounds, and with starters: {roster_slots['QB']} QB, {roster_slots['RB']} RB, {roster_slots['WR']} WR, {roster_slots['TE']} TE, {roster_slots['K']} K, {roster_slots['DST']} DST.",
                         color="success")
    
    return session_data, success_msg, {"display": "block"}, False # Enable interval on success


@app.callback(
    [Output("status-pick-info", "children"),
     Output("status-on-clock", "children"),
     Output("rec-content", "children"),
     Output("rec-alternatives", "children"),
     Output("table-available", "children"),
     Output("table-roster", "children")],
    [Input("interval-component", "n_intervals")],
    [State("session-data", "data")]
)
def update_dashboard(n, session_data):
    """
    Callback triggered periodically by the dcc.Interval component.
    This is the main update loop of the dashboard.
    """
    if not session_data:
        return dash.no_update

    # Use the memoized function to get the manager instance.
    manager = load_draft_manager(session_data['draft_id'], session_data['model_path'])
    user_slot = session_data['user_slot']
    num_teams = session_data['num_teams']

    # Poll the API for the latest draft state
    manager.update_state()

    # Calculate current draft status (round, pick, on-the-clock team)
    next_pick_num = manager.current_pick_no + 1
    current_round = ((next_pick_num - 1) // num_teams) + 1
    pick_in_round = (next_pick_num - 1) % num_teams
    
    # Determine who is on the clock using snake draft logic
    if current_round % 2 == 1:
        on_clock_idx = pick_in_round
    else:
        on_clock_idx = num_teams - 1 - pick_in_round
        
    on_clock_slot = on_clock_idx + 1 # Convert to 1-indexed for display
    is_user_turn = (on_clock_slot == user_slot)
    
    status_text = f"Round {current_round} • Pick {next_pick_num} (Overall)"
    clock_text = f"On Clock: Team {on_clock_slot}" + (" (YOU)" if is_user_turn else "")
    
    # Get a recommendations from the agent for the current team on the clock
    recs = manager.get_recommendation(on_clock_idx, top_k=5)
    
    if not recs:
        return status_text, clock_text, "No recommendations available.", "", "", ""

    top_rec = recs[0]
    
    # Format the top recommendation for display
    rec_display = html.Div([
        html.H2(top_rec['Player'], className="display-4"),
        html.H4(f"{top_rec['Pos']} • {top_rec['Team']}"),
        html.H5(f"Model Confidence: {top_rec['Confidence']:.1%}", className="text-success"),
        dbc.Row([
            dbc.Col(html.H5(f"ECR: {top_rec['ECR']:.2f}"), width=4),
            dbc.Col(html.H5(f"Value: {top_rec['Value']:.2f}"), width=4),
            dbc.Col(html.H5(f"VOR: {top_rec['VOR']:.2f}"), width=4),
        ])
    ])
    
    # Format the alternative options table
    if len(recs) > 1:
        df_alts = pd.DataFrame(recs[1:])
        # Format confidence as percentage string
        df_alts['Confidence'] = df_alts['Confidence'].apply(lambda x: f"{x:.1%}")
        # Select columns
        df_alts = df_alts[['Player', 'Pos', 'Team', 'ECR', 'VOR', 'Value', 'Confidence']]
        
        alt_display = html.Div([
            html.H5("Alternative Options", className="mt-3"),
            dbc.Table.from_dataframe(df_alts, striped=True, bordered=True, hover=True, size='sm', className="text-light small")
        ])
    else:
        alt_display = html.Div()

    # Top Available Players Table
    top_n = manager.available_players.head(25)
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

    return status_text, clock_text, rec_display, alt_display, table_avail, table_roster

if __name__ == '__main__':
    # Run the Dash application
    app.run(debug=False)
