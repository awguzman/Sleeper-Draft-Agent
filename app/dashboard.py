"""
This module defines the Dash application for the Sleeper Draft Agent dashboard.

It acts as the front end, providing the user with a full on AI-based decision system.
"""

# --- Dash Imports
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# --- Other Imports ---
import pandas as pd
import os
import functools
import time
import re
import logging

from app.sleeper import SleeperDraftManager, get_draft_metadata

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.title = "Sleeper Draft Agent"

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Global Rate Limiter & Session Timeout Variables ---
LAST_REQUEST_TIME = 0
RATE_LIMIT_SECONDS = 2.0  # Allow 1 connection request every 2 seconds
SESSION_TIMEOUT_MINUTES = 60 # Pause polling after 60 minutes

# --- Helper Functions ---
def get_available_models():
    """
    Scans the models directory and returns a list of available model configurations.
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    available = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pth"):
            available.append(translate_model(filename))
    return available

def translate_model(model: str):
    """
    Translates a model filename into a user-readable string.
    """
    match = re.search(r'(\d+)team_(\d+)rounds_(\d+)QB_(\d+)RB_(\d+)WR_(\d+)TE_(\d+)K_(\d+)DST', model)

    num_teams = match.group(1)
    num_rounds = match.group(2)
    roster_slots = {'QB': match.group(3),
                    'RB': match.group(4),
                    'WR': match.group(5),
                    'TE': match.group(6),
                    'K': match.group(7),
                    'DST': match.group(8)
                    }

    translation =f"{num_teams} teams, {num_rounds} rounds with starters ({roster_slots['QB']}QB's, {roster_slots['RB']}RB's, {roster_slots['WR']}WR's, {roster_slots['TE']}TE's, {roster_slots['K']}K's, {roster_slots['DST']}DST's)"
    return translation

# --- Memoized Function for Manager Creation ---
@functools.lru_cache(maxsize=10)
def load_draft_manager(draft_id, model_path):
    """
    Creates and returns a SleeperDraftManager instance.
    This function is memoized to ensure that the expensive model loading
    and object creation only happens once per draft_id/model_path combination.

    NOTE: maxsize is set to 10 assuming low user counts! May need to increase in the future.
    """
    logger.info(f"Creating new draft manager for draft ID {draft_id}")
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
                    dbc.Label("Your Draft Slot"),
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
        
        # Interval component to trigger updates periodically (every 30 seconds)
        dcc.Interval(id="interval-component", interval=30000, n_intervals=0, disabled=True), # Initially disabled

        # Session Timeout/Resume Controls
        dbc.Row([
            dbc.Col(
                html.Div(id="session-control-area", className="text-center my-2"),
                width=12
            )
        ]),

        # Top Row: Live status and the agent's recommendation
        dbc.Row([
            # Status Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Status"),
                    dbc.CardBody([
                        html.H3(id="status-pick-info", className="card-title"),
                        html.Hr(),
                        html.H4(id="status-on-clock", className="text-warning"),
                        html.H4(id='picks_away_info', className='text-warning')
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
        ]),

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
    global LAST_REQUEST_TIME

    if not n_clicks:
        # Initial load, do nothing
        return dash.no_update, "", {"display": "none"}, True

    current_time = time.time()
    if current_time - LAST_REQUEST_TIME < RATE_LIMIT_SECONDS:
        logger.warning(f"Rate limit hit. Request rejected for ID: {draft_id}")
        return dash.no_update, dbc.Alert(f"Rate limit hit. Request rejected for ID: {draft_id}", color="danger"), {"display": "none"}, True

    # Update the timestamp only if we are about to make an external API call
    LAST_REQUEST_TIME = current_time

    # Log the attempt to load draft data
    logger.info(f"Attempting to load league data for ID: {draft_id}")
    
    if not draft_id:
        return dash.no_update, dbc.Alert("Please enter a Draft ID.", color="danger"), {"display": "none"}, True

    # Verify draft_id is a digit
    if not draft_id.isdigit():
        return dash.no_update, dbc.Alert("Invalid Draft ID. Draft ID must be an integer.", color="danger"), {"display": "none"}, True

    # Try to auto-detect model from Draft Metadata
    metadata = get_draft_metadata(draft_id)
    
    if not metadata:
        logger.error(f"Failed to fetch draft metadata for ID: {draft_id}")
        return dash.no_update, dbc.Alert("Could not fetch draft data from Sleeper. Please check the Draft ID.", color="danger"), {"display": "none"}, True

    num_teams = metadata['num_teams']
    num_rounds = metadata['num_rounds']
    roster_slots = metadata['roster_slots']
    draft_users = metadata['users']

    # Validate slot input
    if not slot:
        return dash.no_update, dbc.Alert(f"Invalid Draft Slot. It must be an integer between 1 and {num_teams}.", color="danger"), {"display": "none"}, True
    if not slot.is_integer():
        return dash.no_update, dbc.Alert(f"Invalid Draft Slot. It must be an integer between 1 and {num_teams}.",
                                         color="danger"), {"display": "none"}, True
    if not (1 <= slot <= num_teams):
        return dash.no_update, dbc.Alert(f"Invalid Draft Slot. It must be an integer between 1 and {num_teams}", color="danger"), {"display": "none"}, True

    # Retrieve the model path for the detected model
    auto_model_name = f"draft_agent_{num_teams}team_{num_rounds}rounds_{roster_slots['QB']}QB_{roster_slots['RB']}RB_{roster_slots['WR']}WR_{roster_slots['TE']}TE_{roster_slots['K']}K_{roster_slots['DST']}DST.pth"
    auto_model_path = os.path.join(MODELS_DIR, auto_model_name)

    # Error message for missing model
    if not os.path.exists(auto_model_path):
        logger.warning(f"Model not found for this draft configuration: {auto_model_name}")
        
        available_models = get_available_models()
        model_list = [html.Li(model) for model in available_models]
        error_msg = html.Div([
            f"Model not found for this draft configuration: {translate_model(auto_model_name)}.",
            html.Br(),
            "Models are available for the following draft scenarios:",
            html.Ul(model_list)
        ])

        return dash.no_update, dbc.Alert(error_msg, color="danger"), {"display": "none"}, True

    # Store the necessary data in the session.
    session_data = {
        'draft_id': draft_id,
        'user_slot': int(slot),
        'model_path': auto_model_path,
        'num_teams': num_teams,
        'users': draft_users,
        'last_active': time.time()
    }
    
    success_msg = dbc.Alert(f"Successfully connected to draft! Loaded draft agent configured for {num_teams} teams, {num_rounds} rounds, and with starters: {roster_slots['QB']} QB, {roster_slots['RB']} RB, {roster_slots['WR']} WR, {roster_slots['TE']} TE, {roster_slots['K']} K, {roster_slots['DST']} DST.",
                         color="success")
    logger.info(f"Successfully connected to draft ID {draft_id}")
    
    return session_data, success_msg, {"display": "block"}, False # Enable interval on success


@app.callback(
    [Output("status-pick-info", "children"),
     Output("status-on-clock", "children"),
     Output("picks_away_info", "children"),
     Output("rec-content", "children"),
     Output("rec-alternatives", "children"),
     Output("table-available", "children"),
     Output("table-roster", "children"),
     Output("session-control-area", "children"),
     Output("interval-component", "disabled", allow_duplicate=True)],
    [Input("interval-component", "n_intervals"),
     Input("session-data", "data")], #  Trigger immediately on connection
     prevent_initial_call=True

)
def update_dashboard(n, session_data):
    """
    Callback triggered by either the dcc.Interval component or a session data update.

    This is the main update loop of the dashboard.
    """
    if not session_data:
        return dash.no_update

    # --- Session Timeout ---
    last_active = session_data.get('last_active', 0)
    if (time.time() - last_active) > (SESSION_TIMEOUT_MINUTES * 60):
        logger.warning(f"Session for draft {session_data['draft_id']} timed out.")
        pause_button = dbc.Button("Resume Polling", id="btn-resume", color="warning", className="w-50")
        pause_message = dbc.Alert("Polling paused. Are you still Drafting?", color="warning", className="mt-2")
        return dash.no_update, dash.no_update, pause_message, "", "", "", pause_button, True # Disable interval

    # Use the memoized function to get the manager instance.
    manager = load_draft_manager(session_data['draft_id'], session_data['model_path'])
    user_slot = session_data['user_slot']
    num_teams = session_data['num_teams']

    # Poll the API for the latest draft state
    logger.info(f"Updating draft state for ID: {session_data['draft_id']}")
    manager.update_state()

    # Calculate current draft status (round, pick, on-the-clock team)
    next_pick_num = manager.current_pick_no + 1
    current_round = ((next_pick_num - 1) // num_teams) + 1
    pick_in_round = (next_pick_num - 1) % num_teams
    picks_away = 0
    
    # Determine who is on the clock and compute number of picks away from the user using snake draft logic
    if current_round % 2 == 1: # slots: 1 -> num_teams
        on_clock_idx = pick_in_round
        if (on_clock_idx + 1) <= user_slot:
            picks_away = user_slot - (on_clock_idx + 1)
        else:
            picks_away = (num_teams - (on_clock_idx + 1)) + (num_teams - user_slot)
    else: # slots: num_teams -> 1
        on_clock_idx = num_teams - 1 - pick_in_round
        if (on_clock_idx + 1) >= user_slot:
            picks_away = (on_clock_idx + 1) - user_slot
        else:
            picks_away = (num_teams - (on_clock_idx + 1)) + user_slot

    # Create draft status text
    on_clock_slot = on_clock_idx + 1 # Convert to 1-indexed for display
    is_user_turn = (on_clock_slot == user_slot)
    
    status_text = f"Round {current_round} • Pick {next_pick_num} (Overall)"

    if str(on_clock_slot) in session_data['users']:
        clock_text = f"On the Clock: {session_data['users'][str(on_clock_slot)]}"
    else:
        clock_text = f"On the Clock: Team {on_clock_slot}"

    if is_user_turn:
        away_text = "It is your turn to pick!"
    else:
        away_text = f"You are {picks_away} picks away from your next selection"
    
    # Get a recommendations from the agent for the current team on the clock
    recs = manager.get_recommendation(on_clock_idx, top_k=5)
    
    if not recs:
        return status_text, clock_text, away_text, "No recommendations available.", "", "", "", "", False

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

    return status_text, clock_text, away_text, rec_display, alt_display, table_avail, table_roster, "", False

@app.callback(
    [Output("session-data", "data", allow_duplicate=True),
     Output("session-control-area", "children", allow_duplicate=True),
     Output("interval-component", "disabled", allow_duplicate=True)],
    [Input("btn-resume", "n_clicks")],
    [State("session-data", "data")],
    prevent_initial_call=True
)
def resume_polling(n_clicks, session_data):
    """
    Resets the activity timestamp when the user clicks 'Resume'.
    """
    if not n_clicks or not session_data:
        return dash.no_update
    
    logger.info(f"Resuming polling for draft {session_data['draft_id']}")
    session_data['last_active'] = time.time()
    return session_data, "", False


if __name__ == '__main__':
    # Run the Dash application
    app.run(debug=False)
