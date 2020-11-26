import numpy as np
import pandas as pd
import datetime as datetime
import itertools
import re
import streamlit as st

"""Still need to tidy up docstrings and name of file"""

def data_cleaning(df, surfaces, tourneys_to_include, start_year=1999):
    """Performs initial data cleaning on match level data to prepare for data transformation and feature engineering.
    
    Function works with ATP match data obtained from the R package Deuce: https://github.com/skoval/deuce. Probably works with WTA data but you might need to rename the columns.
    
    Args:
        df (pandas.DataFrame): ATP match data from the R package Deuce imported as a pandas DataFrame.
        surfaces (list): List of strings containing one or more of 'Hard', 'Clay', 'Grass', 'Indoor Hard' or 'Carpet'.
        tourneys_to_include (list): Level of matches to be included in aggregation for a tournament can include 'Grand Slams', '250 or 500', 'Davis Cup', 'Masters', 'Challenger', 'Tour Finals' or 'Futures'.
        start_year (int): Year to start aggregating data. This should be at least one year before the first tournmanet you want included in training your model.
    
    Returns: 
        pandas.DataFrame. Cleaned DataFrame ready for feature engineering and aggregation.
        
    """
    
# =============================================================================
#     
#     #Performing elo calculation, see: https://www.betfair.com.au/hub/tennis-elo-modelling
#     all_names = list(set(df.loser_name.unique()) | set(df.winner_name.unique()))
#     
#     elo_dict = {player:[1500] for player in all_names}
# 
#     i = 0
# 
#     previous_winner_elos = []
#     previous_loser_elos = []
# 
#     for index, row in df.iterrows():
# 
#         winner = row.winner_name
#         loser = row.loser_name
# 
#         winner_elos = elo_dict[winner]
#         loser_elos = elo_dict[loser]
# 
#         winner_matches = len(winner_elos)
#         loser_matches = len(loser_elos)
# 
#         winner_old_elo = winner_elos[-1]
#         loser_old_elo = loser_elos[-1]
# 
#         previous_winner_elos.append(winner_old_elo)
#         previous_loser_elos.append(loser_old_elo)
# 
#         pr_p1_win_elo = 1/(1+10**((loser_old_elo - winner_old_elo)/400))
#         pr_p2_win_elo = 1/(1+10**((winner_old_elo - loser_old_elo)/400))
# 
#         winner_K = 250/((winner_matches+5)**0.4)
#         loser_K = 250/((loser_matches+5)**0.4)
# 
#         winner_new_elo = winner_old_elo + winner_K*(1-pr_p1_win_elo)
#         loser_new_elo = loser_old_elo + loser_K*(0-pr_p2_win_elo)
# 
#         winner_elos.append(winner_new_elo)
#         loser_elos.append(loser_new_elo)
# 
#         elo_dict[winner] = winner_elos
#         elo_dict[loser] = loser_elos
#     
#     df.loc[:,'winner_old_elo'] = previous_winner_elos
#     df.loc[:,'loser_old_elo'] = previous_loser_elos
# =============================================================================
    
    df = calculate_elo(df)
    #Renaming columns
    new_cols = [
        'tourney_id', 'tourney_name', 'surface', 'draw_size',
        'tourney_level', 'match_num', 'winner_id', 'winner_seed',
        'winner_entry', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc',
        'winner_age', 'winner_rank', 'winner_rank_points', 'loser_id',
        'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht',
        'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points', 'score',
        'best_of', 'round', 'minutes', 'winner_ace', 'winner_df', 'winner_svpt', 'winner_1stIn',
        'winner_1stWon', 'winner_2ndWon', 'winner_SvGms', 'winner_bpSaved', 'winner_bpFaced', 'loser_ace',
        'loser_df', 'loser_svpt', 'loser_1stIn', 'loser_1stWon', 'loser_2ndWon', 'loser_SvGms',
        'loser_bpSaved', 'loser_bpFaced', 'W1', 'W2', 'W3', 'W4', 'W5', 'L1', 'L2',
        'L3', 'L4', 'L5', 'retirement', 'WTB1', 'LTB1', 'WTB2', 'LTB2', 'WTB3',
        'LTB3', 'WTB4', 'LTB4', 'WTB5', 'LTB5', 'tourney_start_date', 'year',
        'match_id', 'winner_old_elo', 'loser_old_elo', 'winner_old_elo_surface_specific', 'loser_old_elo_surface_specific'
    ]
    
    df.columns = new_cols
    
    #You can change what matches to include. I've chosen to exclude Futures matches and the Challenger tour
    # tourney_levels = 'Grand Slams', '250 or 500', 'Davis Cup', 'Masters', 'Challenger', 'Tour Finals', 'Futures'
    df = df.loc[(df['tourney_level'].isin(tourneys_to_include)) &\
            (df['year'] >= start_year-1) & (df['surface'].isin(surfaces))&\
            (~df['round'].isin(['Q1', 'Q2', 'Q3', 'Q4']))
           ].copy()

    #Converting dates to datetime
    df.loc[:,'tourney_start_date'] = pd.to_datetime(df['tourney_start_date'])
    df.loc[:,'year'] = pd.to_datetime(df['year'], format='%Y')
    
    #Factorizing round
    df.loc[:,'round'] = pd.factorize(df['round'])[0]

    #Parsing game scores
    winner_game_score_cols = ['W1', 'W2', 'W3', 'W4', 'W5']
    loser_game_score_cols = ['L1', 'L2', 'L3', 'L4', 'L5']
    game_score_cols = winner_game_score_cols + loser_game_score_cols
    df.loc[:, game_score_cols] = df.loc[:, game_score_cols].fillna(0)
    df.loc[:,'winner_total_games'] = df[winner_game_score_cols].sum(axis=1)
    df.loc[:,'loser_total_games'] = df[loser_game_score_cols].sum(axis=1)
    df.loc[:,'total_games'] = df['winner_total_games'] + df['loser_total_games']
    df.loc[:,'loser_RtGms'] = df['winner_SvGms']
    df.loc[:,'winner_RtGms'] = df['loser_SvGms']
    
    #Imputing bp data
    df.loc[:,'loser_bp'] = df['winner_bpFaced']
    df.loc[:,'winner_bp'] = df['loser_bpFaced']
    df.loc[:,'loser_bpWon'] = df['winner_bpFaced'] - df['winner_bpSaved'] 
    df.loc[:,'winner_bpWon'] = df['loser_bpFaced'] - df['loser_bpSaved'] 
    
    #Imputing returns data so we can construct features
    df.loc[:,'winner_2ndIn'] = df['winner_svpt'] - df['winner_1stIn'] - df['winner_df']
    df.loc[:,'loser_2ndIn'] = df['loser_svpt'] - df['loser_1stIn'] - df['loser_df']
    df.loc[:,'loser_rtpt'] = df['winner_svpt']
    df.loc[:,'winner_rtpt'] = df['loser_svpt']
    df.loc[:,'winner_rtptWon'] = df['loser_svpt'] -  df['loser_1stWon'] - df['loser_2ndWon']
    df.loc[:,'loser_rtptWon'] = df['winner_svpt'] -  df['winner_1stWon'] - df['winner_2ndWon']
    df.loc[:,'winner_svptWon'] = df['winner_1stWon'] + df['winner_2ndWon']
    df.loc[:,'loser_svptWon'] = df['loser_1stWon'] + df['loser_2ndWon']
    df.loc[:,'winner_total_points'] = df['winner_svptWon'] + df['winner_rtptWon']
    df.loc[:,'loser_total_points'] = df['loser_svptWon'] + df['loser_rtptWon']
    df.loc[:,'total_points'] = df['winner_total_points'] + df['loser_total_points']

    #Dropping columns
    cols_to_drop =[
        'draw_size',
        'winner_seed',
        'winner_entry',
        'loser_seed',
        'loser_entry',
        'score',
        'W1', 'W2', 'W3', 'W4', 'W5', 'L1', 'L2',
        'L3', 'L4', 'L5', 'WTB1', 'LTB1', 'WTB2', 'LTB2', 'WTB3',
        'LTB3', 'WTB4', 'LTB4', 'WTB5', 'LTB5'
        ]
    
    df.drop(cols_to_drop, axis=1, inplace=True)

    #Filling nans values
    df.loc[:,'loser_rank'].fillna(500, inplace=True)
    df.loc[:,'winner_rank'].fillna(500, inplace=True)
    df.loc[:,'winner_hand'].fillna('R', inplace=True)
    df.loc[:,'loser_hand'].fillna('R', inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    return(df)

def calculate_elo(df):
    """Calculates elo for players with breakdowns for specific surfaces. """
    
    #Performing elo calculation, see: https://www.betfair.com.au/hub/tennis-elo-modelling
    all_names = list(set(df.loser_name.unique()) | set(df.winner_name.unique()))
    
    elo_dict_all_surfaces = {player:[1500] for player in all_names}
    elo_dict_grass = {player:[1500] for player in all_names}
    elo_dict_carpet = {player:[1500] for player in all_names}
    elo_dict_clay = {player:[1500] for player in all_names}
    elo_dict_hard = {player:[1500] for player in all_names}

    #Performing all court elo calculation
    i = 0

    previous_winner_elos_all_surfaces = []
    previous_loser_elos_all_surfaces = []
    previous_winner_elos_surface_specific = []
    previous_loser_elos_surface_specific = []

    for index, row in df.iterrows():

        winner = row.winner_name
        loser = row.loser_name
        
        if row['surface'] == 'Grass':
            elo_dict_surface_specific = elo_dict_grass
        elif row['surface'] == 'Carpet':
            elo_dict_surface_specific = elo_dict_carpet
        elif row['surface'] == 'Clay':
            elo_dict_surface_specific = elo_dict_clay
        elif row['surface'] == 'Hard':
            elo_dict_surface_specific = elo_dict_hard
        
        #################################
        #Elo calculation for all surfaces
        #################################
            
        #Getting elo history of the current winner and loser
        winner_elos_all_surfaces = elo_dict_all_surfaces[winner]
        loser_elos_all_surfaces = elo_dict_all_surfaces[loser]

        #Getting number of matches for winner and loser
        winner_matches_all_surfaces = len(winner_elos_all_surfaces)
        loser_matches_all_surfaces = len(loser_elos_all_surfaces)

        #Getting the player's last elo rating
        winner_old_elo_all_surfaces = winner_elos_all_surfaces[-1]
        loser_old_elo_all_surfaces = loser_elos_all_surfaces[-1]

        #Updating the previous elo for the winner and loser of the current match (will be used as variables in prediction)
        previous_winner_elos_all_surfaces.append(winner_old_elo_all_surfaces)
        previous_loser_elos_all_surfaces.append(loser_old_elo_all_surfaces)
        
        #Predicting win probabilities based on elos
        pr_p1_win_elo_all_surfaces = 1/(1+10**((loser_old_elo_all_surfaces - winner_old_elo_all_surfaces)/400))
        pr_p2_win_elo_all_surfaces = 1/(1+10**((winner_old_elo_all_surfaces - loser_old_elo_all_surfaces)/400))
        
        #K factor used in elo calculation (controls speed of update, see the betfair link)
        winner_K_all_surfaces = 250/((winner_matches_all_surfaces+5)**0.4)
        loser_K_all_surfaces = 250/((loser_matches_all_surfaces+5)**0.4)
        
        #Caluclating new elo
        winner_new_elo_all_surfaces = winner_old_elo_all_surfaces + winner_K_all_surfaces*(1-pr_p1_win_elo_all_surfaces)
        loser_new_elo_all_surfaces = loser_old_elo_all_surfaces + loser_K_all_surfaces*(0-pr_p2_win_elo_all_surfaces)

        #Updating elos
        winner_elos_all_surfaces.append(winner_new_elo_all_surfaces)
        loser_elos_all_surfaces.append(loser_new_elo_all_surfaces)
        
        #Storing updated elos in elo dict
        elo_dict_all_surfaces[winner] = winner_elos_all_surfaces
        elo_dict_all_surfaces[loser] = loser_elos_all_surfaces
        
        #################################
        #Elo calculation for specific surfaces
        #################################
            
        #Getting elo history of the current winner and loser
        winner_elos_surface_specific = elo_dict_surface_specific[winner]
        loser_elos_surface_specific = elo_dict_surface_specific[loser]

        #Getting number of matches for winner and loser
        winner_matches_surface_specific = len(winner_elos_surface_specific)
        loser_matches_surface_specific = len(loser_elos_surface_specific)

        #Getting the player's last elo rating
        winner_old_elo_surface_specific = winner_elos_surface_specific[-1]
        loser_old_elo_surface_specific = loser_elos_surface_specific[-1]

        #Updating the previous elo for the winner and loser of the current match (will be used as variables in prediction)
        previous_winner_elos_surface_specific.append(winner_old_elo_surface_specific)
        previous_loser_elos_surface_specific.append(loser_old_elo_surface_specific)
        
        #Predicting win probabilities based on elos
        pr_p1_win_elo_surface_specific = 1/(1+10**((loser_old_elo_surface_specific - winner_old_elo_surface_specific)/400))
        pr_p2_win_elo_surface_specific = 1/(1+10**((winner_old_elo_surface_specific - loser_old_elo_surface_specific)/400))
        
        #K factor used in elo calculation (controls speed of update, see the betfair link)
        winner_K_surface_specific = 250/((winner_matches_surface_specific+5)**0.4)
        loser_K_surface_specific = 250/((loser_matches_surface_specific+5)**0.4)
        
        #Caluclating new elo
        winner_new_elo_surface_specific = winner_old_elo_surface_specific + winner_K_surface_specific*(1-pr_p1_win_elo_surface_specific)
        loser_new_elo_surface_specific = loser_old_elo_surface_specific + loser_K_surface_specific*(0-pr_p2_win_elo_surface_specific)

        #Updating elos
        winner_elos_surface_specific.append(winner_new_elo_surface_specific)
        loser_elos_surface_specific.append(loser_new_elo_surface_specific)
        
        #Storing updated elos in elo dict
        elo_dict_surface_specific[winner] = winner_elos_surface_specific
        elo_dict_surface_specific[loser] = loser_elos_surface_specific
    
    df.loc[:,'winner_old_elo'] = previous_winner_elos_all_surfaces
    df.loc[:,'loser_old_elo'] = previous_loser_elos_all_surfaces 
    df.loc[:,'winner_old_elo_surface_specific'] = previous_winner_elos_surface_specific
    df.loc[:,'loser_old_elo_surface_specific'] = previous_loser_elos_surface_specific
    
    return(df)

def convert_long(df):    
    """Converts cleaned match data into long form by splitting each match into two observations, one for the winner and one for the loser.
    
    E.g. Match: winner is Roger Federer, loser is Novak Djokovic. Creates one observation for Roger Federer and his data and one observation for Novak Djokovic and his data.
    
    Args: 
        df (pandas.DataFrame): Cleaned DataFrame after applying data_cleaning().
    
    Returns:
        df (pandas.DataFrame): DataFrame converted into long form, split by winner and loser.
        
    """
    
    #Separating features into winner and loser so we can create rolling averages for each major tournament
    winner_cols = [col for col in df.columns if col.startswith('w')]
    loser_cols = [col for col in df.columns if col.startswith('l')]
    common_cols = [
        'tourney_id', 'tourney_name', 'surface', 'tourney_level',
       'match_num','best_of', 'round',
       'minutes','retirement', 'tourney_start_date', 'year', 'match_id',
        'total_points', 'total_games'
    ]
    
    #Will also add opponent's rank
    df_winner = df[winner_cols + common_cols + ['loser_rank'] + ['loser_old_elo']]
    df_loser = df[loser_cols + common_cols + ['winner_rank'] + ['winner_old_elo']]
    
    df_winner.loc[:,'won'] = 1
    df_loser.loc[:,'won'] = 0
    
    #Renaming columns
    df_winner.columns = [col.replace('winner','player').replace('loser', 'opponent') for col in df_winner.columns]
    df_loser.columns = df_winner.columns
    
    df_long = df_winner.append(df_loser, ignore_index=True)
    
    return(df_long)

@st.cache
def engineer_player_stats(df_long):
    """Conducts feature engineering. You can add your own features if you want, but make sure that they are prefixed by 'player'.
    
    Args:
        df (pandas.DataFrame): Long form match level data after applying convert_long().
        
    Returns: 
        df (pandas.DataFrame): Long form match level data with additional features.
        
    """
    
    #Creating new features we can play around with, note that not all features may be used
    df_long.loc[:,'player_serve_win_ratio'] = (df_long['player_1stWon'] + df_long['player_2ndWon'])/\
    (df_long['player_1stIn'] + df_long['player_2ndIn'] + df_long['player_df'] )
    
    df_long.loc[:,'player_return_win_ratio'] = df_long['player_rtptWon']/df_long['player_rtpt']
    
    df_long.loc[:,'player_bp_per_game'] = df_long['player_bp']/df_long['player_RtGms']
    
    df_long.loc[:,'player_bp_conversion_ratio'] = df_long['player_bpWon']/df_long['player_bp']
    
    #Setting nans to zero for breakpoint conversion ratio
    df_long.loc[:,'player_bp_conversion_ratio'].fillna(0, inplace=True)
    
    df_long.loc[:,'player_game_win_ratio'] = df_long['player_total_games']/df_long['total_games']
    df_long.loc[:, 'player_game_win_ratio'].fillna(0.5, inplace=True)
    
    df_long.loc[:,'player_point_win_ratio'] = df_long['player_total_points']/df_long['total_points']
    
    #df['player_set_Win_Ratio'] = df['Player_Sets_Won']/df['Total_Sets']
    
    df_long.loc[:,'player_clutch_factor'] = df_long['player_game_win_ratio'] - df_long['player_point_win_ratio']
    
    df_long.loc[:,'player_log_rank'] = np.log(df_long['player_rank'])
    
    df_long.loc[:,'player_log_rank_points'] = np.log(df_long['player_rank_points'])
    
    df_long.loc[:,'player_win_rank_weight'] = df_long['won'] * np.exp(-df_long['opponent_rank']/100)
    
    df_long.loc[:,'player_win_elo_weight'] = df_long['won'] * np.exp(-df_long['opponent_old_elo']/100)

    #Let's try weighting some of the features by the opponent's rank
    
    #df['Player_Set_Win_Ratio_Weighted'] = df['Player_Set_Win_Ratio']*np.exp((df['Player_Rank']-df['Opponent_Rank'])/500)
    df_long.loc[:,'player_game_win_ratio_rank_weighted'] = df_long['player_game_win_ratio']*np.exp((df_long['player_rank']-df_long['opponent_rank'])/500)
    df_long.loc[:,'player_point_win_ratio_rank_weighted'] = df_long['player_point_win_ratio']*np.exp((df_long['player_rank']-df_long['opponent_rank'])/500)
    
    df_long.loc[:,'player_game_win_ratio_elo_weighted'] = df_long['player_game_win_ratio']*(df_long['player_old_elo']-df_long['opponent_old_elo'])
    df_long.loc[:,'player_point_win_ratio_elo_weighted'] = df_long['player_point_win_ratio']*(df_long['player_old_elo']-df_long['opponent_old_elo'])
    
    print('Running')
    
    return(df_long)

def get_tournament_features(df_long, player_names, tourney_start_date, rolling_cols, last_cols, window, days):
    player_data = df_long.loc[(df_long['tourney_start_date'] < tourney_start_date) &\
                         (df_long['tourney_start_date'] > tourney_start_date - datetime.timedelta(days=days)) &\
                         (df_long['player_name'].isin(player_names))]
        
    player_data = player_data.sort_values(['player_name', 'tourney_start_date'], ascending=True)
    
    #Only taking the most recent value for the feature, if specified in last_cols
    if last_cols != None:
        last_features = player_data.groupby('player_name')[last_cols].last().reset_index()

    #Taking a rolling average of the x (window_length) most recent matches before specified tournament date,
    #for features specified in rolling_cols
    #ma_features = player_data.groupby('player_name')[rolling_cols].rolling(window,1).mean().reset_index() works in pandas 0.25.1, but no longer in 1.1.0
    ma_features = player_data[['player_name']+rolling_cols].groupby('player_name')[rolling_cols].rolling(window,1).mean().reset_index()

    #Only taking the most recent rolling average
    ma_features = ma_features.groupby('player_name').tail(1)

    if last_cols != None:
        tournament_features = ma_features.merge(last_features, on = 'player_name', how='left')
    else:
        tournament_features = ma_features
    
    return(tournament_features)

def aggregate_features(df_long, tourney_names, start_year, rolling_cols, last_cols, window, days=365):
    """For all players in a given tournament, calculates a rolling average of their match statistics before the tournament over a specified window.
    
    E.g. For Kei Nishikori at the 2014 US Open, we look at his x past matches (where x is specified by window) before the tournament and take an average of his features. These become the new features which are actually used for predicting his performance in the tournament. This is then repeated for all participants in the 2014 US Open and all instances of the US Open in the input DataFrame.

    Args:
        df (pandas.DataFrame): Long form match level data (ideally after applying get_new_features()).
        tourney_names (list): List containing the names of tournaments you want features for.
        start_year (int): The first year of the specified tournament(s) you want features for.
        rolling_cols (list): List of variables which you want rolling aggregations for.
        last_cols (list): List of variables which you only want the most recent value for.
        window (int): Number of matches in the past to aggregate over.
        days (int): Maximum number of days in the past for a match to be included in the aggregation.
        
    Returns:
        df (pandas.DataFrame): Aggregated features for each player in all of the tournaments identified by (player, tournament, date key.
        
    """
    tourneys = df_long.loc[(df_long['tourney_name'].isin(tourney_names)) &\
                      (df_long.year.dt.year >= start_year)]\
                .groupby(['tourney_name', 'tourney_start_date'])\
                .size().reset_index()[['tourney_name', 'tourney_start_date']]

    for index, row in tourneys.iterrows():
        print(index, row['tourney_name'], row['tourney_start_date'])
        
        player_names = df_long[(df_long['tourney_start_date'] == row['tourney_start_date'])&\
                                 (df_long['tourney_name'] == row['tourney_name'])]['player_name'].unique()        
        
        #Calling get_tournament_features
        tournament_features = get_tournament_features(df_long, player_names, row['tourney_start_date'], rolling_cols, last_cols, window, days)

        #Adding a column telling us what tournament the rolling average is for
        if index == 0:
            df_result = tournament_features
            df_result['tournament_date_index'] = row['tourney_start_date']

        else:
            tournament_features['tournament_date_index'] = row['tourney_start_date']
            df_result = df_result.append(tournament_features)
        
    
    df_result.drop('level_1', axis=1, inplace=True)
    
    return(df_result, rolling_cols + last_cols)

def wrangle_target(df, start_year, tournaments):
    """Wrangling target and rearranging match information to prepare for merging with tournament features.
    
    Args: 
        df (pandas.DataFrame): Cleaned match level data.
        start_year (int): This should be the same as the start year specified in data_cleaning.
    
    Returns: wrangled dataframe ready for merging with tournament features.
        
    """
    
    
    df = df.loc[df.year.dt.year >=start_year]
    df_matches = df.copy()
    
    #Subsetting match data to relevant tournaments to match on
    df_matches = df_matches.loc[df_matches['tourney_name'].isin(tournaments)]

    #Removing unnecessary columns from match data
    cols_to_keep = ['winner_name','loser_name','tourney_name','tourney_start_date', 'tourney_level', 'round']

    #Duplicating matches and swapping the order of players
    df_matches = df_matches[cols_to_keep]
    
    df1 = df_matches.copy()
    df1.columns = ['player_1','player_2','tourney_name','tourney_start_date', 'tourney_level', 'round']
    df1['player_1_win'] = 1

    df2 = df_matches.copy()
    df2.columns = ['player_2','player_1','tourney_name','tourney_start_date', 'tourney_level', 'round']
    df2['player_1_win'] = 0

    df_matches = pd.concat([df1, df2], sort=False)
    df_matches.reset_index(drop=True, inplace=True)
    
    return(df_matches)

def merge_data(df, df_ma, start_year, tournaments, difference=True):
    """Merges rolling features with match level data.
    
    This should be run after generating the rolling features. Rolling features will be matched on player name and tournament date. Wrangling the target allows us to go from winner vs loser to player_1, player_2, player_1_wins. Note that two observations for each match will be generated, the second one swapping the order of player_1 and player_2 to induce variability in the target.
    
    Args: 
        df (pandas.DataFrame): Wrangled match level data
        df_ma (pandas.DataFrame): Pandas DataFrame obtained from running get_ma_features.
        start_year (int): This should be the same as the start year specified in data_cleaning.
        tournaments (list): List of tournament names you want to match features with. This should be the same as the ones specified in get_ma_features.
        difference (bool): Indicates whether or not to difference features. Default is True.
    
    Returns: Pandas DataFrame of match level data merged wih rolling features and target wrangled.
    
    """
    
    #Joining rolling features for p1 with match data
    df = df.merge(df_ma, how='left',
                  left_on = ['player_1', 'tourney_start_date'],
                  right_on = ['player_name', 'tournament_date_index'],
                  validate = 'm:1')

    #Joining rolling features for p2 and adding suffix for features belonging to each player
    df = df.merge(df_ma, how='left',
                  left_on = ['player_2', 'tourney_start_date'],
                  right_on = ['player_name', 'tournament_date_index'],
                  validate = 'm:1',
                  suffixes=('_p1', '_p2'))
    
    feature_names = df_ma.columns.drop(['player_name', 'tournament_date_index'])        
    p1_cols = [i + '_p1' for i in feature_names] 
    p2_cols = [i + '_p2' for i in feature_names] 
    
    if 'player_rank_p1' in df.columns:
        df['player_rank_p1'].fillna(500, inplace=True)
        df['player_rank_p2'].fillna(500, inplace=True)
    
        
    if 'player_log_rank_p1' in df.columns:
        df['player_log_rank_p1'].fillna(np.log(500), inplace=True)
        df['player_log_rank_p2'].fillna(np.log(500), inplace=True)
        
    df[p1_cols].fillna(-1, inplace=True)
    df[p2_cols].fillna(-1, inplace=True)
    
    return(df)

def get_player_difference(merged_df, diff_cols = None):
    """Differences features for player 1 and 2.
    
    Will automatically try to find names of features to be differenced but you can supply your own in diff_cols. Supplied names need to be the same as in the long form dataframe e.g. "player_serve_win_ratio" not "player_serve_win_ratio_p1".
    
    Args:
        df_merged (pandas.DataFrame): Merged dataframe with rolling features and match level data obtained after running merge_data
        diff_cols (list, optional): List of specified feature names to be differenced. Default is None.
        
    Returns:
        pandas.DataFrame: Merged data with features differenced.
    """
    
    df = merged_df.copy()
    
    #If diff_cols has not been supplied then find p1 and p2 col names
    if diff_cols == None: 
        p1_cols = []
        p2_cols = []
        
        for column in df.columns:
            if re.search('^((?!name).)*(_p1)', column) != None:
                p1_cols.append(re.search('^((?!name).)*(_p1)', column)[0])
                
            if re.search('^((?!name).)*(_p2)', column) != None:
                p2_cols.append(re.search('^((?!name).)*(_p2)', column)[0])
    
    #Else use supplied col names                
    else: 
        p1_cols = [i + '_p1' for i in diff_cols] # column names for player 1 stats
        p2_cols = [i + '_p2' for i in diff_cols] # column names for player 2 stats

    # Filling missing values
    df['player_rank_p1'] = df['player_rank_p1'].fillna(500)
    df['player_log_rank_p1'] = df['player_log_rank_p1'].fillna(np.log(500))
    df[p1_cols] = df[p1_cols].fillna(-1)
    
    df['player_rank_p2'] = df['player_rank_p2'].fillna(500)
    df['player_log_rank_p2'] = df['player_log_rank_p2'].fillna(np.log(500))
    df[p2_cols] = df[p2_cols].fillna(-1)
    
    #Naming new columns after differencing 
    if diff_cols == None:
        if p1_cols.copy().sort() != p2_cols.copy().sort():
            raise Exception('Names of p1 and p2 cols are not the same!')
        else:
            new_column_names = [re.search('(.*)(_p1)', column)[1] + '_diff' for column in p1_cols]
            
    else:
        new_column_names = [i + '_diff' for i in diff_cols]

    
    # Take the difference
    df_p1 = df[p1_cols]
    df_p2 = df[p2_cols]
    
    df_p1.columns=new_column_names
    df_p2.columns=new_column_names
    
    df_diff = df_p1 - df_p2
    df_diff.columns = new_column_names
    
    #Dropping spare columns
    df.drop(p1_cols + p2_cols, axis=1, inplace=True)
    
    # Concat the df_diff and raw_df
    df = pd.concat([df, df_diff], axis=1)
    
    return(df)

def f_chain_index_by_year(start_year, end_year, max_train_years, train_val_test = False):
    """Creates indices for forward chaining incremented by years to partition data for training, validation and testing using forward chaining.
    
    E.g. If start year is 2000 and end_year is 2003 the training index will look like [[2000], [2000, 2001], [2000, 2001, 2002]]. The test index will be [2001, 2002, 2003].
    
    start_year (int): The first year you want to include in the training data.
    end_year (int): The last year you want to include in the testing data.
    max_train_years (int): The maximum amount of years you want to include in each training fold.
    train_val_test (bool): Indicates whether or do a three fold train-val-test split as opposed to the default train-test split.
    
    Returns:
        tuple: A tuple of lists with the first item being a list of lists for the training indices, and the sequential items being the a list of indices for validation and testing.
    
    """
    
    years = range(start_year, end_year+1)
    
    #Creating index for training years
    train_years = []
    for index, year in enumerate(years):
        if index == 0:
            train_temp = [year]
            train_years.append(train_temp)
        else:
            train_temp = train_years[index-1].copy()
            train_temp.reverse()
            train_temp = train_temp[:max_train_years-1]
            train_temp.reverse()
            train_years.append(train_temp + [year])
            
    
    if train_val_test == False:
        
        #Dropping last year
        train_years.pop(-1)

        test_years = list(years)
        #Dropping first year
        test_years.pop(0)
        
        return((train_years, test_years))
    else:
        
        #Dropping last two years
        train_years.pop(-1)
        train_years.pop(-1)
        
        #Dropping first and last year
        val_years = list(years)
        val_years.pop(0)
        val_years.pop(-1)

        #Dropping first two years
        test_years = list(years)
        test_years.pop(0)
        test_years.pop(0)
        
        return((train_years, val_years, test_years))



def get_final_features(players, tourney_start_date, df_long, last_cols, rolling_cols, window, days):
    """Generates predictions for all possible player matchups given names of players and date of aggregation.
    
    Essentially runs the entire workflow again but for specified players and a specific date for aggregation.
    
    Args:
        players (list): List of all players participating in the tournament, make sure they match up with names found in the match level data!
        tourney_start_date (datetime): Datetime object corresponding to the start date of the tournament which you want predictions for.
        df_long (pandas.DataFrame): Long form match level data (ideally after applying get_new_features()).
        rolling_cols (list): List of variables which you want rolling aggregations for.
        last_cols (list): List of variables which you only want the most recent value for.
        window (int): Number of matches in the past to aggregate over.
        days (int): Maximum number of days in the past for a match to be included in the aggregation.
        
    Returns:
        pandas.DataFrame: Returns final feature dataframe which can be used to generate predictions for a trained model.
        
    """
    #Generating player permutations
    player_permutations = list(itertools.permutations(players, 2))
    preds_df = pd.DataFrame(player_permutations, columns=['player_1','player_2'])
    preds_df.loc[:,'player_1_win_probability'] = 0.5
    
    tournament_features = get_tournament_features(df_long, players, tourney_start_date, rolling_cols, last_cols, window, days)
    
    #Merging with prediction dataframe
    preds_df = preds_df.merge(tournament_features, how='left',
                                     left_on = 'player_1',
                                     right_on = 'player_name',
                                     validate = 'm:1')
    
    preds_df = preds_df.merge(tournament_features, how='left',
                                     left_on = 'player_2',
                                     right_on = 'player_name',
                                     validate = 'm:1',
                                     suffixes = ('_p1','_p2'))

    #Filling missing values !!!What if not using rank or log_rank?
    preds_df.loc[:,'player_rank_p1'].fillna(500, inplace=True)
    preds_df.loc[:,'player_rank_p2'].fillna(500, inplace=True)
    preds_df.loc[:,'player_log_rank_p1'].fillna(np.log(500), inplace=True)
    preds_df.loc[:,'player_log_rank_p2'].fillna(np.log(500), inplace=True)
    preds_df.loc[:,['player_log_rank_p1','player_log_rank_p2']].fillna(np.log(500), inplace=True)
    preds_df.fillna(-1, inplace=True)
    
    #Differencing player features and generating predictions
    preds_df = get_player_difference(preds_df, diff_cols=last_cols+rolling_cols)  
    
    return(preds_df)
    