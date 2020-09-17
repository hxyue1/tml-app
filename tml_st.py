import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

#loading in data and pickled model
player_features = pd.read_csv('top500_features.csv')
player_names = player_features['player_name']
feature_names = ['player_name', 'player_rank', 'player_log_rank', 'player_rank_points', 'player_log_rank_points', 'player_serve_win_ratio', 'player_return_win_ratio', 'player_bp_per_game', 'player_game_win_ratio', 'player_point_win_ratio', 'player_clutch_factor', 'player_win_rank_weight', 'player_win_elo_weight', 'player_point_win_ratio_rank_weighted', 'player_point_win_ratio_elo_weighted', 'player_old_elo', 'player_game_win_ratio_rank_weighted', 'player_game_win_ratio_elo_weighted']
player_features = player_features[feature_names]
model = pickle.load(open('tML_XGB.pickle', 'rb'))

#User interface for player selection
player1 = st.selectbox('Player 1', index=int(np.where(player_names=='Roger Federer')[0][0]), options=player_features['player_name'])
player2 = st.selectbox('Player 2', index=int(np.where(player_names=='Rafael Nadal')[0][0]), options=player_features['player_name'])

#Getting features for players
player1_feats = player_features[player_features['player_name'] == player1].drop('player_name', axis=1).reset_index(drop=True)
player2_feats = player_features[player_features['player_name'] == player2].drop('player_name', axis=1).reset_index(drop=True)
feats = player1_feats - player2_feats
feats = feats.add_suffix('_diff')

player1_w_pred = np.mean([model.predict_proba(feats)[:,1], model.predict_proba(-feats)[:,0]])

fig, ax = plt.subplots(figsize=(10,10))

fontsize=24

plt.pie([player1_w_pred, 1-player1_w_pred], colors=['Blue', 'Red'], startangle=90, radius=0.8,  wedgeprops=dict(width=0.2))
ax.text(-1.7,0.3,'Player 1 win: ', fontsize=fontsize)
ax.text(-1.7, 0, str(np.round(player1_w_pred*100, 2)) + '%', fontsize=fontsize)
ax.text(1,0.3,'Player 2 win: ', fontsize=fontsize) 
ax.text(1, 0, str(np.round((1-player1_w_pred)*100, 2)) + '%', fontsize=fontsize)

st.pyplot(fig)

