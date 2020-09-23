import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

#loading in data and pickled model
player_features = pd.read_csv('top100_features.csv')
player_names = player_features['player_name']
preds = np.genfromtxt('preds.csv', delimiter=',')

#User interface for player selection
player1 = st.selectbox('Player 1', index=int(np.where(player_names=='Roger Federer')[0][0]), options=player_features['player_name'])
player2 = st.selectbox('Player 2', index=int(np.where(player_names=='Rafael Nadal')[0][0]), options=player_features['player_name'])
player1_w_pred = preds[player_names[player_names == player1].index, player_names[player_names == player2].index ][0]

print(player1_w_pred)

fig, ax = plt.subplots(figsize=(10,10))

fontsize=24

plt.pie([player1_w_pred, 1-player1_w_pred], colors=['Blue', 'Red'], startangle=90, radius=0.8,  wedgeprops=dict(width=0.2))
ax.text(-1.7,0.3,'Player 1 win: ', fontsize=fontsize)
ax.text(-1.7, 0, str(np.round(player1_w_pred*100, 2)) + '%', fontsize=fontsize)
ax.text(1,0.3,'Player 2 win: ', fontsize=fontsize) 
ax.text(1, 0, str(np.round((1-player1_w_pred)*100, 2)) + '%', fontsize=fontsize)

st.pyplot(fig)

