import streamlit as st
import tennisML as tML
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

@st.cache
def load_data():
    data = pd.read_csv('hard_court_final.csv')
    data.loc[:,'tourney_start_date'] = pd.to_datetime(data['tourney_start_date'])
    player_final_features = pd.read_csv('top500_features.csv')
    return(data, player_final_features)

@st.cache
def model_train(df_final, features, early_stopping_rounds, learning_rate, num_leaves, early_stopping_metric, test_metric):
    
    train_indices, val_indices, test_indices = tML.f_chain_index_by_year(2000, 2019, max_train_years=20, train_val_test=True)
    test_scores = []
    val_scores = []
    
    for train_index, val_index, test_index in zip(train_indices, val_indices, test_indices):

        
        #Configuring train, val and test data
        print(df_final.tourney_start_date)
        X_train = df_final.loc[df_final.tourney_start_date.dt.year.isin(train_index), features]
        y_train = df_final.loc[df_final.tourney_start_date.dt.year.isin(train_index), 'player_1_win']
        X_val = df_final.loc[df_final.tourney_start_date.dt.year == val_index, features]
        y_val = df_final.loc[df_final.tourney_start_date.dt.year == val_index, 'player_1_win']
        X_test = df_final.loc[(df_final.tourney_start_date.dt.year == test_index), features]
        y_test = df_final.loc[(df_final.tourney_start_date.dt.year == test_index), 'player_1_win']
                    
        
        #Conforming data to lgb api
        train = lgb.Dataset(data=X_train, label=y_train)
        val = lgb.Dataset(data=X_val, label=y_val)
        test = lgb.Dataset(data=X_test, label=y_test)
        
        #Evaluation metric dependent on user input
        if early_stopping_metric == 'AUC':
            metric = 'auc'
        elif early_stopping_metric == 'Accuracy':
            metric = 'binary_error'
        elif early_stopping_metric == 'Logloss':
            metric = 'binary_logloss'
        param = {'objective':'binary', 'metric':metric, 'learning_rate':learning_rate, 'num_leaves':num_leaves}
        bst = lgb.train(param, train, early_stopping_rounds=early_stopping_rounds, valid_sets=val)
        
        #Getting best validation score
        val_score = bst.best_score['valid_0'][metric]
        if early_stopping_metric == 'Accuracy':
            val_score = 1 - val_score
        val_scores.append(val_score)
        
        #Predicting one step ahead matches
        test_preds = bst.predict(X_test, num_iteration=bst.best_iteration)
        if test_metric == 'AUC':        
            test_score = roc_auc_score(y_test, test_preds)
        elif test_metric == 'Accuracy':
            test_score = accuracy_score(y_test, np.round(test_preds, 0))
        elif test_metric == 'Logloss':
            test_score = log_loss(y_test, test_preds)
        test_scores.append(test_score)
    
    return(test_scores, val_scores, bst)


def get_prediction(player_1, player_2, player_final_features, chosen_features, bst):
    p1_features = player_final_features[player_final_features['player_name'] == player_1]
    p2_features = player_final_features[player_final_features['player_name'] == player_2]
    
    features_diffed = p1_features.reset_index(drop=True).drop('player_name', axis=1)- p2_features.reset_index(drop=True).drop('player_name', axis=1)
    features_diffed = features_diffed.add_suffix('_diff')
    
    preds = np.mean([bst.predict(features_diffed[chosen_features]),1-bst.predict(-features_diffed[chosen_features])])
    print(bst.predict(features_diffed[chosen_features]), features_diffed)
    return(preds)

st.subheader('The Dataset')
df_final, player_final_features = load_data()
ML_cols_subset= [
'player_1_win', 'player_rank_diff', 'player_log_rank_diff',
'player_rank_points_diff', 'player_log_rank_points_diff',
'player_serve_win_ratio_diff', 'player_return_win_ratio_diff',
'player_bp_per_game_diff',
'player_game_win_ratio_diff', 'player_point_win_ratio_diff',
'player_clutch_factor_diff', 
'player_win_rank_weight_diff', 'player_win_elo_weight_diff',
'player_point_win_ratio_rank_weighted_diff',
'player_point_win_ratio_elo_weighted_diff', 'player_old_elo_diff', 'player_old_elo_surface_specific_diff',
    'player_game_win_ratio_rank_weighted_diff', 'player_game_win_ratio_elo_weighted_diff']
st.dataframe(df_final)
    
#Machine Learning
features_list = ML_cols_subset.copy()
features_list.remove('player_1_win')
chosen_features = st.multiselect('Choose features', features_list, default=['player_serve_win_ratio_diff', 'player_return_win_ratio_diff', 'player_old_elo_diff'] )
early_stopping_metric = st.selectbox('Early Stopping Metric', [ 'Accuracy', 'AUC',  'Logloss'], index=2)
test_metric = st.selectbox('Final Evaluation Metric', [ 'Accuracy', 'AUC',  'Logloss'], index=2)

#GBM Parameters
st.subheader('Gradient Boosting Parameters')
early_stopping_rounds = st.slider('Early stopping rounds', 5,50)
learning_rate = st.slider('Learning rate', 0., 1., value=0.1, step=0.01)
num_leaves = st.slider('Number of leaves', 2, 100)

#ML Output
test_scores, val_scores, bst = model_train(df_final, chosen_features, early_stopping_rounds, learning_rate, num_leaves, early_stopping_metric, test_metric)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(np.arange(2001,2019), val_scores, label = 'Validation Scores')
ax.plot(np.arange(2002,2020), test_scores, label= 'Test Scores')
ax.legend()
st.pyplot(fig)
st.text('Average validation ' +  early_stopping_metric + ' is ' + str(np.mean(val_scores)))
st.text('Average test ' + test_metric + ' is ' + str(np.mean(test_scores)))

#Making predictions
st.subheader('Match Prediction')
players = player_final_features['player_name']

player_1 = st.selectbox('Choose player 1', players, index=int(players[players=='Roger Federer'].index[0]))
player_2 = st.selectbox('Choose player 2', players, index=int(players[players=='Rafael Nadal'].index[0]))
preds = get_prediction(player_1, player_2, player_final_features, chosen_features, bst)

#Plotting prediction
fig, ax = plt.subplots(figsize=(10,10))
fontsize=24
plt.pie([preds, 1-preds], colors=['Blue', 'Red'], startangle=90, radius=0.8,  wedgeprops=dict(width=0.2))
ax.text(-1.7,0.3,'Player 1 win: ', fontsize=fontsize)
ax.text(-1.7, 0, str(np.round(preds*100, 2)) + '%', fontsize=fontsize)
ax.text(1,0.3,'Player 2 win: ', fontsize=fontsize) 
ax.text(1, 0, str(np.round((1-preds)*100, 2)) + '%', fontsize=fontsize)
st.pyplot(fig)

