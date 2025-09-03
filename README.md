The target of this project is to analyze and predict ATP match winners with a machine learning model. 
This project uses a Random Forest to evaluate player data, rankings, and performance histories to predict the outcome of tennis matches with high accuracy.

After import the data, new column have been created representing code for each opponent. 
A target variable (wich is binary) has been created representing if the "player1" won the game with a "1" or if it didn't with a "0".

The train data is from 2005-01-01 to 2023-10-09 and the test data is after 2023-10-29.
The predictors that I use for this model are "playerCode", "opponentCode", "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2". 

With this model I got an accuracy of 0.6758. 

My recommendation for those who are interested by this model is to quantify some categorial variable (like courtType, groundType, TournamentCode (I have already quantify these variables ^^ and I have not incorporate them in the model)) and try to improve the accuracy.

If someone want the update database contact me: "nanemawendemi@gmail.com"
