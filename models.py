# ATP-Tennis-Predictor
#Analyzing and predicting ATP match winners with a machine learning model. This project uses a Random Forest to evaluate player data, rankings, and performance histories to predict the outcome of tennis matches with high accuracy.

import numpy as np
import pandas as pd

matches = pd.read_csv("atp_tennis.csv")
matches.head()

matches.shape

matches["date"] = pd.to_datetime(matches["Date"])
matches.dtypes

# Create a new column representing the Indoor or Outdoor value
matches["courtType"] = matches["Court"].astype("category").cat.codes
# Create a new column representing the kind of surface
matches["groundType"] = matches["Surface"].astype("category").cat.codes
# Create a new column representing a code for each opponent
matches["opponentCode"] = matches["Player_2"].astype("category").cat.codes
# Create a new column representing a code for each opponent
matches["playerCode"] = matches["Player_1"].astype("category").cat.codes
# Create a new column representing a code for each opponent
matches["TournamentCode"] = matches["Tournament"].astype("category").cat.codes
# Create a new column representing if the player1 won the game with a 1 or if it didn't with a 0
matches["target"] = (matches["Winner"] == matches["Player_1"]).astype("int")


# Training the machine learning model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 200, min_samples_split = 3, random_state = 1)

# Divide the dataset into train and test sections
conditionDate = matches['date'] > '2005-01-01'
conditionDate2 = matches['date'] < '2023-10-29'
combinedCondition = conditionDate & conditionDate2
train = matches[combinedCondition]

test = matches[matches["date"] > '2023-10-29']

# A list of all the prdictors
predictors = ["playerCode", "opponentCode", "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]

# Fit the random forest model, fitting the predictors and trying to predict the target
rf.fit(train[predictors], train["target"])

prediction = rf.predict(test[predictors])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test['target'], prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

probabilities = rf.predict_proba(test[predictors])
probabilities

def get_player_id(player_name, role="Player_1"):
    if role == "Player_1":
        row = matches.loc[matches["Player_1"] == player_name, "playerCode"]
    else:
        row = matches.loc[matches["Player_2"] == player_name, "opponentCode"]

    if not row.empty:
        return row.iloc[0]
    else:
        return None

# Utilisation de la fonction pour obtenir les identifiants
firstPlayer = "Lehecka J."
firstOpponent = "Thompson J."

firstPlayerCode = get_player_id(firstPlayer, role="Player_1")
firstOpponentCode = get_player_id(firstOpponent, role="Player_2")

secondPlayer = firstOpponent
secondOpponent = firstPlayer

secondPlayerCode = get_player_id(secondPlayer, role="Player_1")
secondOpponentCode = get_player_id(secondOpponent, role="Player_2")

print("ID for FIRSTPLAYER:", firstPlayer, firstPlayerCode)
print("ID for FIRST OPPONENT:", firstOpponent, firstOpponentCode)
print("ID for SECOND PLAYER:", secondPlayer, secondPlayerCode)
print("ID for SECOND OPPONENT:", secondOpponent, secondOpponentCode)

# First Data
firstPlayerCode = firstPlayerCode
firstOpponentCode = firstOpponentCode
firstPlayerRank = 30
firstPlayerPoints = 1645
firstPlayerOdds = 1.14

# Second Data
secondPlayerCode = secondPlayerCode
secondOpponentCode = secondOpponentCode
secondPlayerRank = 37
secondPlayerPoints = 1400
secondPlayerOdds = 5.6

firstData = {
    "playerCode": firstPlayerCode,
    "opponentCode": firstOpponentCode,
    "Rank_1": firstPlayerRank,
    "Rank_2": secondPlayerRank,
    "Pts_1": firstPlayerPoints,
    "Pts_2": secondPlayerPoints,
    "Odd_1": firstPlayerOdds,
    "Odd_2": secondPlayerOdds,
}

secondData = {
    "playerCode": secondPlayerCode,
    "opponentCode": secondOpponentCode,
    "Rank_1": secondPlayerRank,
    "Rank_2": firstPlayerRank,
    "Pts_1": secondPlayerPoints,
    "Pts_2": firstPlayerPoints,
    "Odd_1": secondPlayerOdds,
    "Odd_2": firstPlayerOdds,
}


# Convert firstData to a DataFrame with a single row
firstData_df = pd.DataFrame([firstData])
secondData_df = pd.DataFrame([secondData])

# Extract predictors in the same order as used during training
firstDataInput = firstData_df[predictors]
secondDataInput = secondData_df[predictors]

# Make predictions
prediction = rf.predict(firstDataInput)


# Make a prediction on the new data
print(f"{firstPlayer} win result is: {prediction[0]}")
prediction = rf.predict(secondDataInput)
print(f"{secondPlayer} win result is: {prediction[0]}")

probabilityPlayerOne = rf.predict_proba(firstDataInput)
print(f"{firstPlayer} probability is: {probabilityPlayerOne}")

probabilityPlayerTwo = rf.predict_proba(secondDataInput)
print(f"{secondPlayer} probability is: {probabilityPlayerTwo}")

probabilityPlayerOne.shape

confidence = ((probabilityPlayerOne[0, 1] + probabilityPlayerTwo[0, 0]) / 2)
confidence

# Generate prediction, passing the test data and the predictors
preds = rf.predict(test[predictors])

# Need to describe the accuracy
from sklearn.metrics import accuracy_score

# Calculate the accuracy passing the test data, with the predictors and the prediction
acc = accuracy_score(test["target"], preds)

acc
