import pandas as pd

df_bayes = pd.read_csv("scores_bayes.csv")
df_complementNB = pd.read_csv("scores_complement_nb.csv")
df_KN = pd.read_csv("scores_neighbors.csv")
df_SVC = pd.read_csv("scores_svc.csv")

print("bayes", df_bayes.loc[df_bayes['rank_test_score'] == 1])
print("KN", df_KN.loc[df_KN['rank_test_score'] == 1])
print("CNB", df_complementNB.loc[df_complementNB['rank_test_score'] == 1])
print("SVC", df_SVC.loc[df_SVC['rank_test_score'] == 1])