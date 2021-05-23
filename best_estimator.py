import pandas as pd

scores = [
    'scores_bayes.csv',
    'scores_neighbors.csv',
    'scores_svc.csv',
    'scores_complement_nb.csv',
]

for score in scores:


    df = pd.read_csv(score)

    df_best = df.loc[df["rank_test_score"] == 1].index

    print(df_best[0])


    best_stimator = df.iloc[df_best[0], 12:-1]

    best_stimator.to_csv(f'best_{score}')

