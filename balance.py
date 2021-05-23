import pandas as pd




df = pd.read_csv('output_got_complete.csv')

df_true = df[df['rating_alternateName'] == 'True']
print(df_true.shape)

df_false = df[df['rating_alternateName'] == 'False'][0: df_true.shape[0]]
print(df_false.shape)


df_mixed = df[(df['rating_alternateName'] != 'True') & (df['rating_alternateName'] != 'False')][0: df_true.shape[0]]

print(df_mixed.shape)


df_balanced = pd.concat([df_false, df_mixed, df_true], ignore_index=True, sort=False)

df_balanced.to_csv('output_balanced.csv')