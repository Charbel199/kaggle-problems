import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
def preprocess_data(df):
    df.loc[df['Name'].str.find('Mr.') != -1, 'title'] = int(0)
    df.loc[df['Name'].str.find('Dr.') != -1, 'title'] = int(2)
    df['title'] = df['title'].fillna(int(1))
    df=df.drop('Name', axis=1)
    df=df.drop('Ticket', axis=1)
    df.Sex = pd.factorize(df.Sex)[0]
    df['Cabin'] = df['Cabin'].fillna('not-defined')
    df['Cabin'] = df['Cabin'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))
    df['Cabin'] = df['Cabin'].apply(lambda x: x.strip()[0])
    df.Cabin = pd.factorize(df.Cabin)[0]
    df.Embarked = pd.factorize(df.Embarked)[0]
    return df


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
results_df = pd.read_csv('data/gender_submission.csv')
test_results_df = pd.merge(test_df, results_df, on='PassengerId')

processed_train_df = preprocess_data(train_df)

print(processed_train_df.to_string())
# print(test_theory.to_string())
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)s
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
#
# print(train_df.to_string())
