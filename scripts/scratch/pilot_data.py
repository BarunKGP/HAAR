import pickle
from sklearn.model_selection import train_test_split


with open('../../../epic-kitchens-100-annotations/EPIC_100_train.pkl', 'rb') as f:
    df_train = pickle.load(f)

df_train100 = df_train[df_train.video_id.apply(lambda x: len(x) == 7)]


participants = df_train100.participant_id.unique()
p_train, p_test = train_test_split(participants, test_size=0.15)

train = df_train100[df_train100['participant_id'].isin(p_train)]
test = df_train100[df_train100['participant_id'].isin(p_test)]

print('Train participants: ', p_train, '\n')
print('Test participants: ', p_test)

print(train.info())
print(test.info())

with open('../../data/train100_version_fix.pkl', 'xb') as handle:
    pickle.dump(train, handle)
print('Wrote train pickle')

with open('../../data/test100_version_fix.pkl', 'xb') as handle:
    pickle.dump(test, handle)
print('Wrote test pickle')


