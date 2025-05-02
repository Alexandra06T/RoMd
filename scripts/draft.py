import pandas as pd

train_data = pd.read_csv('../train_data_M.csv')
valid_data = pd.read_csv('../validation_data_M.csv')
test_data = pd.read_csv('../test_data_M.csv')

train_data['sentence_len'] = train_data['sample'].apply(lambda x: len(x.split()))
test_data['sentence_len'] = test_data['sample'].apply(lambda x: len(x.split()))
valid_data['sentence_len'] = valid_data['sample'].apply(lambda x: len(x.split()))

print(train_data['sentence_len'].mean())
print(test_data['sentence_len'].mean())
print(valid_data['sentence_len'].mean())
print(train_data['sentence_len'].max())
print(test_data['sentence_len'].max())
print(valid_data['sentence_len'].max())
print(train_data['sentence_len'].min())
print(test_data['sentence_len'].min())
print(valid_data['sentence_len'].min())