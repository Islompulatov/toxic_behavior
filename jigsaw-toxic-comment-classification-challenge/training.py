from data_loader import TrainData, train_test_split, collate_test, collate_train
from torch.utils.data import DataLoader



train_df, test_df = train_test_split(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\toxic_behavior\jigsaw-toxic-comment-classification-challenge\train.csv\train.csv')
train_df, test_df = TrainData(train_df), TrainData(test_df)

train_loader = DataLoader(train_df, batch_size=32, collate_fn=collate_train(vectorizer=train_df.vectorizer), shuffle=True)
test_loader = DataLoader(test_df, batch_size=32, collate_fn=collate_test(vectorizer=test_df.vectorizer))