# coding: utf-8
import pickle
import numpy as np
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
    
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)
    
vocab = set()
all_data = train_data + test_data
for story, question, ans in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
    
vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab) + 1
max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)
train_story_text = []
train_question_text = []
train_answers = []
for story, question, ans in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(ans)
    
train_stroy_seq = tokenizer.texts_to_sequences(train_story_text)
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len):
    X = []
    Xq = []
    Y = []
    for story, question, ans in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y = np.zeros(len(word_index) + 1)
        y[word_index[ans]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        return pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y)
        
input_train, question_train, ans_train = vectorize_stories(train_data)
