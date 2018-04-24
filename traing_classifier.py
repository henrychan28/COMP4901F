from textblob.classifiers import NaiveBayesClassifier
import pickle
train = []
emotions = ['ANGER', 'FEAR', 'JOY', 'LOVE', 'SADNESS', 'SURPRISE']
for emotion in emotions:
    with open('NLP_Training_Data/{0}'.format(emotion)) as f:
        content = f.readlines()
    for x in content:
        train.append((x.strip(), emotion))
    print('[{0}]Number of entry: {1} '.format(emotion, len(content)))
print("Start Training...")
cl = NaiveBayesClassifier(train)
filename='NBC.tb'
print("Saving model...")
pickle.dump(cl, open(filename, 'wb'))
print("Model saved")
