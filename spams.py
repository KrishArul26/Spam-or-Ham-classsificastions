import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


def reading_Files(path):
    """
    Method Name:reading_Files
    Description: Reading the files which is exit in the working directory
    """
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def random_directory(path, classification):
    """
        Method Name:random_directory
        Description: To append the message and their particular classification
        """
    rows = []
    index = []
    for filename, message in reading_Files(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data1 = DataFrame({'message': [], 'class': []})

data1 = data1.append(random_directory(r'C:\Users\ragav\Downloads\spam\emails\spam', 'spam'))
data1 = data1.append(random_directory(r'C:\Users\ragav\Downloads\spam\emails\ham', 'ham'))

data1.head()

#training data using MultinomialNB classifier
vector = CountVectorizer()
counts = vector.fit_transform(data1['message'].values)

classifier = MultinomialNB()
targets = data1['class'].values
classifier.fit(counts, targets)

# saved model 
#joblib.dump(classifier, 'spam.pkl')

def prediction(example):
    clas = joblib.load('spam.pkl')
    example_counts = vector.transform(example)
    prediction = clas.predict(example_counts)
    return prediction
