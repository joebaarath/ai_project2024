from flask import Flask, request, render_template, send_file
import pandas as pd
import tensorflow as tf
import re
import string
import numpy as np
import pickle

# Standard library imports
import os
import random
from csv import DictReader, DictWriter

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Machine learning and model building
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# TensorFlow and Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate, Bidirectional, LSTM, Dropout, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint


# Load the Keras model
model = tf.keras.models.load_model('./MLP_best_model.keras')

app = Flask(__name__)


# Initialise global variables
base_kaggle_input_path = "/kaggle/input"
base_kaggle_path = "/kaggle"
base_kaggle_working_path = "/kaggle/working"

label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

# Set file names
# file_train_instances = "./train_instances.csv"
# file_train_bodies = "./train_bodies.csv"
test_instances = "test_instances"
file_test_instances = f"./{test_instances}.csv"
test_bodies_name = "test_bodies"
file_test_bodies = f"./{test_bodies_name}.csv"
# file_predictions = "predictions_test.csv"
# file_valid_instances = "./valid_instances.csv"
# file_valid_bodies = "./valid_bodies.csv"

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90

# FNC Data Class
class FNCData:
    """
    Define class for Fake News Challenge data
    """
    def __init__(self, file_instances, file_bodies):
        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filename):
        """
        Read Fake News Challenge data from CSV file

        Args:
            filename: str, filename + extension

        Returns:
            rows: list, of dict per instance
        """
        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows

# Define relevant functions
def pipeline_train(test, lim_unigram):
    """
    Process train set, create relevant vectorizers

    Args:
        train: FNCData object, train set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()
    """
    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    train_stances = []
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_tfidf_track = {}
    body_tfidf_track = {}

    # # Identify unique heads and bodies
    # for instance in train.instances:
    #     head = instance['Headline']
    #     body_id = instance['Body ID']
    #     if head not in heads_track:
    #         heads.append(head)
    #         heads_track[head] = 1
    #     if body_id not in bodies_track:
    #         bodies.append(train.bodies[body_id])
    #         bodies_track[body_id] = 1
    #         body_ids.append(body_id)

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).fit(heads + bodies)  # Train and test sets

    # # Process train set
    # for instance in train.instances:
    #     head = instance['Headline']
    #     body_id = instance['Body ID']
    #     head_tf = tfreq[id_ref[head]].reshape(1, -1)
    #     body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
    #     if head not in head_tfidf_track:
    #         head_tfidf = tfidf_vectorizer.transform([head]).toarray()
    #         head_tfidf_track[head] = head_tfidf
    #     else:
    #         head_tfidf = head_tfidf_track[head]
    #     if body_id not in body_tfidf_track:
    #         body_tfidf = tfidf_vectorizer.transform([train.bodies[body_id]]).toarray()
    #         body_tfidf_track[body_id] = body_tfidf
    #     else:
    #         body_tfidf = body_tfidf_track[body_id]
    #     if (head, body_id) not in cos_track:
    #         tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
    #         cos_track[(head, body_id)] = tfidf_cos
    #     else:
    #         tfidf_cos = cos_track[(head, body_id)]
    #     feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    #     train_set.append(feat_vec)
    #     train_stances.append(label_ref[instance['Stance']])

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer

def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """
    Process test set

    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        test_set: list, of numpy arrays
    """
    # Initialise
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process test set
    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            head_bow = bow_vectorizer.transform([head]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
            heads_track[head] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[head][0]
            head_tfidf = heads_track[head][1]
        if body_id not in bodies_track:
            body_bow = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test_set.append(feat_vec)

    return test_set

def split_instances_bodies_labels_and_export_csv(test_df):
    
    test_instances = extract_data(test_df, 'instances')
    test_instances.to_csv("test_instances.csv", index=False)
    
    test_bodies = extract_data(test_df, 'bodies')
    test_bodies.to_csv(f"{test_bodies_name}.csv", index=False)
    
    # test_labels = extract_data(test_df, 'labels')
    # test_labels.to_csv("test_labels.csv", index=False)
        
    return test_instances, test_bodies, None
    

def load_model(sess):
    """
    Load TensorFlow model

    Args:
        sess: TensorFlow session
    """
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')

def save_predictions(pred, file):
    """
    Save predictions to CSV file

    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension
    """
    with open(file, 'w') as csvfile:
        fieldnames = ['Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for instance in pred:
            writer.writerow({'Stance': label_ref_rev[instance]})
            
            
def extract_data(df, data_type):
    if data_type == 'instances':
        return df[["Headline", "Body ID"]]
        # return df[["Headline", "Body ID", "Stance"]]
    elif data_type == 'bodies':
        return df[["Body ID", "articleBody"]]
    # elif data_type == 'labels':
    #     return df["Stance"]
    else:
        raise ValueError("Invalid data type specified. Choose 'instances', 'bodies', or 'labels'.")


def remove_exact_rows(merged_df):
    # Check if all rows are exactly the same
    all_same = merged_df.duplicated(keep=False)
    
    # Filter out rows where all values match
    matched_instances =  merged_df[all_same]

    # Print the matched instances
    print("Cases where all rows are exactly the same:")
    print(matched_instances)
    
    return matched_instances


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            df = pd.read_csv(file)
                        
            # Apply preprocessing to specified columns
            columns_to_preprocess = ['Headline', 'articleBody']
            for column in columns_to_preprocess:
                df[column] = df[column].astype(str).apply(preprocess_text)
            # df['stance_cat'] = df['Stance'].map({'agree':0, 'disagree':1, 'discuss':2, 'unrelated':3}).astype(int)
            ##################
            
            test_df = df
            test_instances, test_bodies, test_labels =  split_instances_bodies_labels_and_export_csv(test_df)

            raw_test = FNCData(file_test_instances, file_test_bodies)
            
            with open('bow_vectorizer.pickle', 'rb') as f:
                bow_vectorizer = pickle.load(f)
                
            with open('tfreq_vectorizer.pickle', 'rb') as f:
                tfreq_vectorizer = pickle.load(f)
                
            with open('tfidf_vectorizer.pickle', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)

            test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            
            
            # test_labels = pd.read_csv("./test_labels.csv")
            # test_labels['stance_cat'] = test_labels['Stance'].map(label_ref).astype(int)
            # stan = test_labels['stance_cat'].tolist()

            # X_test, y_test = test_set, test_labels
            X_test = test_set
            # y_test = np.array(y_test)
            
            X_test = np.stack(X_test, axis=0)
            
            # Convert to integers if not already
            # y_test = df.iloc[:, 1].astype(int)  # Access all rows of the second column and convert to int
            
            # print(y_test)
            
            print("check X_test shape")
            print(X_test.shape)
            print("check X_test value")
            print(X_test)
            predictions = model.predict(X_test)
            
            print(predictions)
            
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Add predicted classes to original DataFrame
            df['Predicted_Stance'] = predicted_classes

            # Convert predicted classes back to label names if needed
            # Map back to original stance names if necessary
            stance_mapping = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
            df['Predicted_Stance_Label'] = df['Predicted_Stance'].map(stance_mapping)
            
            df.drop('Predicted_Stance', axis=1, inplace=True)
            
            # Save the DataFrame with predictions to a new CSV file
            # Generate output filename
            original_name = os.path.splitext(file.filename)[0]  # Extract filename without extension
            output_filename = f"{original_name}_predictions.csv"
            
            df.to_csv(output_filename, index=False)
            render_template('index.html', message='Submit CSV for Stance Prediction')
            return send_file(output_filename, as_attachment=True)
            

    return render_template('index.html', message='Please Upload CSV')
    # return render_template('index.html', message='Upload new CSV')

def preprocess_text(text):
    # Series of substitutions and removals as defined
    text = re.sub(r'%', 'percent', text)
    # Replace '&' with 'and'
    text = re.sub(r'&', 'and', text)
    # Remove symbols, including punctuation marks and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove Chinese characters
    text = re.sub(r'[\u4E00-\u9FFF]+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\w+', ' ', text)
    # Remove emojis
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # Chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                      "]+", re.UNICODE)
    text = re.sub(emoj, ' ', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    return text

# Ensure templates and static folders are created and properly set up
app.run(debug=True)
