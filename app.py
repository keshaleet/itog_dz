import os
import requests
import tarfile
import email
import re
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
import joblib
from flask import Flask, request, jsonify

# Этап 1: Сбор и обработка данных

# Скачивание датасета
url = "https://spamassassin.apache.org/old/publiccorpus/"
files = ["20021010_easy_ham.tar.bz2", "20021010_hard_ham.tar.bz2", "20021010_spam.tar.bz2"]

for file in files:
    if not os.path.exists(file):
        response = requests.get(url + file, stream=True)
        with open(file, 'wb') as f:
            f.write(response.raw.read())

# Распаковка датасета
for file in files:
    if not os.path.exists(file.replace('.tar.bz2', '')):
        with tarfile.open(file, 'r:bz2') as tar:
            tar.extractall()

# Обработка данных
def clean_email(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_emails(directory, limit=100):
    emails = []
    for filename in os.listdir(directory)[:limit]:
        with open(os.path.join(directory, filename), 'r', encoding='latin-1') as f:
            msg = email.message_from_file(f)
            text = get_email_text(msg)
            emails.append(clean_email(text))
    return emails

def get_email_text(msg):
    if msg.is_multipart():
        return ' '.join(get_email_text(part) for part in msg.get_payload())
    else:
        return msg.get_payload(decode=True).decode(errors='ignore')

spam_emails = load_emails('spam', limit=100)
ham_emails = load_emails('easy_ham', limit=100) + load_emails('hard_ham', limit=100)

data = {'text': spam_emails + ham_emails, 'label': ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)}
df = pd.DataFrame(data)

# Определяем функцию для получения эмбеддингов
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Проверьте, существует ли обученная модель
model_file = 'spam_classifier.pkl'
tokenizer_dir = 'spam_tokenizer'
metrics_file = 'model_metrics.txt'

def save_metrics(f1, recall, precision):
    with open(metrics_file, 'w') as f:
        f.write(f"F1: {f1}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")

def load_metrics():
    with open(metrics_file, 'r') as f:
        metrics = f.read()
    return metrics

if not os.path.exists(model_file) or not os.path.exists(tokenizer_dir):
    # Этап 2: Обучение модели ML

    # Загрузка токенизатора и модели BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = get_embeddings(df['text'].tolist())

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Обучение модели
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Оценка модели на валидационной выборке
    y_val_pred = clf.predict(X_val)
    print(f'Validation Precision: {precision_score(y_val, y_val_pred, pos_label="spam")}')
    print(f'Validation Recall: {recall_score(y_val, y_val_pred, pos_label="spam")}')
    print(f'Validation F1 Score: {f1_score(y_val, y_val_pred, pos_label="spam")}')

    # Оценка модели на тестовой выборке
    y_test_pred = clf.predict(X_test)
    print(f'Test Precision: {precision_score(y_test, y_test_pred, pos_label="spam")}')
    print(f'Test Recall: {recall_score(y_test, y_test_pred, pos_label="spam")}')
    print(f'Test F1 Score: {f1_score(y_test, y_test_pred, pos_label="spam")}')

    # Кросс-валидация
    scores = cross_val_score(clf, embeddings, df['label'], cv=5, scoring='f1_macro')
    print(f'Cross-validation F1: {scores.mean()}')

    # Этап 3: Упаковка модели

    # Сохранение модели
    joblib.dump(clf, model_file)

    # Сохранение токенизатора
    tokenizer.save_pretrained(tokenizer_dir)

else:
    # Загружаем уже обученную модель и токенизатор
    clf = joblib.load(model_file)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    model = BertModel.from_pretrained('bert-base-uncased')


# Этап 4: Публикация модели

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    embeddings = get_embeddings([text])
    prediction = clf.predict(embeddings)
    return jsonify({'prediction': prediction[0]})

# Функционал для тестирования модели
def test_model(text):
    embeddings = get_embeddings([text])
    prediction = clf.predict(embeddings)
    return prediction[0]

if __name__ == '__main__':
    # Пример тестирования модели
    test_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
    prediction = test_model(test_text)
    print(f'Test text: {test_text}')
    print(f'Prediction: {prediction}')

    # Запуск Flask приложения
    app.run(debug=True)