import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

file_path = 'spam_data.csv'
df = pd.read_csv(file_path)

print("--- DATASET SUMMARY ---")
print(df['is_spam'].value_counts())  # This will show you exactly how many are Spam vs Legit
print("-----------------------\n")

feature_cols = ['words', 'links', 'capital_words', 'spam_word_count']
X = df[feature_cols]
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


def evaluate_email_text(text):
    word_count = len(text.split())
    links_count = len(re.findall(r'http[s]?://|www\.|.com|.org', text))
    cap_words = len(re.findall(r'\b[A-Z]{2,}\b', text))

    trigger_list = ['free', 'win', 'money', 'urgent', 'cash', 'prize', 'offer', 'lottery']
    spam_word_count = sum(1 for word in text.lower().split() if word in trigger_list)

    features_df = pd.DataFrame([[word_count, links_count, cap_words, spam_word_count]],
                               columns=feature_cols)

    prediction = model.predict(features_df)
    return "SPAM" if prediction[0] == 1 else "LEGITIMATE", features_df


spam_msg = "URGENT! You win a FREE PRIZE. Visit www.win-money.com to claim your CASH."
result1, feats1 = evaluate_email_text(spam_msg)
print(f"\nManual Spam Test: {result1}")

legit_msg = "Hi, are we still meeting for coffee tomorrow? Let me know if you are free."
result2, feats2 = evaluate_email_text(legit_msg)
print(f"Manual Legit Test: {result2}")


plt.figure(figsize=(6, 4))
sns.countplot(x='is_spam', data=df, hue='is_spam', palette='viridis', legend=False)
plt.title('Task 2: Spam vs Legitimate Distribution')
plt.xlabel('Class (0 = Legitimate, 1 = Spam)')
plt.ylabel('Number of Emails')
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.title('Task 2: Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()