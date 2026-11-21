from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    "Win a free iPhone now",
    "Claim your prize today",
    "Congratulations you won money",
    "Get free cash now",
    "Click here to win a lottery",
    "Meeting scheduled for tomorrow",
    "Please send the project report",
    "Hello how are you",
    "Your assignment is completed",
    "Let's meet tomorrow"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("EMAIL SPAM DETECTOR")
print("-------------------")
print("Model Accuracy:", accuracy * 100, "%")

while True:
    email = input("\nEnter an email (type exit to stop): ")

    if email.lower() == "exit":
        break

    email_vector = vectorizer.transform([email])
    result = model.predict(email_vector)

    print("Prediction:", result[0])
