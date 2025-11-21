# credit_tree.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 60
ages = np.random.randint(18, 70, size=n)
income = np.random.randint(800, 10000, size=n)
credit_score = np.random.randint(300, 850, size=n)
has_job = np.random.choice([0, 1], size=n, p=[0.15, 0.85])
existing_debt = np.random.randint(0, 20000, size=n)
loan_amount = np.random.randint(500, 20000, size=n)

approval_score = (
    0.4 * (income / income.max()) +
    0.4 * (credit_score / 850) +
    0.1 * has_job -
    0.1 * (existing_debt / (existing_debt.max() + 1)) -
    0.15 * (loan_amount / loan_amount.max())
)
noise = np.random.normal(0, 0.03, size=n)
approved = (approval_score + noise > 0.45).astype(int)

df = pd.DataFrame({
    "age": ages,
    "income": income,
    "credit_score": credit_score,
    "has_job": has_job,
    "existing_debt": existing_debt,
    "loan_amount": loan_amount,
    "approved": approved
})

X = df.drop(columns=["approved"])
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Tree rules:\n", export_text(clf, feature_names=list(X.columns)))

plt.figure(figsize=(16,10))
plot_tree(clf, feature_names=X.columns, class_names=["Rechazado","Aprobado"], filled=False, rounded=True)
plt.title("Árbol de Decisión - Aprobación de crédito")
plt.tight_layout()
plt.savefig("tree_credit.png", dpi=200)
plt.show()

df.to_csv("credit_dataset_simulated.csv", index=False)
