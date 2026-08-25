import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- Step 1: Load Excel file ----------
df = pd.read_excel("customer_churn.xlsx")
df.columns = df.columns.str.strip()   

print("Columns in dataset:", df.columns.tolist())

# ---------- Step 2: Select features and target ----------
X = df[['Customer Age', 'Monthly Charges', 'Tenure',
        'Contract Type', 'Internet Service',
        'Support Calls', 'Total Spend']]
y = df['Churn']

# ---------- Step 3: Split data ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Step 4: Train Random Forest ----------
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# ---------- Step 5: Prediction & Accuracy ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Step 6: User Input Prediction ----------
print("\n--- Customer Churn Prediction ---")
age = float(input("Customer Age: "))
mc = float(input("Monthly Charges: "))
tenure = float(input("Tenure (months): "))
contract = int(input("Contract Type (1=Long, 0=Monthly): "))
internet = int(input("Internet Service (1=Yes, 0=No): "))
calls = int(input("Support Calls: "))
spend = float(input("Total Spend: "))

result = model.predict([[age, mc, tenure, contract, internet, calls, spend]])

if result[0] == 1:
    print("Customer is likely to CHURN 💔")
else:
    print("Customer is likely to STAY 💚")
