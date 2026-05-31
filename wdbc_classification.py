from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ==========================
# STEP 1: LOAD DATASET
# ==========================

wdbc = load_breast_cancer()

X = wdbc.data
y = wdbc.target

print("Data Shape:", X.shape)
print("Target Shape:", y.shape)
print("Target Names:", wdbc.target_names)
print("First 5 Feature Names:", wdbc.feature_names[:5])

# ==========================
# STEP 2: SPLIT DATASET
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# STEP 3: DEFAULT CLASSIFIER
# ==========================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Metrics")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))

# ==========================
# STEP 4: TRAIN & COMPARE CLASSIFIERS
# ==========================

models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("\nClassifier Comparison")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")

# ==========================
# STEP 5: FIND BEST CLASSIFIER
# ==========================

best_classifier = max(results, key=results.get)

print("\nBest Classifier:", best_classifier)
print("Best Accuracy:", results[best_classifier])

best_model = models[best_classifier]

best_prediction = best_model.predict(X_test)

# ==========================
# STEP 6: CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, best_prediction)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("wdbc_classification_matrix.png")
plt.close()

# ==========================
# STEP 7: SCATTER PLOT
# ==========================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Breast Cancer Dataset")

plt.savefig("wdbc_classification_scatter.png")
plt.close()

print("\nImages saved successfully.")