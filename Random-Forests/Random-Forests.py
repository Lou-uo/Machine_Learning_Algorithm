import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def load_dataset(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_path)
    dataset = pd.read_csv(full_path)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    return X, y


def preprocess_data(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, sc


def train_random_forest(X_train, y_train, n_estimators=10, criterion='entropy', random_state=0):
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state
    )
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy Score: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:")
    print(report)
    
    return cm, accuracy, report


def plot_decision_boundary(X, y, classifier, title):
    from matplotlib.colors import ListedColormap
    
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y = load_dataset('Social_Network_Ads.csv')
    
    X_train, X_test, y_train, y_test, sc = preprocess_data(X, y)
    
    classifier = train_random_forest(X_train, y_train, n_estimators=10, criterion='entropy', random_state=0)
    
    y_pred = classifier.predict(X_test)
    
    evaluate_model(y_test, y_pred)
    
    plot_decision_boundary(X_train, y_train, classifier, 'Random Forest (Training Set)')
    plot_decision_boundary(X_test, y_test, classifier, 'Random Forest (Test Set)')