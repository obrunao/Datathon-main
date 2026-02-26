from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    return {
        "classification_report": report,
        "auc": auc,
        "confusion_matrix": cm
    }
