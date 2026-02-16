from sklearn.model_selection import cross_val_score

def choose_evaluation_metric(y, task_type):
    if task_type == 'regression':
        return 'neg_root_mean_squared_error'
    
    class_distrubution = y.value_counts(normalize=True)
    
    if len(class_distrubution) == 2 and class_distrubution.min() < 0.1:
        return 'roc_auc'
    
    return 'f1_weighted'
        
def evaluate_model(model, X, y, task_type):
    
    scoring = choose_evaluation_metric(y, task_type)
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    return scores.mean()