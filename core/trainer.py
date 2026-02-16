def train_and_select_best(models, X, y, task_type, evaluator):
    
    results = {}
    
    for name, model in models.items():
        score = evaluator(model, X, y, task_type)
        results[name] = score
        print(f"{name}: {score:.4f}")
        
    best_model_name = max(results, key=results.get) #type: ignore
    best_model = models[best_model_name]
    
    best_model.fit(X, y)
    
    return best_model_name, best_model, results