import itertools
import pandas as pd

def grid_search_c3(task_fn, icon_folder_name, test_folder_name, output_csv="hyperparameter_optimization.csv"):
    contrast_thresholds = [0.02, 0.03, 0.04, 0.05]
    octave_layers = [4, 6, 8, 10]
    ransac_thresholds = [1.0, 2.0, 3.0]
    lowe_ratios = [0.7, 0.75, 0.8]
    edge_thresholds = [10, 15, 20]
    gaussian_blurs = [1.0, 1.3, 1.6]
    # ransac_iterations = [100, 250, 500, 750, 1000]


    combinations = list(itertools.product(
        contrast_thresholds, 
        octave_layers, 
        ransac_thresholds,
        lowe_ratios,
        edge_thresholds,
        gaussian_blurs
        # ransac_iterations
    ))
    
    results = []
    
    print(f"Starting Grid Search over {len(combinations)} combinations...")
    
    for i, (ct, oct_l, rt, lr, et, gb) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: ct={ct}, oct={oct_l}, rt={rt}, lr={lr}, et={et}, gb={gb}")
        
        acc, tpr, fpr, fnr = task_fn(
            icon_folder_name, 
            test_folder_name,
            contrast_threshold=ct,
            n_octave_layers=oct_l,
            ransac_threshold=rt,
            lowe_ratio=lr,
            edge_threshold=et,
            gaussian_blur=gb,
            # ransac_iterations=ri,
            verbose=False,
            tuning=True
        )
        
        results.append({
            "contrast_threshold": ct,
            "octave_layers": oct_l,
            "ransac_threshold": rt,
            "lowe_ratio": lr,
            "edge_threshold": et,
            "gaussian_blur": gb,
            "accuracy": acc,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        })
        
        # Save intermediate results so nothing is lost if canceled
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
    print(f"Grid search completed. Results saved to {output_csv}")