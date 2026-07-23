import os
import pandas as pd
from config import SOLVE_THRESHOLD


def format_ascii_table(df):
    headers = df.columns.tolist()
    col_widths = [max(df[col].astype(str).map(len).max(), len(col)) + 2 for col in headers]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {headers[i].ljust(col_widths[i]-2)} " for i in range(len(headers))) + "|"
    
    rows = [sep, header_row, sep]
    
    for _, row in df.iterrows():
        rows.append("|" + "|".join(f" {str(row[col]).ljust(col_widths[i]-2)} " for i, col in enumerate(headers)) + "|")
        
    rows.append(sep)
    
    return "\n".join(rows)

def analyze_logs(base_dir):
    results = []
    
    for arch in ["mlp", "pqc"]:
        arch_dir = os.path.join(base_dir, arch)
        
        if not os.path.exists(arch_dir):
            continue
            
        for opt in os.listdir(arch_dir):
            file = os.path.join(arch_dir, opt, "training_logs.csv")
            
            if not os.path.isfile(file):
                continue
                
            df = pd.read_csv(file)
            num_seeds = df["seed"].nunique()
            solves = 0
            
            steps_to_solve = []
            eps_to_solve = []
            times_to_solve = []
            grad_norms = []
            cond_nums = []
            
            for seed, seed_group in df.groupby("seed"):
                solved_rows = seed_group[seed_group["eval_mean_reward"] >= SOLVE_THRESHOLD]
                
                if not solved_rows.empty:
                    solves += 1
                    first_solve_idx = solved_rows["global_episode"].idxmin()
                    solve_row = seed_group.loc[first_solve_idx]
                    
                    steps_to_solve.append(solve_row["total_steps"])
                    eps_to_solve.append(solve_row["global_episode"])
                    
                    if "time_seconds" in seed_group.columns:
                        times_to_solve.append(solve_row["time_seconds"])
                        
                if "grad_norm" in seed_group.columns:
                    grad_norms.append(seed_group["grad_norm"].mean())
                    
                if "condition_number" in seed_group.columns:
                    valid_conds = pd.to_numeric(seed_group["condition_number"], errors='coerce').dropna()
                    
                    if not valid_conds.empty:
                        cond_nums.append(valid_conds.mean())
                        
            success_rate = f"{(solves / num_seeds) * 100 if num_seeds > 0 else 0:.1f}% ({solves}/{num_seeds})"
            mean_steps = sum(steps_to_solve) / len(steps_to_solve) if steps_to_solve else 0
            mean_eps = sum(eps_to_solve) / len(eps_to_solve) if eps_to_solve else 0
            mean_time = sum(times_to_solve) / len(times_to_solve) if times_to_solve else 0
            mean_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
            mean_cond = round(sum(cond_nums) / len(cond_nums), 4) if cond_nums else "N/A"
            
            results.append({
                "Architecture": arch.upper(),
                "Optimizer": opt.upper(),
                "Success Rate": success_rate,
                "Mean Steps": round(mean_steps, 1),
                "Mean Episodes": round(mean_eps, 1),
                "Mean Time (s)": round(mean_time, 2),
                "Mean Grad Norm": round(mean_grad, 4),
                "Mean Cond Num": mean_cond
            })
            
    return results

if __name__ == "__main__":
    results_data = analyze_logs("logs")
    
    if results_data:
        df_results = pd.DataFrame(results_data)
        df_results = df_results.sort_values(by=["Architecture", "Optimizer"])
        
        output_string = format_ascii_table(df_results)
        
        with open("analysis_summary.txt", "w") as f:
            f.write(output_string)
            f.write("\n")
            
        print("Analysis complete. Saved to analysis_summary.txt.")
        