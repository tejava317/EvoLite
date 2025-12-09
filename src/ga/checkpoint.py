import os
import csv

def save_checkpoint_csv(population, name, save_dir="src/ga/checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"population_{name}.csv")

    fieldnames = ["workflow_roles", "pass_at_k", "tokens"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in population:
            wf = entry["workflow"]
            fitness = entry["fitness"]

            workflow_roles = " -> ".join([agent.role for agent in wf.agents])

            row = {
                "workflow_roles": workflow_roles,
                "pass_at_k": fitness.get("pass_at_k", None),
                "tokens": fitness.get("token", None),
            }

            writer.writerow(row)

    print(f"[CSV Checkpoint Saved] {path}")
