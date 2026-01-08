import json
import pandas as pd

input_file = "data/problems_data.jsonl"
output_file = "data/problems.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        title = obj.get("title", "").strip()
        description = obj.get("description", "").strip()
        input_desc = obj.get("input_description", "").strip()
        output_desc = obj.get("output_description", "").strip()
        problem_class = obj.get("problem_class", "").strip().lower()
        score = obj.get("problem_score", None)

        # 🔹 BASIC VALIDATION (SAFE, NOT STRICT)
        if not description:
            continue
        if problem_class not in ["easy", "medium", "hard"]:
            continue
        if score is None:
            continue

        rows.append({
            "title": title,
            "description": description,
            "input_description": input_desc,
            "output_description": output_desc,
            "problem_class": problem_class,
            "problem_score": float(score)
        })

df = pd.DataFrame(rows)

# Final cleanup
df.dropna(inplace=True)

df.to_csv(output_file, index=False)
print(f"✅ Clean dataset saved with {len(df)} rows")
