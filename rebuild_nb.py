import json
import re

with open("training/train_colab_quick.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Replace base model
        source = source.replace("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
        
        # Replace max tokens
        source = source.replace("TRAIN_MAX_TOKENS = 16", "TRAIN_MAX_TOKENS = 256")
        source = source.replace("DEMO_MAX_TOKENS  = 200", "DEMO_MAX_TOKENS  = 400")
        
        # Replace parsing logic
        old_parse_block = """    for line in text.split('\\n'):
        stripped = line.strip().lower()
        for prefix in ('decision:', 'recommendation:', 'action:'):
            if stripped.startswith(prefix):
                rest = stripped[len(prefix):].strip()
                word = re.sub(r'[^a-z_]', '', rest.split()[0]) if rest.split() else ''
                if word in ACTION_MAP:
                    return ACTION_MAP[word]
    # Fallback: first word of full output
    w = re.sub(r'[^a-z_]', '', text.lower().split()[0]) if text.strip() else 'execute'
    return ACTION_MAP.get(w, ActionType.EXECUTE)"""
        
        new_parse_block = """    # Robust conversational extraction
    match = re.search(r'(?i)(?:RECOMMENDATION|recommendation|decision|action|I recommend|my recommendation is|we should)\s*:?\s*([A-Za-z_]+)', text)
    if match:
        word = re.sub(r'[^a-z_]', '', match.group(1).lower())
        if word in ACTION_MAP:
            return ACTION_MAP[word]
            
    # Fallback: scan for any valid action in the last 150 chars
    tail = text[-150:].lower()
    for w in ACTION_MAP.keys():
        if w in tail:
            return ACTION_MAP[w]
            
    return ActionType.EXECUTE"""
        
        if old_parse_block in source:
            source = source.replace(old_parse_block, new_parse_block)
            
        # Replace tracking labels
        source = source.replace("grpo-qwen1.5b-quick-30ep", "grpo-qwen3b-conversational")
        source = source.replace("tags=['grpo','qwen1.5b','cofounder','hackathon','quick']", "tags=['grpo','qwen3b','conversational','cofounder']")
        source = source.replace("the-pivot-lora", "co-founder-3b-lora")
        
        # Make sure source is correctly split into lines again for the notebook format
        lines = []
        for line in source.splitlines():
            lines.append(line + "\\n")
        # Remove trailing \\n from the last line if necessary (jupyter does this sometimes, but standard is fine)
        if lines and lines[-1].endswith("\\n"):
            lines[-1] = lines[-1][:-1]
            
        cell["source"] = lines

with open("training/train_colab_3b_conversational.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook Rebuilt Successfully")
