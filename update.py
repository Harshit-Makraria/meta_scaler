import json
import re
with open("training/train_colab_3b_conversational.ipynb", "r", encoding="utf-8") as f:
    txt = f.read()

txt = txt.replace("TRAIN_MAX_TOKENS=150", "TRAIN_MAX_TOKENS=256")
txt = txt.replace("TRAIN_MAX_TOKENS = 150", "TRAIN_MAX_TOKENS = 256")
txt = txt.replace("DEMO_MAX_TOKENS = 300", "DEMO_MAX_TOKENS = 400")

def replacer(match):
    return """match = re.search(r'(?i)(?:RECOMMENDATION|recommendation|decision|action|I recommend|my recommendation is|we should)\\\s*:?\\\\s*([A-Za-z_]+)', text)\n    if match:\n        word = re.sub(r'[^a-z_]', '', match.group(1).lower())\n        if word in ACTION_MAP:\n            return ACTION_MAP[word]\n    \n    tail = text[-200:].lower()\n    for w in set(ACTION_MAP.keys()):\n        if w in tail:\n            return ACTION_MAP[w]\n    return ActionType.EXECUTE"""

pattern = r"for line in text\.split.*?return ACTION_MAP\.get\(w, ActionType\.EXECUTE\)"
txt = re.sub(pattern, replacer, txt, flags=re.DOTALL)

with open("training/train_colab_3b_conversational.ipynb", "w", encoding="utf-8") as f:
    f.write(txt)
print("Updated parsing!")
