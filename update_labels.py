import json
with open("training/train_colab_3b_conversational.ipynb", "r", encoding="utf-8") as f:
    txt = f.read()

txt = txt.replace("grpo-qwen1.5b-quick-30ep", "grpo-qwen3b-conversational")
txt = txt.replace("tags=['grpo','qwen1.5b','cofounder','hackathon','quick']", "tags=['grpo','qwen3b','conversational','cofounder']")
txt = txt.replace("train_colab_quick", "train_colab_3b_conversational")
txt = txt.replace("the-pivot-lora", "co-founder-3b-lora")

with open("training/train_colab_3b_conversational.ipynb", "w", encoding="utf-8") as f:
    f.write(txt)
print("Updated WandB and HF names!")
