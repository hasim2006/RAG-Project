"""
export_finetune_data.py — Prepares fine-tuning datasets from user feedback.
"""
import json
import os

INPUT_FILE = "dataset_for_finetuning.jsonl"
OUTPUT_FILE = "formatted_finetune_dataset.jsonl"

def process_logs():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File '{INPUT_FILE}' not found. Record some negative feedback first.")
        return

    formatted_entries = []
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            
            # Format into standard LLM fine-tuning chat template
            chat_format = {
                "messages": [
                    {"role": "system", "content": "You are a helpful corporate policy assistant."},
                    {"role": "user", "content": entry.get("query", "")},
                    # Note: You can replace 'answer' with corrected ideal responses during review
                    {"role": "assistant", "content": entry.get("answer", "")}
                ]
            }
            formatted_entries.append(chat_format)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for item in formatted_entries:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(formatted_entries)} entries into '{OUTPUT_FILE}'!")

if __name__ == "__main__":
    process_logs()