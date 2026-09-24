"""
Split caption dataset into train and validation sets
"""
import json
from sklearn.model_selection import train_test_split

def main():
    print("Creating train/validation split...")

    # Load all data
    with open("training/data/captions_all.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Total examples: {len(data)}")

    # 90/10 split
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    # Save splits
    with open("training/data/captions_train.jsonl", "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")

    with open("training/data/captions_val.jsonl", "w") as f:
        for example in val_data:
            f.write(json.dumps(example) + "\n")

    print(f"✓ Train examples: {len(train_data)}")
    print(f"✓ Validation examples: {len(val_data)}")
    print(f"✓ Files created:")
    print(f"    - training/data/captions_train.jsonl")
    print(f"    - training/data/captions_val.jsonl")

if __name__ == "__main__":
    main()
