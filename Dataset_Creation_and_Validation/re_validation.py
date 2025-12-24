import torch
import ast
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Dataset Class
class RelationDataset(Dataset):
    def __init__(self, head_tail_file, ner_file, relation_file, tokenizer, relation_map, max_length=128):
        self.tokenizer = tokenizer
        self.relation_map = relation_map
        self.max_length = max_length
        self.samples = self.load_data(head_tail_file, ner_file, relation_file)

    def load_data(self, head_tail_file, ner_file, relation_file):
        """Loads all sentences from the input files using IMGID as the key."""

        # Load all files into dictionaries using IMGID as the key
        head_tail_dict = self.load_head_tail(head_tail_file)
        ner_dict = self.load_ner_labels(ner_file)
        relation_dict = self.load_relations(relation_file)

        # Merge all data using IMGID
        all_img_ids = set(head_tail_dict.keys()) | set(ner_dict.keys()) | set(relation_dict.keys())

        samples = []
        for img_id in all_img_ids:
            head = head_tail_dict.get(img_id, {}).get("head", "Unknown")
            tail = head_tail_dict.get(img_id, {}).get("tail", "Unknown")
            relation = relation_dict.get(img_id, "None")
            ner_labels = ner_dict.get(img_id, [])

            samples.append({
                "img_id": img_id,
                "head": head,
                "tail": tail,
                "relation": relation,
                "ner": ner_labels
            })

        #print(f"✅ Loaded {len(samples)} samples from {relation_file}")
        return samples

    def load_head_tail(self, file_path):
        """Reads head-tail entity pairs using IMGID as the key."""
        head_tail_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            img_id = None
            for line in f:
                line = line.strip()
                if line.startswith("IMGID"):
                    img_id = line.split(":")[1].strip()
                    head_tail_dict[img_id] = {}
                elif line.startswith("Head Entity") and img_id:
                    head_tail_dict[img_id]["head"] = line.split(":")[1].strip()
                elif line.startswith("Tail Entity") and img_id:
                    head_tail_dict[img_id]["tail"] = line.split(":")[1].strip()
        return head_tail_dict

    def load_ner_labels(self, file_path):
        """Reads the NER labels using IMGID as the key."""
        ner_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            img_id = None
            for line in f:
                line = line.strip()
                if line.startswith("IMGID"):
                    img_id = line.split(":")[1].strip()
                    ner_dict[img_id] = []
                elif line and img_id:
                    ner_dict[img_id].append(line)
        return ner_dict

    def load_relations(self, file_path):
        """Loads relations using IMGID as the key."""
        relation_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = ast.literal_eval(line)  # Convert dictionary string to actual dictionary
                    relation_dict[item["img_id"]] = item["relation"]
                except Exception as e:
                    print(f"Skipping invalid line in {file_path}: {line} | Exception: {e}")

        #print(f"✅ Loaded {len(relation_dict)} relations from {file_path}")
        return relation_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Ensures each sample gets processed correctly."""
        item = self.samples[idx]
        input_text = f"{item['head']} [SEP] {item['tail']}"
        encoded = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        label = self.relation_map.get(item['relation'], self.relation_map["None"])  # Default to 'None' if missing
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dynamically extract correct relation labels
def extract_relations(file_path):
    unique_relations = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = ast.literal_eval(line.strip())  # Convert string to dictionary safely
                unique_relations.add(item["relation"])
            except Exception as e:
                print(f"Error processing line: {line.strip()} | Exception: {e}")
    return sorted(unique_relations)

# Get relations from train and test data
train_relations = extract_relations("/content/train_relations.txt")
test_relations = extract_relations("/content/test_relations.txt")

# Ensure "None" is included
relation_list = sorted(set(train_relations + test_relations + ["None"]))
relation_map = {relation: i for i, relation in enumerate(relation_list)}

# Load Train and Test Datasets
train_dataset = RelationDataset("/content/train_head_tail.txt", "/content/train_ner_labels.txt", "/content/train_relations.txt", tokenizer, relation_map)
test_dataset = RelationDataset("/content/test_head_tail.txt", "/content/test_ner_labels.txt", "/content/test_relations.txt", tokenizer, relation_map)

# Model Initialization
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(relation_list)).to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="/content/results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/content/logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Prediction Function (Excludes "None" relation)
def predict(test_head_tail, test_ner, test_relation):
    test_dataset = RelationDataset(test_head_tail, test_ner, test_relation, tokenizer, relation_map)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).tolist()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().tolist())

    # Remove "None" relation from metrics calculation
    valid_indices = [i for i in range(len(true_labels)) if relation_list[true_labels[i]] != "None"]
    filtered_true = [true_labels[i] for i in valid_indices]
    filtered_pred = [predictions[i] for i in valid_indices]

    # Compute metrics
    report = classification_report(
        filtered_true, filtered_pred, labels=list(range(len(relation_list) - 1)),  # Excludes "None"
        target_names=[r for r in relation_list if r != "None"], digits=4
    )
    accuracy = accuracy_score(filtered_true, filtered_pred)

    # Print results
    print(report)
    print(f"Accuracy: {accuracy:.4f}")

# Run Prediction
predict("/content/test_head_tail.txt", "/content/test_ner_labels.txt", "/content/test_relations.txt")
