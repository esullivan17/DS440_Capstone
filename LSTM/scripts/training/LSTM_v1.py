import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load your preprocessed data ===
data_path = "patient_feature_data.pt"
patient_data = torch.load(data_path)

# === Hyperparameters ===
INPUT_DIM = 100
HIDDEN_DIM = 64
NUM_CLASSES = 3
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 30  # upper bound for each sequence

# === Custom Dataset ===
class PatientDataset(Dataset):
    def __init__(self, data_dict):
        self.entries = []
        for patient_id, entry in data_dict.items():
            sorted_slices = sorted(entry["slices"], key=lambda x: x["rel_path"])
            features = [s["features"] for s in sorted_slices][:MAX_SEQUENCE_LENGTH]
            grade = entry["grade"]
            self.entries.append((features, grade))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        features, label = self.entries[idx]
        return features, label, len(features)

# === Collate Function for Padding ===
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    sequences = [torch.stack(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)  # shape: [batch, max_len, input_dim]
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

# === LSTM Classifier ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Select the last valid output from each sequence
        last_outputs = torch.stack([output[i, l - 1] for i, l in enumerate(lengths)])
        logits = self.fc(last_outputs)
        return logits

# === Stratified Train/Validation Split ===
grade_buckets = {0: [], 1: [], 2: []}
for pid, entry in patient_data.items():
    grade_buckets[entry["grade"]].append((pid, entry))

train_dict, val_dict = {}, {}
for grade, items in grade_buckets.items():
    random.shuffle(items)
    split = int(0.8 * len(items))
    for pid, entry in items[:split]:
        train_dict[pid] = entry
    for pid, entry in items[split:]:
        val_dict[pid] = entry

# === Loaders ===
train_dataset = PatientDataset(train_dict)
val_dataset = PatientDataset(val_dict)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for sequences, labels, lengths in train_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels, lengths in val_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Grade 0", "Grade 1", "Grade 2"]))
