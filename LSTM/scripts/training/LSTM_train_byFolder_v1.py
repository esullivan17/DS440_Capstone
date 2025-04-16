import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict, Counter
import random
import numpy as np

# Random Seed
SEED = 29164  # Set to a fixed seed for reproducibility
# SEED = random.randint(0, 99999)  # Uncomment to use a random seed instead

print(f"\n Random seed used: {SEED}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load preprocessed multi-sequence data 
data_path = "LSTM/data/processed/patient_feature_data_v6.pt"
patient_data = torch.load(data_path)

# Hyperparameters 
INPUT_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 30

# Dataset Class 
class MultiSeqDataset(Dataset):
    def __init__(self, patient_dict, max_seq_len=30):
        self.entries = []
        for patient_id, entry in patient_dict.items():
            label = entry["grade"]
            for i, seq in enumerate(entry["sequences"]):
                seq = seq[:max_seq_len]
                self.entries.append((patient_id, seq, label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient_id, sequence, label = self.entries[idx]
        return patient_id, sequence, label, len(sequence)

# Collate Function 
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    patient_ids, sequences, labels, lengths = zip(*batch)
    sequences = [torch.stack(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return patient_ids, padded_sequences, torch.tensor(labels), torch.tensor(lengths)

# LSTM Model 
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
        last_outputs = torch.stack([output[i, l - 1] for i, l in enumerate(lengths)])
        logits = self.fc(last_outputs)
        return logits

# Train/Validation Split 
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

train_dataset = MultiSeqDataset(train_dict, max_seq_len=MAX_SEQUENCE_LENGTH)
val_dataset = MultiSeqDataset(val_dict, max_seq_len=MAX_SEQUENCE_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Class Weights 
train_labels = [entry["grade"] for entry in train_dict.values()]
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_labels)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track Best 
best_accuracy = 0
best_f1 = 0
best_preds, best_labels, best_epoch = {}, {}, 0
best_model_state = None

# Training 
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for patient_ids, sequences, labels, lengths in train_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation 
    model.eval()
    patient_predictions = defaultdict(list)
    patient_true_labels = {}
    with torch.no_grad():
        for patient_ids, sequences, labels, lengths in val_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs, 1)

            for pid, pred, true in zip(patient_ids, predicted.cpu().numpy(), labels.cpu().numpy()):
                patient_predictions[pid].append(pred)
                patient_true_labels[pid] = true

    # Majority vote per patient
    final_preds = []
    final_labels = []
    for pid, preds in patient_predictions.items():
        vote = Counter(preds).most_common(1)[0][0]
        final_preds.append(vote)
        final_labels.append(patient_true_labels[pid])

    precision, recall, f1, _ = precision_recall_fscore_support(final_labels, final_preds, labels=[0, 1, 2], zero_division=0)
    correct = sum(p == l for p, l in zip(final_preds, final_labels))
    accuracy = correct / len(final_labels)

    print(f"\n Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}")
    print(classification_report(final_labels, final_preds, labels=[0,1,2], target_names=["Grade 0", "Grade 1", "Grade 2"], zero_division=0))

    # Check for non-zero f1-scores for all classes
    if all(f > 0 for f in f1) and accuracy > best_accuracy:
        best_accuracy = accuracy
        best_f1 = np.mean(f1)
        best_preds = final_preds.copy()
        best_labels = final_labels.copy()
        best_epoch = epoch + 1
        best_model_state = model.state_dict()

# Final Best 
print(f"\n Best Epoch: {best_epoch} | Accuracy: {best_accuracy:.2f} | Seed: {SEED}")
print(" Classification Report (Best Epoch):")
print(classification_report(best_labels, best_preds, labels=[0,1,2], target_names=["Grade 0", "Grade 1", "Grade 2"], zero_division=0))

# Save Best Model 
if best_model_state is not None:
    save_path = f"LSTM/models/best_model_epoch{best_epoch}_seed{SEED}.pt"
    torch.save(best_model_state, save_path)
    print(f"\n Saved best model to {save_path}")
