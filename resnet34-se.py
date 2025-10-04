import os, math, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report # <-- WAJIB: Tambahkan ini

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ==========================================================
# 1. KONFIGURASI DAN DATA LOADING
# ==========================================================
class FoodDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {
            "gado_gado": 0, "bakso": 1, "rendang": 2,
            "nasi_goreng": 3, "soto_ayam": 4
        }
        self.target_names = list(self.label_map.keys()) # Untuk classification report
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.label_map[self.data.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label

# === Hyperparameters & Setup ===
IMG_ROOT = "train" 
CSV_PATH = "train.csv"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FoodDataset(csv_file=CSV_PATH, root_dir=IMG_ROOT, transform=transform)

generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

BATCH_SIZE = 32   
EPOCHS = 10       
LR = 1e-3         
OPTIM = "adam"    # Pastikan 'adam' atau 'sgd'

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==========================================================
# 2. DEFINISI MODEL (RESNET DENGAN SE)
# ==========================================================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# --- Squeeze-and-Excitation (SE) Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- BasicBlock DENGAN SE ---
class BasicBlockSE(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)

        out = self.se(out)   # <-- Apply SE

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# --- BasicBlock STANDAR (Diperlukan jika Anda ingin melatihnya di file lain) ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


# --- Kelas ResNet Utama ---
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet34_se(num_classes=5):
    """Builder untuk ResNet-34 DENGAN SE Block"""
    return ResNet(BasicBlockSE, [3,4,6,3], num_classes=num_classes)

# ==========================================================
# 3. UTILITY TRAINING DAN EVALUASI
# ==========================================================
criterion = nn.CrossEntropyLoss()

def run_epoch(model, loader, train=True, optimizer=None): # optimizer menjadi argumen
    if train:
        model.train()
    else:
        model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return loss, acc

def get_classification_report(model, loader, device, target_names):
    """Menghasilkan classification report pada validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # zero_division=0 ditambahkan untuk menghindari warning jika kelas tidak diprediksi
    return classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)


# ==========================================================
# 4. EKSEKUSI TRAINING, SAVING, DAN PLOTTING
# ==========================================================
if __name__ == "__main__":
    
    model = resnet34_se(num_classes=5).to(device)
    
    # Setup optimizer
    if OPTIM.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif OPTIM.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=False)
    else:
        raise ValueError("OPTIM harus 'adam' atau 'sgd'.")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print(f"\n{'='*10} Training ResNet-34 SE {'='*10}")

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, train=True, optimizer=optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, train=False, optimizer=optimizer)
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc {tr_acc:.2f}% | Val Acc {va_acc:.2f}%")

    print(f"🎉 Training ResNet-34 SE selesai!")
    
    # --- Classification Report ---
    report = get_classification_report(model, val_loader, device, dataset.target_names)
    print(f"\n=== Classification Report for ResNet-34 SE ===\n{report}")
    
    # --- Saving ---
    MODEL_NAME = "resnet34-se" # Nama dasar untuk file
    
    # Simpan bobot model
    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")

    # Siapkan data meta
    meta_data = {
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "optimizer": OPTIM, 
        "split": "80/20", "seed": 42, "report": report # Tambahkan report ke meta
    }
    
    # Simpan history dan meta
    # Kita menggunakan format yang kompatibel dengan loader Anda (menyimpan meta secara eksplisit)
    np.savez(f"{MODEL_NAME}_history.npz",
             train_loss=np.array(history["train_loss"]),
             val_loss=np.array(history["val_loss"]),
             train_acc=np.array(history["train_acc"]),
             val_acc=np.array(history["val_acc"]),
             meta=np.array([json.dumps(meta_data)], dtype=object))
             
    print(f"Saved: {MODEL_NAME}.pth & {MODEL_NAME}_history.npz")

    # ==========================================================
    # 5. PLOTTING DAN PERBANDINGAN
    # ==========================================================
    
    # --- Fungsi Utilitas Pemuatan ---
    def load_npz_history_for_plot(file_path):
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            # Hanya muat metrik yang dibutuhkan untuk plot
            return {k: v.tolist() for k, v in data.items() if k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        return None

    def safe_last(h_dict, key):
        lst = h_dict.get(key, []) 
        return lst[-1] if lst else np.nan

    # --- Pengumpulan History ---
    all_histories = {"ResNet34-SE (Eksplorasi)": history}
    
    # Coba load ResNet-34 Standar (baseline tanpa SE) jika ada
    # Asumsi file baseline ResNet-34 standar adalah resnet34_history.npz
    hist_std_file = "resnet34_history.npz"
    hist_std = load_npz_history_for_plot(hist_std_file)
    if hist_std:
        all_histories["ResNet34 (Baseline)"] = hist_std
        
    # Coba load Plain34 jika ada
    hist_plain_file = "plain34_history.npz"
    hist_plain = load_npz_history_for_plot(hist_plain_file)
    if hist_plain:
        all_histories["Plain34"] = hist_plain

    print(f"\nTotal model yang berhasil dimuat untuk perbandingan: {len(all_histories)}")

    # --- Plotting Overlay (Loss dan Accuracy) ---
    if len(all_histories) > 1:
        
        # Plot Validation Loss
        plt.figure(figsize=(10, 6))
        for name, hist in all_histories.items():
            plt.plot(hist["val_loss"], label=f"{name} Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Validation Loss Comparison"); plt.legend(); plt.grid(True)
        plt.show()

        # Plot Validation Accuracy
        plt.figure(figsize=(10, 6))
        for name, hist in all_histories.items():
            plt.plot(hist["val_acc"], label=f"{name} Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation Accuracy Comparison"); plt.legend(); plt.grid(True)
        plt.show()
    
    # --- Tabel Ringkas (Epoch Terakhir) ---
    summary_data = { "Model": [], "Train Loss": [], "Val Loss": [], "Train Acc %": [], "Val Acc %": [] }
    
    # Urutan tampilan yang logis
    display_order_keys = [k for k in ["Plain34", "ResNet34 (Baseline)", "ResNet34-SE (Eksplorasi)"] if k in all_histories]

    for name in display_order_keys:
        hist = all_histories[name]
        summary_data["Model"].append(name)
        summary_data["Train Loss"].append(safe_last(hist, "train_loss"))
        summary_data["Val Loss"].append(safe_last(hist, "val_loss"))
        summary_data["Train Acc %"].append(safe_last(hist, "train_acc"))
        summary_data["Val Acc %"].append(safe_last(hist, "val_acc"))

    if summary_data["Model"]:
        df = pd.DataFrame(summary_data)
        df = df.fillna('N/A')
        df['Train Loss'] = df['Train Loss'].map(lambda x: f'{float(x):.4f}' if x != 'N/A' else x)
        df['Val Loss'] = df['Val Loss'].map(lambda x: f'{float(x):.4f}' if x != 'N/A' else x)
        df['Train Acc %'] = df['Train Acc %'].map(lambda x: f'{float(x):.2f}' if x != 'N/A' else x)
        df['Val Acc %'] = df['Val Acc %'].map(lambda x: f'{float(x):.2f}' if x != 'N/A' else x)

        print("\n=== Perbandingan Kinerja (Epoch Terakhir) ===")
        print(df.to_string(index=False))