import torch
import torch.nn as nn
import torch.nn.init as init

# ====================
# Convolutional Helpers
# ====================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution dengan padding=1"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# ====================
# ResNet Blocks
# ====================

# A: Standard Basic Block (Post-activation)
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
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# D: Pre-activation Block (ResNet V2 Style)
class PreActBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # downsample dari ResNet V2 selalu menerima output BN-ReLU
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = x
        
        if self.downsample is not None:
            # Downsample pada PreAct (V2) dilakukan pada input yang sudah diaktivasi (out)
            shortcut = self.downsample(out) 

        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        
        out += shortcut
        return out

# E: Kernel Modified Block (First convolution uses 5x5 kernel)
class KernelModifiedBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(KernelModifiedBlock, self).__init__()
        # Modifikasi: Conv pertama menggunakan kernel 5x5
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv kedua tetap 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


# ====================
# Main ResNet Architecture
# ====================

class ResNet(nn.Module):
    """
    Struktur ResNet-34 umum untuk semua varian.
    """
    def __init__(self, block, layers, num_classes=5):
        super().__init__()
        self.inplanes = 64
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers (Config ResNet-34: [3, 4, 6, 3])
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        # Inisialisasi Bobot
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsampling shortcut
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
    
    def _initialize_weights(self):
        """Standard ResNet weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ====================
# Model Builders (ResNet-34 Configuration)
# ====================

def resnet34(num_classes=5):
    """
    Standard ResNet-34 model (menggunakan BasicBlock).
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def preact_resnet34(num_classes=5):
    """
    Pre-activation ResNet-34 model (menggunakan PreActBasicBlock).
    """
    return ResNet(PreActBasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def kernel_resnet34(num_classes=5):
    """
    Modified ResNet-34 model dengan 5x5 first conv di blocks (menggunakan KernelModifiedBlock).
    """
    return ResNet(KernelModifiedBlock, [3, 4, 6, 3], num_classes=num_classes)

# ====================
# Example Usage
# ====================

if __name__ == '__main__':
    # Contoh: Membuat dan menguji model standar ResNet-34
    model_std = resnet34(num_classes=5)
    print("Standard ResNet-34 Model berhasil dibuat.")

    # Contoh: Mengecek output shape
    dummy_input = torch.randn(1, 3, 224, 224) 
    output = model_std(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [1, 5])")