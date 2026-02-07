import sys

if len(sys.argv) < 3:
    print("Usage: python make_seed.py <random_seed> <test_dataset_ratio>")
    sys.exit(1)
    
seed = int(sys.argv[1])
TEST_RATIO = float(sys.argv[2])

if TEST_RATIO <= 0 or TEST_RATIO >= 1:
    print("Test ratio must be in range (0, 1) (exclusive).")
    sys.exit(1)

# standard library
import os
import pickle

# third party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

batch_size = 1

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = nn.Sequential(
            # first block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LayerNorm([64, 240, 240]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # second block
            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([192, 60, 60]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # third block
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([256, 30, 30]),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 30, 30]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # fourth block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 15, 15]),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([1024, 8, 8]),

            # final block
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([1024, 4, 4])
        )

        # manually initialize all Conv2d layers' weights with He Normal (Kaiming Normal)
        for m in self.model:
            if isinstance(m, nn.Conv2d): # if layer is a conv layer
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # initialize weights with He Normal
                if m.bias is not None: # initialize bias with zeros
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
    
class ProtoModel(nn.Module):
    def __init__(self):
        super(ProtoModel, self).__init__()
        self.feature_extractor = FeatureExtractor()  # from earlier
        
        # the output from the extractor is [batch, 1024, 4, 4], flatten it
        self.flatten = nn.Flatten()

        # dense layer, output shape is [batch, 147]
        self.fc = nn.Linear(in_features=1024 * 4 * 4, out_features=7 * 7 * 3)

        # sigmoid activation for output normalization
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)             # [batch, 1024, 4, 4]
        x = self.flatten(x)                       # [batch, 1024*4*4]
        x = self.fc(x)                            # [batch, 147]
        x = self.activation(x)                    # [batch, 147]
        x = x.view(-1, 7, 7, 3)                   # Reshape to [batch, 7, 7, 3]
        return x

print("MODEL CLASS DEFINED")

def locals_loss(beta=5.0, gamma=5.0):  
    '''
    beta: weight for localization loss
    gamma: weight for objectness loss
    '''

    def binary_focal_loss(pred, target, alpha=0.25, gamma=2.0, eps=1e-8):
        pred = pred.clamp(eps, 1.0 - eps)  # avoid log(0)
        # compute p_t
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = - alpha_t * (1 - p_t) ** gamma * torch.log(p_t)
        return loss.mean()

    def focal_localization_loss(pred_coords, true_coords, mask, alpha=0.25, gamma=2.0, eps=1e-8):
        d = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=-1))
        pt = 1 - torch.sigmoid(20 * (d - 0.1))
        pt = pt.clamp(eps, 1.0 - eps)
        loss = - alpha * (1 - pt) ** gamma * mask * torch.log(pt)
        
        return (loss).sum() / mask.sum()
        
    # actual loss function
    def loss_func(predicted, true):
        # goal is to sum each loss for each prediction in each batch
        loc_loss = 0
        obj_loss = 0
        
        # iterate through each image in the batch
        for i in range(batch_size):
            ith_predicted = predicted[i]
            ith_true = true[i]

            obj_mask = ith_true[..., 2]
            true_coordinates = ith_true[..., :2]

            obj_pred = ith_predicted[..., 2]
            pred_coordinates = ith_predicted[..., :2]

            # find localization loss
            loc_loss += focal_localization_loss(pred_coordinates, true_coordinates, obj_mask)

            # find objectness loss
            ith_obj_loss = binary_focal_loss(obj_pred, obj_mask)
            obj_loss += ith_obj_loss

        # first find mean loss
        loc_loss /= batch_size
        obj_loss /= batch_size

        # then find total loss
        total_loss = beta * loc_loss + gamma * obj_loss
        return total_loss

    return loss_func

print("LOSS FUNCTION CLASS DEFINED")

with open(r"../LOCALS-Single-PKL-DATASET.pkl", "rb") as f:
    data = pickle.load(f)
    
class PKLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
    
dataset = PKLDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# constants
DATASET_SIZE = len(dataset)  # Should be 499
TRAIN_RATIO = 1 - TEST_RATIO

# lengths of splits
train_len = int(TRAIN_RATIO * DATASET_SIZE)
test_len = DATASET_SIZE - train_len

torch.manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print("TRAIN & TEST DATASET LOADED")

print("STARTING TRAINING")
device = 'cuda'

# model, optimizer, and custom loss
model = ProtoModel()
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-5)
criterion = locals_loss()

# training parameters
num_epochs = 100
loss_ot = [] # loss overtime

for epoch in range(num_epochs):
    # training on training dataset
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, targets in pbar:
        inputs = inputs.to(device).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Avg Training Loss = {avg_loss}")
    loss_ot.append(avg_loss)

print('TRAINING COMPLETED, SAVING AND LOADING MODEL')
torch.save(model.state_dict(), "model-single.pth")
model = ProtoModel()
model.to(device)
model.load_state_dict(torch.load("model-single.pth"))
model.eval()

# function that calculates the pearson correlation coefficient given a list of points
def pearson_corr(points):
    points_array = np.array(points)

    x = points_array[:, 0]
    y = points_array[:, 1]

    corr = np.corrcoef(x, y)[0, 1]
    return corr

# function that calculates the mCS value mentioned in the paper
def find_mCS(model, dataloader, threshold=0.5, num_batches=100):
    correlations = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images_tensor = images.permute(0, 3, 1, 2).to(device)

            predictions_batch = model(images_tensor)
            batch_size = images_tensor.shape[0]

            for i in range(batch_size):
                prediction = predictions_batch[i]
                prediction = prediction.cpu().numpy()

                label = labels[i]

                # extract predicted points
                predicted_points = []
                for row in prediction:
                    for cell in row:
                        if cell[-1] > threshold:
                            predicted_points.append(list(cell[:2]))

                # extract label points
                label_points = []
                for row in label:
                    for cell in row:
                        if cell[-1] > 0:
                            label_points.append(list(cell[:2]))

                if not predicted_points:
                    correlations.append(0)
                else:
                    correlations.append(abs(pearson_corr(label_points + predicted_points)))

    return np.mean(correlations) if correlations else 0.0

mCS = find_mCS(model, test_loader, num_batches=100)
print("mCS CALCULATED")

def find_recall_precision_f1_score(model, dataloader, threshold=0.5, num_batches=100):
    model.eval()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # ensure images are [batch (B), C, H, W] on the correct device
            images_tensor = images.permute(0, 3, 1, 2).float().to(device)
            
            predictions_batch = model(images_tensor) # [B, 7, 7, 3]
            batch_size = images_tensor.shape[0]
            
            for i in range(batch_size):
                prediction = predictions_batch[i]
                prediction = prediction.cpu().numpy()
                
                label = labels[i]
                label_numpy = label.cpu().numpy()
                
                # compare predictions vs labels
                for j in range(7):
                    for k in range(7):
                        pred_obj = prediction[j, k, -1]
                        label_obj = label_numpy[j, k, -1]
                        
                        if pred_obj > threshold and label_obj > 0:
                            true_positives += 1
                        elif pred_obj > threshold and label_obj == 0:
                            false_positives += 1
                        elif pred_obj <= threshold and label_obj > 0:
                            false_negatives += 1
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, precision, f1_score

recall, precision, f1 = find_recall_precision_f1_score(model, test_loader, num_batches=100)
print("RECALL, PRECISION & F1 SCORE CACLULATED")

def find_mAP_hard(dataloader, class_threshold, model):
    model.eval()
    all_confidence = []
    num_trues = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            images = images.permute(0, 3, 1, 2)
            predictions = model(images).detach().cpu()

            predictions = predictions.view(-1, predictions.shape[-1])
            trues = labels.view(-1, labels.shape[-1])

            num_trues += torch.sum(trues[:, 2] == 1).item()

            for i in range(predictions.shape[0]):
                    pred_x, pred_y, pred_class_conf = predictions[i]
                    true_x, true_y, true_class_conf = trues[i]
                    
                    if pred_class_conf >= class_threshold:
                        if true_class_conf == 1:
                            pred_coord = torch.tensor([pred_x, pred_y])
                            true_coord = torch.tensor([true_x, true_y])
            
                            d = torch.sqrt(torch.sum((pred_coord - true_coord) ** 2, dim=-1))
                            confidence = 1 - torch.sigmoid(69 * (d - 0.1))
                            all_confidence.append(confidence)
            
    dist_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    APs = []

    for dist_threshold in dist_thresholds:
        TP = []
        FP = []
        
        for confidence in all_confidence:
            if confidence >= dist_threshold:
                TP.append(1)
                FP.append(0)
            else:
                TP.append(0)
                FP.append(1)
    
        TP_cum = np.cumsum(TP)
        FP_cum = np.cumsum(FP)
        
        precisions = TP_cum / (TP_cum + FP_cum + 1e-6)
        recalls = TP_cum / num_trues
    
        AP = np.trapezoid(precisions, recalls)
        APs.append(AP)

    return sum(APs) / len(APs)

def find_mAP_hard2(dataloader, class_threshold, model):
    model.eval()
    dist_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    APs = []
    
    for dist_threshold in dist_thresholds:
        all_detections = []
        num_trues = 0
        for images, labels in dataloader:
            images = images.to(device)
            images = images.permute(0, 3, 1, 2)
            predictions = model(images).detach().cpu()

            # batch size is one, so no need to worry
            predictions = predictions.view(-1, predictions.shape[-1])
            trues = labels.view(-1, labels.shape[-1])
            taken_trues = [False] * trues.shape[0]

            num_trues += torch.sum(trues[:, 2] == 1).item()
            predictions = predictions[predictions[:, -1].argsort(descending=True)]
            
            for i in range(predictions.shape[0]):
                pred_x, pred_y, pred_conf = predictions[i]
                if pred_conf >= class_threshold:
                    closest_idx = -1
                    closest_distance = torch.inf
                    for j in range(trues.shape[0]):
                        true_x, true_y, true_conf = trues[j]
                        if true_conf >= 0.5:
                            pred_coord = torch.tensor([pred_x, pred_y])
                            true_coord = torch.tensor([true_x, true_y])
            
                            dist = torch.sqrt(torch.sum((pred_coord - true_coord) ** 2, dim=-1))
                            if dist < closest_distance:
                                closest_distance = dist
                                closest_idx = j
                    
                    dist_conf = 1 - torch.sigmoid(69 * (closest_distance - 0.1))
                    if closest_idx != -1 and not taken_trues[closest_idx] and dist_conf >= dist_threshold:
                        taken_trues[closest_idx] = True
                        all_detections.append((pred_conf.item(), 1, 0))
                    else:
                        all_detections.append((pred_conf.item(), 0, 1))
    
        all_detections.sort(key=lambda x: x[0], reverse=True)
        TP = [d[1] for d in all_detections]
        FP = [d[2] for d in all_detections]
        
        TP_cum = np.cumsum(TP)
        FP_cum = np.cumsum(FP)

        precisions = TP_cum / (TP_cum + FP_cum + 1e-6)
        recalls = TP_cum / (num_trues + 1e-6)
        AP = np.trapezoid(precisions, recalls)
        APs.append(AP)
    
    return sum(APs)/len(APs)

mAP1 = find_mAP_hard(test_loader, 0.5, model)
mAP2 = find_mAP_hard2(test_loader, 0.5, model)
print("mAP CALCULATED")

with open('seed_data.csv', 'a') as file:
    file.write(f'{mCS},{recall},{precision},{f1},{mAP1},{mAP2},{seed},{TEST_RATIO}\n')
print("DATA SAVED")