import torch
import numpy as np
from torch.utils.data import DataLoader

from .math import pearson_corr
from .constants import NUM_GRID_CELLS

def find_recall_precision_f1_score(model, data, threshold=0.5, num_batches=100):
    if not isinstance(data, DataLoader):
        data = DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )
    model.eval()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data):
            if batch_idx >= num_batches:
                break
            
            images_tensor = images.to(model.device)
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

def find_mAP(model, data, class_threshold=0.5):
    if not isinstance(data, DataLoader):
        data = DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )
    model.eval()
    
    dist_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    APs = []
    
    for dist_threshold in dist_thresholds:
        all_detections = []
        num_trues = 0
        for batched_images, batched_labels in data:
            batched_images = batched_images.to(model.device)
            batched_predictions = model(batched_images).detach().cpu()
            batch_size = batched_images.shape[0]
            
            for b in range(batch_size):
                predictions = batched_predictions[b]
                labels = batched_labels[b]
                
                predictions = predictions.reshape(-1, predictions.shape[-1])
                trues = labels.reshape(-1, labels.shape[-1])
                taken_trues = [False] * trues.shape[0]

                num_trues += torch.sum(trues[:, 2] == 1).item()
                indices = torch.arange(predictions.shape[0])
                order = predictions[:, -1].argsort(descending=True)
                predictions = predictions[order]
                indices = indices[order]
                
                for i in range(predictions.shape[0]):
                    pred_xnb, pred_ynb, pred_conf = predictions[i]
                    cell_idx = indices[i]
                    row = cell_idx // NUM_GRID_CELLS
                    col = cell_idx % NUM_GRID_CELLS
                    pred_x = ((col / NUM_GRID_CELLS) + (pred_xnb * (1 / NUM_GRID_CELLS)))
                    pred_y = ((row / NUM_GRID_CELLS) + (pred_ynb * (1 / NUM_GRID_CELLS)))
                    
                    if pred_conf >= class_threshold:
                        closest_idx = -1
                        closest_distance = torch.inf
                        for j in range(trues.shape[0]):
                            true_xnb, true_ynb, true_conf = trues[j]
                            row = j // NUM_GRID_CELLS
                            col = j % NUM_GRID_CELLS
                            true_x = ((col / NUM_GRID_CELLS) + (true_xnb * (1 / NUM_GRID_CELLS)))
                            true_y = ((row / NUM_GRID_CELLS) + (true_ynb * (1 / NUM_GRID_CELLS)))
                            
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

def find_mCS(model, data, threshold=0.5, num_batches=100):
    if not isinstance(data, DataLoader):
        data = DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )
    model.eval()
    
    correlations = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data):
            if batch_idx >= num_batches:
                break

            images_tensor = images.to(model.device)

            predictions_batch = model(images_tensor)
            batch_size = images_tensor.shape[0]

            for i in range(batch_size):
                prediction = predictions_batch[i]
                prediction = prediction.cpu().numpy()

                label = labels[i]

                # extract predicted points
                predicted_points = []
                for row in range(prediction.shape[0]):
                    for col in range(prediction.shape[1]):
                        cell = prediction[row][col]
                        if cell[-1] > threshold:
                            xnb, ynb, c = cell
                            xn = ((col / NUM_GRID_CELLS) + (xnb * (1 / NUM_GRID_CELLS)))
                            yn = ((row / NUM_GRID_CELLS) + (ynb * (1 / NUM_GRID_CELLS)))
                            predicted_points.append([xn, yn])

                # extract label points
                label_points = []
                for row in range(label.shape[0]):
                    for col in range(label.shape[1]):
                        cell = label[row][col]
                        if cell[-1] > 0:
                            xnb, ynb, c = cell
                            xn = ((col / NUM_GRID_CELLS) + (xnb * (1 / NUM_GRID_CELLS)))
                            yn = ((row / NUM_GRID_CELLS) + (ynb * (1 / NUM_GRID_CELLS)))
                            label_points.append([xn, yn])

                if not predicted_points:
                    correlations.append(0)
                else:
                    correlations.append(abs(pearson_corr(label_points + predicted_points)))

    return np.mean(correlations) if correlations else 0.0