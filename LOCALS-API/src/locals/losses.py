import torch

def locals_loss(beta=1.0, gamma=5.0):  
    '''
    beta: weight for localization loss
    gamma: weight for objectness loss
    '''
    
    def binary_focal_loss(pred, target, alpha=0.25, gamma=2.0, eps=1e-8):
        '''
        pred: predicted probabilities (after sigmoid)
        target: ground truth (0 or 1)
        alpha: weighting factor for class imbalance
        gamma: focusing parameter for hard examples
        '''
        pred = pred.clamp(eps, 1.0 - eps)  # avoid log(0)
        pt = pred * target + (1 - pred) * (1 - target)
        loss = - alpha * (1 - pt) ** gamma * (target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
        return loss.mean()

    def focal_localization_loss(pred_coords, true_coords, mask, alpha=0.25, gamma=2.0, eps=1e-8):
        d = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=-1))
        pt = 1 - torch.sigmoid(20 * (d - 0.1))
        pt = pt.clamp(eps, 1.0 - eps)
        loss = - alpha * (1 - pt) ** gamma * mask * torch.log(pt)
        if mask.sum() == 0:
            return 0
        
        return (loss).sum() / mask.sum()
        
    # actual loss function
    def loss_func(predicted, true):
        # goal is to sum each loss for each prediction in each batch
        loc_loss = 0
        obj_loss = 0
        
        # iterate through each image in the batch
        for i in range(true.shape[0]):
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
        loc_loss /= true.shape[0]
        obj_loss /= true.shape[0]

        # then find total loss
        total_loss = beta * loc_loss + gamma * obj_loss
        return total_loss

    return loss_func