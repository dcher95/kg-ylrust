import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.nn import Module

def single_pass_clf(dataloader: DataLoader, model: Module, device: torch.device) -> None:
    """
    Perform a single forward pass with a classification model on a single image from a DataLoader.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader containing the dataset.
        model (Module): A PyTorch model for classification.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        None
    """
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(dataloader))

    # 2. Get a single image from the batch and its label
    img_single, label_single = img_batch[0], label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # Remove batch dimension and permute to (H, W, C) for visualization
    img_single = img_single.squeeze(0)  # Remove batch dimension
    img_single_for_vis = img_single.permute(1, 2, 0)  # Convert to (H, W, C)

    # Normalize the image to [0, 1] for visualization
    img_single_for_vis = (img_single_for_vis - img_single_for_vis.min()) / (img_single_for_vis.max() - img_single_for_vis.min())

    # Add batch dimension back for prediction
    img_single = img_single.unsqueeze(0)  # Add batch dimension

    # 3. Perform a forward pass on the single image
    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))

    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    pred_probs = torch.softmax(pred, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{pred_probs}\n")
    print(f"Output prediction label:\n{pred_label.item()}\n")
    print(f"Actual label:\n{label_single.item()}")

    # 5. Plot the image and its label
    plt.figure(figsize=(8, 4))

    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img_single_for_vis)
    plt.title(f"Actual: {label_single.item()}")
    plt.axis('off')

    # Plot the predicted probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(len(pred_probs[0])), pred_probs[0].cpu().numpy())
    plt.title(f"Predicted: {pred_label.item()}")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()
