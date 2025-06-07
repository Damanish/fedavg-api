import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (same as your training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model_checkpoint(model_class, checkpoint_path, output_dim):
    """
    Load model from checkpoint file
    """
    model = model_class(output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def predict(model, image: Image.Image):
    """
    Takes model and PIL Image, processes it, and returns predictions and probabilities
    """
    try:
        # Transform the input image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Convert probabilities to list for JSON serialization
        probs_list = probabilities.cpu().numpy().tolist()[0]
        cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

        
        return {
            "prediction": predicted_class,
            "prediction_label": cifar10_classes[predicted_class],
            "probabilities": probs_list
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")