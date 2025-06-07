from fastapi import FastAPI
from config import set_seed, seed
from model import CNNmodel
from data import load_cifar10, load_cifar10_test, Dirichlet_partition, create_client_datasets
from train import train_fedavg
from pydantic import BaseModel
from fastapi import UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
from inference import predict,load_model_checkpoint
app = FastAPI()

class TrainRequest(BaseModel):
    output_dim: int = 10
    num_rounds: int = 10
    fraction: float = 0.6
    lr: float = 0.03
    momentum: float = 0.9
    epochs: int = 7
    num_clients: int = 5
    alpha: float = 0.5
    batch_size: int = 64 

@app.get("/")
def root():
    return {"message": "Hello, FastAPI is running!"}

@app.post("/train")
def train_model(req: TrainRequest):
    set_seed(seed)
    
    trainset, _ = load_cifar10()
    testset, testloader = load_cifar10_test(req.batch_size)
    
    client_indices = Dirichlet_partition(trainset, req.num_clients, req.alpha)
    client_datasets = create_client_datasets(trainset, client_indices)

    best_acc, _ = train_fedavg(
        client_datasets=client_datasets,
        testloader=testloader,
        num_rounds=req.num_rounds,
        fraction=req.fraction,
        lr=req.lr,
        momentum=req.momentum,
        epochs=req.epochs,
        num_clients=req.num_clients,
        model=CNNmodel,
        output_dim=req.output_dim
    )

    return {"message": "Training complete", "best_accuracy": best_acc}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:

        model = load_model_checkpoint(
        model_class=CNNmodel,
        checkpoint_path="best_global_model.pth",
        output_dim=10
)
        # Convert uploaded file to PIL Image
        image = Image.open(file.file).convert("RGB")
        
        # Pass model and image to predict function
        result = predict(model, image)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}