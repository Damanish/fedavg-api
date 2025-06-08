# Federated Learning with FedAvg on CIFAR-10

A complete implementation of Federated Averaging (FedAvg) algorithm for distributed machine learning on the CIFAR-10 dataset. This project includes a FastAPI-based web service, Docker containerization, and end-to-end pipeline for federated learning.

## Features

- **Federated Averaging Implementation**: Complete FedAvg algorithm implementation
- **CIFAR-10 Dataset**: Image classification using the popular CIFAR-10 dataset
- **FastAPI Web Service**: RESTful API endpoints for training and prediction
- **Docker Support**: Containerized application for easy deployment
- **Modular Architecture**: Clean separation of concerns with multiple Python modules
- **End-to-End Pipeline**: From data preprocessing to model deployment

## What is Federated Averaging (FedAvg)?

**Federated Averaging (FedAvg)** is a foundational algorithm in **federated learning**, a machine learning paradigm where data remains **decentralized** across multiple devices or clients. Instead of collecting all the data at a central server, FedAvg allows models to be trained locally and only the **model updates** (not the raw data) are shared.

### Key Concepts:
- **Data Privacy**: Raw data never leaves client devices, which is important in sensitive domains like healthcare or finance.
- **Decentralized Training**: Each client trains a local model on its own data.
- **Model Averaging**: The server averages the local model updates to improve a global model.

### Why FedAvg?
FedAvg is efficient and simple, yet powerful. It reduces communication costs and enhances privacy, making it a go-to algorithm for federated learning applications.


## Project Structure

```
fedavg/
├── fedavg_api/
│   ├── main.py           # FastAPI entry point
│   ├── train.py          # Federated training logic
│   ├── data.py           # Dataset loading and partitioning
│   ├── model.py          # CNN model definition
│   ├── inference.py      # Inference API
│   ├── config.py         # Configurations
├── Dockerfile            # Docker image definition
├── requirements.txt      # Dependencies
├── fedavg.ipynb          # Jupyter notebook
├── .gitignore

```

---

## ⚙️ How to Run

You can run this project in **three different ways**:

---

### 1. Run via Jupyter Notebook (for experimentation)

```bash
pip install -r requirements.txt
jupyter notebook fedavg.ipynb
```
Walks through local training, aggregation, and evaluation in an interactive manner.

Ideal for learning and visualization.

### 2. Run via FastAPI with Uvicorn (for API serving)
```bash
pip install -r requirements.txt
uvicorn fedavg_api.main:app --host 0.0.0.0 --port 8000
```

Launches a REST API server for training and inference.

Swagger Docs: http://localhost:8000/docs

### 3. Run via Docker (for deployment)
```bash
# Build the image
docker build -t fedavg-api .

# Run the container
docker run -p 8000:8000 --name fedavg_container fedavg-api
```
API is served at http://localhost:8000/docs

No Python environment setup needed on host machine.

##If you don’t want to clone or set anything up manually, just use the prebuilt Docker image:
```bash
docker run -p 8000:8000 damanish/fedavg-api
```

### Training Endpoint
- **URL**: `POST /train`
- **Description**: Initiates federated learning training process

### Prediction Endpoint
- **URL**: `POST /predict`
- **Description**: Makes predictions using the trained federated model
- **Parameters**: 
  - `image`: Input image for classification

## Dataset

This project uses the **CIFAR-10** dataset, which consists of:
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images and 10,000 test images

## Federated Learning Algorithm

The implementation uses **Federated Averaging (FedAvg)**, which:

1. **Initialization**: Server initializes a global model
2. **Client Selection**: Random subset of clients participate in each round
3. **Local Training**: Each client trains on local data
4. **Model Aggregation**: Server averages client model updates
5. **Global Update**: Updated global model is distributed to clients


## Requirements

```
fastapi
uvicorn[standard]
torch
torchvision
pillow
python-multipart
numpy
pydantic
```
Install all dependencies:
```bash
pip install -r requirements.txt
```

## Performance

The federated learning setup achieves:
- **Convergence**: Typically converges within 5-10 rounds
- **Accuracy**: Achieves ~75% accuracy on Cifar10 test set then number of clients = 5 and alpha(for dirichlet partitioning) = 0.5
- **Privacy**: Data remains distributed across clients

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- FedAvg algorithm by McMahan et al.
- CIFAR-10 dataset by Krizhevsky & Hinton
- FastAPI framework for the web service
- PyTorch for deep learning implementation

## Contact

Ripu Daman Singh Bankawat :- rdsbankawat@gmail.com

---
