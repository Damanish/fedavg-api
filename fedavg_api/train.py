import copy
import torch
import random
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def client_update(model,dataloader,optimizer,loss_fn,epochs = 1):
    model.train()
    model.to(device)
    for _ in range(epochs):
        for x,y in dataloader:
            x,y =x.to(device),y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output,y)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_models(client_models):
    avg_model = copy.deepcopy(client_models[0])
    for key in avg_model.keys():
        for i in range(1,len(client_models)):
            avg_model[key] += client_models[i][key]
        avg_model[key] = torch.div(avg_model[key], len(client_models))

    return avg_model

def evaluate(model,dataloader):
    model.eval()
    model.to(device)
    correct,total = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct+= (pred==y).sum().item()
            total+= y.size(0)
    return 100.0*correct/total

def train_fedavg(client_datasets,testloader,num_rounds,fraction,lr,momentum,epochs,num_clients,model,output_dim):

    global_model = model(output_dim).to(device)
    best_acc = 0
    best_model_state = None
    for round in range(num_rounds):
        print(f"\n--- Round {round + 1} ---")

        selected_clients = random.sample(range(num_clients),int(fraction*num_clients))

        client_models=[]
        for client_id in selected_clients:
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr, momentum)
            dataloader = DataLoader(client_datasets[client_id],batch_size=64,shuffle=True)
            loss_fn=torch.nn.CrossEntropyLoss()

            updated_weights = client_update(local_model,dataloader,optimizer,loss_fn,epochs)
            client_models.append(updated_weights)

        new_global_weights = aggregate_models(client_models)
        global_model.load_state_dict(new_global_weights)

        test_acc = evaluate(global_model, testloader)
        print(f"Test Accuracy: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(global_model.state_dict())

    torch.save(best_model_state, "best_global_model.pth")
    print(f"Saved the best accuracy model. Best Accuracy : {best_acc:.2f}%")
    return best_acc , best_model_state
