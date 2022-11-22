import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from aijack.collaborative import FedAvgClient, FedAvgServer
from aijack.utils import NumpyDataset
import pickle
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

batch_size = 32 # batch size of data loaders of clients
height = 19 # dimension of data
width = 52  # dimension of data

class Dnn_Network(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        #self.conv2 = nn.Conv2d(16, 32, 4, 2)
        #self.fc1 = nn.Linear(32 * 2 * 10, 32)
        #self.fc2 = nn.Linear(32, 4)
        self.dense1 = nn.Linear(width*height, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.pred_layer = nn.Linear(128, 4)

        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.reshape(x, (len(x), x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.dense1(x)
        x = self.dense_relu1(x)

        x = self.dense2(x)
        x = self.dense_relu2(x)
        preds = self.pred_layer(x)
        return preds

    def name(self):
        return "Dnn_Network"

#load the data and some preprocessing
def dataset_discription(fold, dataset="crema"):
    train_dict = {}
    test_dict = {}
    if dataset == "crema":
        fold_path = "dataset path ..."
    elif dataset == "iemocap":
        fold_path = "dataset path ..."
    elif dataset == "mel":
        fold_path = "dataset path ..."

    with open(fold_path + "fold" + str(fold) + "/" + "training_znorm.pkl", 'rb') as f:
        train_dict = pickle.load(f)
    f.close()
    with open(fold_path + "fold" + str(fold) + "/" + "test_znorm.pkl", 'rb') as f:
        test_dict = pickle.load(f)
    f.close()

    speakers = set([])
    label_info = {}
    test_label_info = {}


    c = 0
    for d in train_dict:
        if c%50 == 0:
            print(f"Train Dict: {c}")
        c += 1
        speakerID = train_dict[d]['speaker_id']
        label = train_dict[d]["label"]
        data = train_dict[d]['data']
        if train_dict[d]['speaker_id'] in label_info:
            label_code = 0
            if label == "neu":
                label_info[speakerID]['neu'] += 1
                #label_code = 0
            elif label == "ang":
                label_info[speakerID]['ang'] += 1
                #label_code = 1
            elif label == "hap":
                label_info[speakerID]['hap'] += 1
                #label_code = 2
            elif label == "sad":
                label_info[speakerID]['sad'] += 1
                #label_code = 3

            label_info[speakerID]['SUM'] += 1
            label_info[speakerID]['data'].append(data)
            label_info[speakerID]['label'].append(label)
        else:
            label_info[speakerID] = {'ang': 0, 'sad': 0, 'hap': 0, 'neu': 0, 'SUM': 0, 'data' : [], 'label': []}
            label_code = 0
            if label == "neu":
                label_info[speakerID]['neu'] += 1
                label_code = 0
            elif label == "ang":
                label_info[speakerID]['ang'] += 1
                label_code = 1
            elif label == "hap":
                label_info[speakerID]['hap'] += 1
                label_code = 2
            elif label == "sad":
                label_info[speakerID]['sad'] += 1
                label_code = 3

            label_info[speakerID]['SUM'] += 1
            label_info[speakerID]['data'].append(data)
            label_info[speakerID]['label'].append(label)

    c = 0
    for d in test_dict:
        if c%50 == 0:
            print(f"Test Dict: {c}")
        c += 1
        speakerID = test_dict[d]['speaker_id']
        label = test_dict[d]["label"]
        data = test_dict[d]['data']
        if test_dict[d]['speaker_id'] in test_label_info:
            label_code = 0
            if label == "neu":
                test_label_info[speakerID]['neu'] += 1
                label_code = 0
            elif label == "ang":
                test_label_info[speakerID]['ang'] += 1
                label_code = 1
            elif label == "hap":
                test_label_info[speakerID]['hap'] += 1
                label_code = 2
            elif label == "sad":
                test_label_info[speakerID]['sad'] += 1
                label_code = 3

            test_label_info[speakerID]['SUM'] += 1
            test_label_info[speakerID]['data'].append(data)
            test_label_info[speakerID]['label'].append(label)
        else:
            test_label_info[speakerID] = {'ang': 0, 'sad': 0, 'hap': 0, 'neu': 0, 'SUM': 0, 'data': [], 'label': []}
            label_code = 0
            if label == "neu":
                test_label_info[speakerID]['neu'] += 1
                label_code = 0
            elif label == "ang":
                test_label_info[speakerID]['ang'] += 1
                label_code = 1
            elif label == "hap":
                test_label_info[speakerID]['hap'] += 1
                label_code = 2
            elif label == "sad":
                test_label_info[speakerID]['sad'] += 1
                label_code = 3

            test_label_info[speakerID]['SUM'] += 1
            test_label_info[speakerID]['data'].append(data)
            test_label_info[speakerID]['label'].append(label)

    #print(label_info)

    return label_info, test_label_info

#find clients that have most distributed data w.r.t labels
def check_distributed_label(data):
    if data['neu'] > 10 and data['ang'] > 10 and data['hap'] > 10 and data['sad'] > 10:
        return True
    return False

# prepare data loaders for each client - prepare data loader for server(for test purpose only)
def prepare_dataloaders_per_speaker(i, features = "crema"):
    train_clients_data, test_clients_data = dataset_discription(i, features) # load data
    label_dataset = ['neu', 'sad', 'ang', 'hap']
    le = LabelEncoder()
    le.fit(label_dataset)

    train_loaders = []
    trainsets = []
    num_data_per_client = []
    distributed_clients = []

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    num_client = len(list(train_clients_data.keys()))

    #prepare data loaders of clients
    for cl in train_clients_data:
        cl1_X = np.array(train_clients_data[cl]['data'])
        cl1_X = cl1_X.reshape(len(cl1_X), height, width)
        transformed = le.transform(train_clients_data[cl]['label'])
        cl1_Y = np.array(transformed)
        trainset_1 = NumpyDataset(cl1_X, cl1_Y, transform=transform)
        trainloader_1 = torch.utils.data.DataLoader(
            trainset_1, batch_size=batch_size, shuffle=True, num_workers=0
        )
        train_loaders.append(trainloader_1)
        trainsets.append(trainset_1)
        num_data_per_client.append(len(cl1_X))

        if check_distributed_label(train_clients_data[cl]):
            #save id of client
            distributed_clients.append(len(num_data_per_client)-1)


    x_test = []
    y_test = []
    for cl in test_clients_data:
        cl1_X = np.array(test_clients_data[cl]['data'])
        transformed = le.transform(test_clients_data[cl]['label'])
        cl1_Y = np.array(transformed)
        for x,y in zip(cl1_X, cl1_Y):
            x = x.reshape(height, width)
            x_test.append(x)
            y_test.append(y)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # this dataset is accessible only by server and not used for training either by server or clients
    global_testset = NumpyDataset(
        x_test,
        y_test,
        transform=transform,
    )

    global_testloader = torch.utils.data.DataLoader(
        global_testset, batch_size=1000, shuffle=True, num_workers=0
    )

    return x_test, y_test, train_loaders, global_testset, global_testloader, trainsets, num_client, num_data_per_client, distributed_clients

def save_csv(arr, name, save_path):
    dict = {name: arr}
    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv(save_path.joinpath(name + ".csv"))

def find_first_available_index(arr):
    for i in range(len(arr)):
        if arr[i] == False:
            return i
    return -1

def sort_clients_based_on_num_data(all_clients, num_data_per_client):
    sorted = []
    flag = [False for _ in range(len(all_clients))]
    # sorted clients based on number of data
    selected_index = find_first_available_index(flag)
    while(selected_index != -1):
        if selected_index == -1:
            return sorted
        for j in range(0, len(all_clients)):
            if num_data_per_client[selected_index] < num_data_per_client[j] and flag[j] == False:
                selected_index = j
        sorted.append(selected_index)
        flag[selected_index] = True
        selected_index = find_first_available_index(flag)
    return sorted

def federated(distribution="equal", modelType="non-DP", features="crema"):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(device)
    best_score = 0
    best_fold = 1
    for fold in range(3, 4):
        X_test, y_test, trainloaders, global_trainloader, global_trainset, \
        trainsets, client_num, num_data_per_client, \
        distributed_clients = prepare_dataloaders_per_speaker(fold, features= features)
        num_client = client_num
        save_path = Path("./").joinpath("results", "_dist_" + distribution+ "_" + modelType+ "_clientNum" +   str(client_num)+ "_features_" +  features)
        model_path = save_path.joinpath("fold" + str(fold))
        Path.mkdir(model_path, parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        clients = []
        optimizers = []

        for cl in range(client_num):
            net_1 = Dnn_Network()
            client_1 = FedAvgClient(net_1, user_id=cl, send_gradient=False)
            client_1.to(device)
            optimizer_1 = optim.Adam(
                client_1.parameters(), lr=0.001
            )
            clients.append(client_1)
            optimizers.append(optimizer_1)

        global_model = Dnn_Network()
        global_model.to(device)
        # initialization of clients
        server = FedAvgServer(clients, global_model)

        loss_dict = {}
        for d in range(client_num):
            loss_dict[d] = []

        accuracies = []
        recalls = []
        f1s = []
        presicions = []
        losses = []
        sample_list_client = [i for i in range(num_client)]
        local_epochs = 1
        for epoch in range(200):
            # clients training
            for client_idx in sample_list_client:
                client = clients[client_idx]
                trainloader = trainloaders[client_idx]
                optimizer = optimizers[client_idx]

                running_loss = 0.0
                i=0
                for _ in range(local_epochs):
                    for _, data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = client(inputs)
                        loss = criterion(outputs, labels.to(torch.int64))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        i+=1
                    loss_save = running_loss / i
                    '''print(
                        f"epoch {epoch}: client-{client_idx+1}",
                         loss_save
                    )'''
                    if i != 0:
                        loss_dict[client_idx].append(loss_save)
                    else:
                        loss_dict[client_idx].append(0.0)

            if epoch%49 == 0 or epoch==10:
                torch.save(server.clients[sample_list_client[0]].state_dict(), model_path.joinpath("model_client_E_"+str(epoch)+"_cl_"+str(sample_list_client[0])+".pt"))

            # aggregation and distributing aggregated model to clients
            # use_gradients=False means FedAvg
            # use_gradients=True means FedSGD
            server.update(use_gradients=False)
            server.distribtue()

            in_preds = []
            in_label = []
            valid_loss = 0.0
            with torch.no_grad():
                for data in global_trainloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = server.server_model(inputs)
                    loss = criterion(outputs, labels.to(torch.int64))
                    valid_loss += loss.item()
                    in_preds.append(outputs)
                    in_label.append(labels)
                in_preds = torch.cat(in_preds)
                in_label = torch.cat(in_label).cpu()
            predicted = np.array(torch.argmax(in_preds, axis=1).cpu())
            acc = accuracy_score(predicted, np.array(in_label))
            recall = recall_score(predicted, np.array(in_label), average="weighted")
            f1 = f1_score(predicted, np.array(in_label), average="weighted")
            precision = precision_score(predicted, np.array(in_label), average="weighted")
            losses.append(valid_loss / len(global_trainloader))
            print(f"epoch {epoch}: accuracy is ", acc)
            accuracies.append(acc)
            recalls.append(recall)
            f1s.append(f1)
            presicions.append(precision)

            score = accuracy_score(np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label))
            if score >= best_score and epoch>50:
                best_score = score
                best_fold = fold
                torch.save(server.server_model.state_dict(), model_path.joinpath("server_model.pt"))
                torch.save(server.clients[0].state_dict(), model_path.joinpath("model_client.pt"))
                print("BEST SCORE: ", best_score)
        with open(model_path.joinpath("loss.json"), 'wb') as fp:
            pickle.dump(loss_dict, fp)
        save_csv(accuracies, "Acc", model_path)
        save_csv(recalls, "Recall", model_path)
        save_csv(f1s, "F1", model_path)
        save_csv(presicions, "Precision", model_path)
        save_csv(losses, "Val_loss", model_path)
    print("BEST SCORE: ", best_score)
    print("BEST FOLD: ", best_fold)
    dict1 = {"BEST FOLD":best_fold, "BEST SCORE":best_score}
    file1 = open(model_path.joinpath("best.txt"), "w")
    str1 = repr(dict1)
    file1.write("dict1 = " + str1 + "\n")
    file1.close()

if __name__ == '__main__':
    federated(distribution="equal", modelType="non-DP-FedAvg", features="crema")