import torch
import torch.utils.data
import torch.nn.functional as F
import torch.autograd as autograd

from fedavg.datasets import get_dataset


class Client(object):

    def __init__(self, conf, model, train_df):
        self.conf = conf

        self.local_model = model
        self.train_df = train_df
        self.train_dataset = get_dataset(conf, self.train_df)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True)

    def calculate_mean_data(self, mean_batch: int):
        data, label = [], []
        for X, y in self.train_dataset:
            data.append(X)
            label.append(torch.tensor(y))
        data = torch.stack(data, dim=0)
        label = torch.stack(label, dim=0)

        random_ids = torch.randperm(len(data))
        data, label = data[random_ids], label[random_ids]
        data = torch.split(data, mean_batch)
        label = torch.split(label, mean_batch)

        self.Xmean, self.ymean = [], []
        for d, l in zip(data, label):
            self.Xmean.append(torch.mean(d, dim=0))
            self.ymean.append(torch.mean(F.one_hot(l, num_classes=self.conf["num_classes"][self.conf["which_dataset"]]).to(dtype=torch.float32), dim=0))
        self.Xmean = torch.stack(self.Xmean, dim=0)
        self.ymean = torch.stack(self.ymean, dim=0)
        return self.Xmean, self.ymean

    def get_mean_data(self, Xg, Yg):
        self.Xg = Xg
        self.Yg = Yg

    def local_train(self, model, lamb):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],weight_decay=self.conf["weight_decay"])
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'], weight_decay=self.conf["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                inputX = (1 - lamb) * data
                inputX.requires_grad_()

                idg = torch.randint(len(self.Xg), (1, ))
                xg = self.Xg[idg:idg+1]
                yg = self.Yg[idg:idg+1]

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                    inputX = inputX.cuda()
                    xg = xg.cuda()
                    yg = yg.cuda()

                optimizer.zero_grad()
                feature, output = self.local_model(inputX)

                loss1 = (1 - lamb) * criterion(output, target)
                loss2 = lamb * criterion(output, yg.expand_as(output))

                gradients = autograd.grad(outputs=loss1, inputs=inputX,
                                          create_graph=True, retain_graph=True)[0]
                loss3 = lamb * torch.inner(gradients.flatten(start_dim=1), xg.flatten(start_dim=1))
                loss3 = torch.mean(loss3)

                loss = loss1 + loss2 + loss3
                loss.backward()

                optimizer.step()

            acc, loss = self.model_eval()
            print("Epoch {0} done. train_loss ={1}, train_acc={2}".format(e, loss, acc))

        return self.local_model.state_dict()

    @torch.no_grad()
    def model_eval(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = torch.nn.CrossEntropyLoss()
        for batch_id, batch in enumerate(self.train_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.local_model(data)

            total_loss += criterion(output, target)    # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size

        return acc, total_l
