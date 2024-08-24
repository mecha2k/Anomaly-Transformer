import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def association_discrepancy(series, prior, win_size=100):
    prior_d = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)
    prior_d = prior_d.repeat(1, 1, 1, win_size)
    series1 = my_kl_loss(series, (prior / prior_d).detach())
    series2 = my_kl_loss((prior / prior_d).detach(), series)
    priors1 = my_kl_loss((prior / prior_d), series.detach())
    priors2 = my_kl_loss(series.detach(), (prior / prior_d))
    return torch.mean(series1 + series2), torch.mean(priors1 + priors2)


def association_discrepancy_t(series, prior, win_size=100, temperature=50):
    prior_d = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)
    prior_d = prior_d.repeat(1, 1, 1, win_size)
    series1 = my_kl_loss(series, (prior / prior_d).detach()) * temperature
    priors1 = my_kl_loss((prior / prior_d), series.detach()) * temperature
    series1 = (
        my_kl_loss(
            series,
            (
                prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, win_size)
            ).detach(),
        )
        * temperature
    )
    priors1 = (
        my_kl_loss(
            (prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
            series.detach(),
        )
        * temperature
    )
    return series1, priors1


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + "_checkpoint.pth"))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.model = None
        self.optimizer = None

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="train",
            dataset=self.dataset,
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="val",
            dataset=self.dataset,
        )
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="test",
            dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="thre",
            dataset=self.dataset,
        )

        print("train_loader: ", len(self.train_loader))
        print("vali_loader: ", len(self.vali_loader))
        print("test_loader: ", len(self.test_loader))
        print("thre_loader: ", len(self.thre_loader))
        print("train_dataset length: ", len(self.train_loader.dataset))
        print("vali_dataset length: ", len(self.vali_loader.dataset))
        print("test_dataset length: ", len(self.test_loader.dataset))
        print("thre_dataset length: ", len(self.thre_loader.dataset))

        x, y = next(iter(self.train_loader))
        print("train_loader data shape: ", x.shape, y.shape)

        self.build_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else device)
        print(f"{self.device} is available in torch")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)

            series_loss = 0.0
            priors_loss = 0.0
            for u in range(len(prior)):
                s_loss, p_loss = association_discrepancy(series[u], prior[u], self.win_size)
                series_loss += s_loss
                priors_loss += p_loss
            series_loss = series_loss / len(prior)
            priors_loss = priors_loss / len(prior)

            rec_loss = self.criterion(output, inputs)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * priors_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                inputs = input_data.float().to(self.device)

                output, series, prior, _ = self.model(inputs)

                # calculate Association discrepancy
                series_loss = 0.0
                priors_loss = 0.0
                for u in range(len(prior)):
                    s_loss, p_loss = association_discrepancy(series[u], prior[u], self.win_size)
                    series_loss += s_loss
                    priors_loss += p_loss
                series_loss = series_loss / len(prior)
                priors_loss = priors_loss / len(prior)

                rec_loss = self.criterion(output, inputs)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * priors_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + "_checkpoint.pth")
            )
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction="none")

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)
            loss = torch.mean(criterion(inputs, output), dim=-1)

            # calculate Association discrepancy
            series_loss = 0.0
            priors_loss = 0.0
            for u in range(len(prior)):
                s_loss, p_loss = association_discrepancy_t(
                    series[u], prior[u], self.win_size, temperature=temperature
                )
                series_loss += s_loss
                priors_loss += p_loss

            metric = torch.softmax((-series_loss - priors_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)
            loss = torch.mean(criterion(inputs, output), dim=-1)

            series_loss = 0.0
            priors_loss = 0.0
            for u in range(len(prior)):
                s_loss, p_loss = association_discrepancy_t(
                    series[u], prior[u], self.win_size, temperature=temperature
                )
                series_loss += s_loss
                priors_loss += p_loss

            metric = torch.softmax((-series_loss - priors_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)
            loss = torch.mean(criterion(inputs, output), dim=-1)

            series_loss = 0.0
            priors_loss = 0.0
            for u in range(len(prior)):
                s_loss, p_loss = association_discrepancy_t(
                    series[u], prior[u], self.win_size, temperature=temperature
                )
                series_loss += s_loss
                priors_loss += p_loss

            metric = torch.softmax((-series_loss - priors_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        preds = (test_energy > threshold).astype(int)
        labels = test_labels.astype(int)

        print("preds :   ", preds.shape)
        print("labels:   ", labels.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(labels)):
            if labels[i] == 1 and preds[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if labels[j] == 0:
                        break
                    else:
                        if preds[j] == 0:
                            preds[j] = 1
                for j in range(i, len(labels)):
                    if labels[j] == 0:
                        break
                    else:
                        if preds[j] == 0:
                            preds[j] = 1
            elif labels[i] == 0:
                anomaly_state = False
            if anomaly_state:
                preds[i] = 1

        preds = np.array(preds)
        labels = np.array(labels)
        print("preds :   ", preds.shape)
        print("labels:   ", labels.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(labels, preds)
        precision, recall, f_score, support = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        return accuracy, precision, recall, f_score
