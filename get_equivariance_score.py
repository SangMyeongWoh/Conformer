import h5py
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import entropy
import argparse

# f1 = h5py.File('epoch-127_extra_test.hdf5', 'r')
# key_list = list(f1.keys())
# default = f1[key_list[0]][()]
# print(key_list[0:10])
# print(default.flatten().shape)

class PL_classifier(pl.LightningModule):
    def __init__(self, trainer):
        super(PL_classifier, self).__init__()
        self.trainer = trainer
        self.fc1 = torch.nn.Linear(151296, 151296 // 24)
        torch.nn.init.normal(self.fc1.weight, mean=0.0, std=1.0)
        self.fc1.requires_grad = True
        self.fc2 = torch.nn.Linear(151296 // 24, 2)
        torch.nn.init.normal(self.fc2.weight, mean=0.0, std=1.0)
        self.fc2.requires_grad = True
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.saved_result = {'img_key': [], 'label': [], 'logits_0': [], \
                             'logits_1': [], 'abs': [], 'entropy': [], 'correct': []}
        self.epoch_now = -1



    def forward(self, feature):
        feature = torch.nn.functional.normalize(feature)
        first = self.fc1(feature)
        logits = self.fc2(first)
        return logits

    def training_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0 and self.epoch_now != self.current_epoch:
            self.epoch_now = self.current_epoch
            self.trainer.save_checkpoint("custom_checkpoint/" + "classifier_" +
                                        "epoch-" + str(self.current_epoch) + "_" +
                                        "global_step-" + str(self.global_step) + ".ckpt")
        features, labels, img_keys = batch
        loss = self.loss_function(self(features), labels)
        return loss

    def test_step(self, batch, batch_idx):
        features, labels, img_keys = batch
        logits = F.softmax(self(features))
        logits_list = logits.tolist()
        _, predicted = logits.max(1)
        predicted = predicted.tolist()
        for pred, target, img_key, _logits in zip(predicted, labels, img_keys, logits_list):
            self.saved_result['img_key'].append(img_key)
            self.saved_result['label'].append(target.item())
            self.saved_result['logits_0'].append(_logits[0])
            self.saved_result['logits_1'].append(_logits[1])
            self.saved_result['entropy'].append(entropy(_logits, base=2))
            if pred != int(target.item()):
                self.saved_result['correct'].append("False")
            else:
                self.saved_result['correct'].append("True")


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

class PL_classifier_data(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4):
        super(PL_classifier_data, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        self.train_set = Features(data_dir='epoch-127_extra_train.hdf5')
        self.test_set = Features(data_dir='epoch-127_extra_test.hdf5')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

class Features(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # f1 = h5py.File(test_path, 'r')
        # key_list = list(f1.keys())
        # default = f1[key_list[0]][()]

        self.f1 = h5py.File(self.data_dir, 'r')
        self.key_list = list(self.f1.keys())

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, index):
        default = self.f1[self.key_list[index]][()]
        label = 1
        if "_F" in self.key_list[index]:
            label = 0

        return default.flatten(), label, self.key_list[index]


def main():
    # region parser
    parser = argparse.ArgumentParser('Vit', add_help=False)

    parser.add_argument('--batch_size', type=int, required=True, help="batch size for single GPU")

    # endregion
    parser.add_argument("--mode", type=str, required=True, help="train or test")
    parser.add_argument("--max_epochs", type=int, default=100, required=False, help="max_epochs")
    parser.add_argument("--gpus", type=str, required=False, default="0,1,2")
    # parser.add_argument("--ckp_path", type=str,
    #                     default="humanpuzzle_checkpoint/train_firsthalf/vertical_flip_offset_epoch-7_global_step-5061.ckpt",
    #                     required=False,
    #                     help="ckp_path")


    args, unparsed = parser.parse_known_args()

    if args.mode == 'train':

        gpus = [int(val) for val in args.gpus.split(",")]
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus, strategy="ddp", precision=16,
                             default_root_dir="/workspace/custom_checkpoint")
        model = PL_classifier(trainer)
        data_module = PL_classifier_data(batch_size=args.batch_size)
        data_module.setup()

        trainer.fit(model, data_module.train_dataloader())

    elif args.mode == 'test':
        print("??")
        gpus = [int(val) for val in args.gpus.split(",")]

        trainer = pl.Trainer(gpus=gpus, strategy="dp", precision=16)
        data_module = PL_classifier_data(batch_size=args.batch_size)
        data_module.setup()


        model = PL_classifier(trainer)
        state_dict = torch.load(args.ckp_path)['state_dict']
        model.load_state_dict(state_dict)
        trainer.test(model, data_module.test_dataloader())
        dataframe = pd.DataFrame(model.saved_result)
        dataframe.to_excel(os.path.join("extra_result.xlsx"))


if __name__ == "__main__":
    print("??")
    main()