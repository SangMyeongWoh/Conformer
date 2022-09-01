import torchvision.datasets as dset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.optim as optim
from conformer import Conformer
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from CustomDataSet import CocoCaption
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import math
import sys

class PL_conformer_caption(pl.LightningModule):
    def __init__(self, trainer, args, conformer_path=None):
        super(PL_conformer_caption, self).__init__()
        self.args = args
        self.trainer = trainer
        self.conformer = Conformer(patch_size=16, channel_ratio=6, embed_dim=768,
                                   depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True)
        if conformer_path is not None:
            self.conformer.load_state_dict(torch.load(conformer_path))

        self.gpt2lmhead = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.epoch_now = 0


    def forward(self, imgs, texts):
        x, x_t = self.conformer(imgs, extract=True)
        #### x_t shape Batchsize / sequence_length / hidden_size

        token_ids_list = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)) for text in texts]

        device_now = 'cuda:' + str(x_t.get_device())

        stacked_tensor = None
        attention_masks = None
        for token_ids, conf_feature in zip(token_ids_list, x_t):
            text_embeds = self.gpt2lmhead.transformer.wte(torch.tensor(token_ids, device=device_now))
            # if 256 - (conf_feature.shape[0] + text_embeds.shape[0]) < 0:
            #     print("wtf")
            #     print(conf_feature.shape[0])
            #     print(text_embeds.shape[0])
            #     print(token_ids)
            #     print(len(token_ids))
            if (conf_feature.shape[0] + text_embeds.shape[0]) >= 256:
                cat_val = torch.cat([conf_feature, text_embeds])
                cat_val = cat_val[:256]
                attention_mask = torch.tensor([1] * 256, device=device_now)
            else:
                zero_pad = torch.zeros(256 - (conf_feature.shape[0] + text_embeds.shape[0]),
                                       conf_feature.shape[1],
                                       device=device_now)

                attention_mask = torch.tensor([1] * (conf_feature.shape[0] + text_embeds.shape[0]) +
                                              [0] * (256 - (conf_feature.shape[0] + text_embeds.shape[0])), device=device_now)

                cat_val = torch.cat([conf_feature, text_embeds, zero_pad])
            if stacked_tensor is None:
                attention_masks = attention_mask.unsqueeze(dim=0)
                stacked_tensor = cat_val.unsqueeze(dim=0)
            else:
                attention_masks = torch.cat([attention_masks, attention_mask.unsqueeze(dim=0)])
                stacked_tensor = torch.cat([stacked_tensor, cat_val.unsqueeze(dim=0)])

        out = self.gpt2lmhead.transformer(inputs_embeds=stacked_tensor, attention_mask=attention_masks)
        lm_outs = self.gpt2lmhead.lm_head(out[0][:, 197:, :])

        return lm_outs, token_ids_list, x, x_t

        # for caption in captions:
        #     token_ids_list = [50300]
        #     token_ids_list.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(caption)))
        #     token_ids_list.append(50300)



    def training_step(self, batch, batch_idx):
        if self.current_epoch != self.epoch_now:
            self.epoch_now = self.current_epoch
            self.trainer.save_checkpoint("custom_checkpoint/" +
                                        "epoch-" + str(self.current_epoch) + "_" +
                                        "global_step-" + str(self.global_step) + ".ckpt")
        imgs, texts = batch
        lm_outs, token_ids_list, x, x_t = self(imgs=imgs, texts=texts)
        loss = 0
        for token_ids, lm_out in zip(token_ids_list, lm_outs):
            if len(token_ids) >= 59:
                token_ids = token_ids[:59]
            shift_logit = lm_out[:len(token_ids)-1, :]
            shift_label = torch.tensor(token_ids[1:], device='cuda:' + str(shift_logit.get_device()))
            # print("shift_logit: ", shift_logit.shape)
            # print("shift_label: ", shift_label.shape)
            # print("shift_logits: ", shift_logit)
            # print("shift_label: ", shift_label)
            loss += self.loss_function(shift_logit, shift_label)
        #loss = loss/len(token_ids_list)
        # if loss < 2.6:
        #     print("x_t: ", x_t)
        #     print("x: ", x)
        if not math.isfinite(loss):
            print("texts: ", texts)
            print("x_t: ", x_t)
            print("x: ", x)
            print("shift_logits: ", shift_logit)
            print("shift_label: ", shift_label)
            print("loss is not finite.")
            sys.exit(1)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return [optimizer], [optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.max_epochs)]

class PL_conformer_caption_data(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4):
        super(PL_conformer_caption_data, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        self.train_set = CocoCaption(tokenizer=None, mode='train')
        self.test_set = CocoCaption(tokenizer=None, mode='test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )


def main():
    parser = argparse.ArgumentParser('conformer_caption', add_help=False)

    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument("--mode", type=str, required=True, help="train or test")
    parser.add_argument("--max_epochs", type=int, default=300, required=False, help="max_epochs")
    parser.add_argument("--num_workers", type=int, default=4, required=False, help="num of workers")
    parser.add_argument("--gpus", type=str, required=False, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ckp_path", type=str, required=False, help="ckp_path")

    args, unparsed = parser.parse_known_args()

    if args.mode == 'train':
        gpus = [int(val) for val in args.gpus.split(",")]
        logger = TensorBoardLogger(save_dir="logs", name="conformer")
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus, strategy="ddp", precision=16,
                             default_root_dir="/workspace/custom_checkpoint", logger=logger)
        model = PL_conformer_caption(trainer=trainer, args=args)
        data_module = PL_conformer_caption_data(batch_size=args.batch_size, num_workers=args.num_workers)
        data_module.setup()
        trainer.fit(model, data_module.train_dataloader())


if __name__ == "__main__":
    main()