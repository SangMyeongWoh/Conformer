from CustomDataSet import CocoCaption
from train_cococaption import PL_conformer_caption
import torch
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# data = CocoCaption(tokenizer=None)
# dataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
#
# model = PL_conformer_caption(trainer=None)
# iterator = iter(dataloader)
# imgs, texts = next(iterator)
# lm_outs, token_ids_list = model(imgs=imgs, texts=texts)
#
# loss = 0
# for token_ids, lm_out in zip(token_ids_list, lm_outs):
#     shift_logit = lm_out[:len(token_ids)-1, :].contiguous()
#     shift_label = torch.tensor(token_ids[1:])
#     # print("shift_logit: ", shift_logit.shape)
#     # print("shift_label: ", shift_label.shape)
#     loss += model.loss_function(shift_logit, shift_label)
# loss = loss / len(token_ids_list)
#
# print(loss)

list = [50256, 464, 257, 4314, 1241, 2823, 286, 281, 2607, 1989, 2523, 3354, 286, 257, 6915, 290, 3084, 11, 262, 7825, 276, 3005, 3150, 286, 617, 2607, 18791, 11, 290, 2166, 290, 3641, 11, 257, 279, 1018, 319, 257, 22647, 3526, 13, 220, 50256]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

val = tokenizer.convert_ids_to_tokens(list)
print(val)

# print(lm_out.shape)
# print(lm_out.size(-1))
# print(lm_out.view(-1, lm_out.size(-1)))
# print(labels)


