import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 稳定扩散的空条件表示没有任何文本提示。
# 这个函数加载预训练的 CLIP 模型和分词器，并生成空文本对应的嵌入向量。
def sd_null_condition(path):
    text = ""
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", revision=None)
    tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", revision=None)
    with torch.no_grad():
        empty_inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        emptyembed = text_encoder(empty_inputs.input_ids)[0]
    del text_encoder, tokenizer
    return emptyembed


if __name__ == '__main__':
    pass
