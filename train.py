import transformers

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

from tqdm import tqdm

from torchvision.transforms import Compose, Resize, CenterCrop

from datasets import load_from_disk

import random


from PIL import Image

import matplotlib.pyplot as plt

from torch.optim import optimizer


from peft import get_peft_model, LoraConfig, TaskType

flickr = load_from_disk("flickr30k_dataset/")

split = flickr['test'].train_test_split(train_size=0.9, seed=42)

train_set = split['train']
val_set = split['test']

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
llama = transformers.AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B'
).cuda()

tokenizer.pad_token = tokenizer.eos_token 

CLIP_model_id = "openai/clip-vit-large-patch14-336"
CLIP = transformers.CLIPModel.from_pretrained(CLIP_model_id).cuda()
processor = transformers.CLIPProcessor.from_pretrained(CLIP_model_id)#

CLIP.eval()
for param in CLIP.parameters(): param.requires_grad = False

class flickr_dataset(Dataset):
    def __init__(self, data, tokenizer=tokenizer, processor=processor):
        self.images = data['image']
        self.caption = data['caption']
        self.tokenizer = tokenizer
        self.processor = processor
        self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_std = [0.26862954, 0.26130258, 0.27577711]

        self.preprocess = T.Compose([
            T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(336),
            T.ToTensor(),  # Converts to [0, 1] float and shape [C, H, W]
            T.Normalize(mean=self.clip_mean, std=self.clip_std)
    ])
        


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        caption = self.caption[index]

        # Tokenize text
        encoded = self.tokenizer(
            caption, #[random.randint(0,4)]
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )

        test = self.preprocess(img)

        # Process image
    #     processed = self.processor(images=img, return_tensors='pt')
    #     print ("==" * 20)
    #     print (test.shape)
    #     print (encoded)
    #     img_clamped = self.preprocess(img).clone().detach().clamp(0, 1)  # Keep normalization artifacts

    # # Convert to [H, W, 3] for display
    #     img_np = img_clamped.permute(1, 2, 0).cpu().numpy()

    #     plt.imshow(img_np)
    #     plt.axis("off")
    #     plt.title("CLIP-normalized image (clamped for display)")
    #     plt.show()
        return self.preprocess(img), encoded

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # Typical for LLaMA
)

llama_with_lora = get_peft_model(llama, lora_config)

class captioning(nn.Module):
    def __init__(self, CLIP, llama):
        super().__init__()
        self.CLIP = CLIP
        self.llama = llama
        self.mlp = nn.Sequential(
            nn.Linear(1024,1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024,2048),
            nn.LayerNorm(2048)
        )

    def forward(self, image, input_ids, attention_mask):
        
        # Encode image with CLIP
        with torch.no_grad():
                vision_outputs = CLIP.vision_model((image))
                image_embed = vision_outputs.last_hidden_state[:, 1:, :]

        # print ("image_embed", image_embed.shape)

        image_token = self.mlp(image_embed) # [B, 1, D]

        # Embed input_ids via LLaMA embedding layer
        input_embeds = self.llama.base_model.model.model.embed_tokens(input_ids)

        # print("image_token shape:", image_token.shape)
        # print("input_embeds shape:", input_embeds.shape)    

        assert image_token.ndim == 3
        assert input_embeds.ndim == 3
        # Concatenate image token and text tokens

        inputs_embeds = torch.cat([image_token, input_embeds], dim=1)

        # Adjust attention mask
        image_token_len = image_token.shape[1]

        extended_mask = torch.cat([
            torch.ones(image_token.shape[0], image_token_len, device=image_token.device),
            attention_mask
        ], dim=1)

        return self.llama(
            inputs_embeds=inputs_embeds
            #attention_mask=extended_mask
        )
    


# def collate_fn(batch):
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
#     input_ids = torch.stack([item["input_ids"] for item in batch])
#     attention_mask = torch.stack([item["attention_mask"] for item in batch])
#     return {
#         "pixel_values": pixel_values,
#         "input_ids": input_ids,
#         "attention_mask": attention_mask
#     }


def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    model.to(device)
    best_val_loss = float('inf')

    epoch_losses = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding

    for epoch in range(epochs):
        train_epoch_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for img, caption in pbar:
            # print ("img",img.shape)
            # print ("caption",caption)
            pixel_values = img.to(device)
            cap = caption.to(device)
            idx = random.randint(0,4)
            caps = (cap['input_ids'][:,idx,:]).to(device)

            mask = cap['attention_mask'][:,idx,:].to(device)

            # Shift labels for causal loss
            caps_in = caps[:, :-1]
            caps_out = caps[:, 1:]
            mask = mask[:, 1:]

            # print ("img", pixel_values.shape)
            # print ("caps_in", caps_in.shape)
            # print ("mask", mask.shape)
            # Forward pass
            outputs = model(
                image=pixel_values,
                input_ids=caps_in,
                attention_mask=mask
            )

            #print (outputs)
            # print (outputs.logits[:, 576:, :].shape)
            # print (caps_out.shape)

            loss = F.cross_entropy(
                        outputs.logits[:,576:, :].reshape(-1, outputs.logits.size(-1)),
                        caps_out.reshape(-1),
                        reduction='none'
                    )
            

            mask_flat = mask.reshape(-1).float()
            valid_token_count = mask_flat.sum().clamp(min=1.0)
            loss = (loss * mask_flat).sum() / valid_token_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_epoch_loss = train_epoch_loss/len(train_loader)
        epoch_losses.append(train_epoch_loss)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)

        print (f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss}")
        print (f"Caption: {tokenizer.decode(caps[0])}\n")
        print (f"len {mask[0].sum()}")
        print ("Pred:",tokenizer.decode(pred_ids[0]))

val = flickr_dataset(val_set[:50])

train_loader = DataLoader(val, batch_size=2, shuffle=True)

model = captioning(CLIP=CLIP, llama=llama_with_lora).cuda()

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6
# )

lr = 1E-5
optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

train(model, train_loader, optimizer, device="cuda", epochs=3)
