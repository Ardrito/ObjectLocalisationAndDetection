from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse

import io

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

import transformers

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Set up environment ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load CLIP & Tokenizer ===
CLIP = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

CLIP.eval()
for param in CLIP.parameters():
    param.requires_grad = False

vocab = tokenizer.get_vocab()

token_embedding = CLIP.text_model.embeddings.token_embedding.to(device)

# === Load token embedding ===
token_embedding.weight.requires_grad = False

# === Decoder Layer & Decoder (Same as in training) ===
class decoder_layer(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + mask
        A = torch.softmax(scores, dim=-1)
        H = torch.matmul(A, V).transpose(1, 2).reshape(B, T, self.embed_dim)
        H = self.out_proj(H)

        x = self.norm1(x + H)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class decoder(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.patchPRoj = nn.Linear(768, embed_dim)
        self.decoder_layers = nn.ModuleList([decoder_layer(embed_dim, num_heads, output_dim) for _ in range(num_layers)])
        self.ln1 = nn.LayerNorm(embed_dim)
        self.reg_out = nn.Linear(embed_dim, output_dim)

    def get_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, patch_embedding, output_embedding):
        projected_patch = self.patchPRoj(patch_embedding)
        x = torch.cat((projected_patch, output_embedding), dim=1)

        seq_len = x.size(1)
        pe = self.get_positional_encoding(seq_len, x.size(-1)).to(x.device)
        x = self.ln1(x + pe.unsqueeze(0))

        for layer in self.decoder_layers:
            x = layer(x)

        return self.reg_out(x)

# === Instantiate model ===
model = decoder(embed_dim=512, output_dim=len(tokenizer), num_heads=4, num_layers=2).to(device)

# === Load weights ===
checkpoint = torch.load("checkpoints_4/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.4815, 0.4578, 0.4082],
        std=[0.2686, 0.2613, 0.2758]
    )
])


# === Inference Function ===
def inference(model, idx, start_token=49406, end_token=49407, max_len=77, device='cuda'):
    model.eval()
     # Shape: (1, 4, 196)
    
    with torch.no_grad():

        # Start with <start> token
        generated = [start_token]
        #img = get_test_image(idx)
        img = idx
        for _ in range(max_len):
            y = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

            y_emb = token_embedding(y).to(device)
            # print (y_emb.shape)
            # print (img.shape)
            
            logits = model(img, y_emb)# (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1]  # (vocab_size,)
            next_token = torch.argmax(next_token_logits).item()
            #print (next_token)
            
            generated.append(next_token)
            
            if next_token == end_token:
                break

    return generated[1:]


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    with torch.no_grad():
        vision_outputs = CLIP.vision_model(preprocess(image).unsqueeze(0).to(device))
        patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]

    caption_ids = inference(model, patch_embeddings)
    caption = tokenizer.decode(caption_ids, skip_special_tokens=True)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), caption, fill="white", font=font)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.post("/justJSON")
def returnJson(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    with torch.no_grad():
        vision_outputs = CLIP.vision_model(preprocess(image).unsqueeze(0).to(device))
        patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]

    caption_ids = inference(model, patch_embeddings)
    caption = tokenizer.decode(caption_ids, skip_special_tokens=True)

    return {"caption": caption}