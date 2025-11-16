import os
import torch
from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# HuggingFace tokeni .env'den çek
HF_TOKEN = os.getenv("HF_TOKEN")

# Tokeni login et
login(HF_TOKEN)

app = Flask(__name__)

# Cihazı belirle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli LOCALDEN yükle
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)


@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    error = None
    user_prompt = ""   # 💜 Prompt’u tutmak için

    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()

        if not user_prompt:
            error = "Please enter a prompt."
        else:
            try:
                full_prompt = (
                    f"{user_prompt}, cinematic poster art, dramatic lighting, high detail, "
                    f"sharp focus, vibrant colors, digital illustration, award-winning movie poster style"
                )

                # Görsel üretimi
                image = pipe(
                    full_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=12
                ).images[0]

                # Kaydet
                image.save("static/poster.png")
                image_url = "static/poster.png"

            except Exception as e:
                error = f"An error occurred: {e}"

    # 💜 user_prompt'u template'e geri gönderiyoruz → formda kalıyor
    return render_template("index.html", image_url=image_url, error=error, user_prompt=user_prompt)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
