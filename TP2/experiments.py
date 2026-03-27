from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline_utils import (
    DEFAULT_MODEL_ID,
    load_text2img,
    to_img2img,
    get_device,
    make_generator,
)

ROOT = Path(__file__).resolve().parent

ECOMMERCE_PROMPT = (
    "professional e-commerce product photo of a ceramic coffee mug on a clean white background, "
    "soft studio lighting, sharp focus, catalog style"
)


def save(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def ensure_product_input_image() -> Path:
    """Crée une image placeholder légère si aucun fichier source n'est fourni."""
    p = ROOT / "inputs" / "product_sample.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        img = Image.new("RGB", (512, 512), color=(245, 245, 245))
        d = ImageDraw.Draw(img)
        d.rectangle([100, 150, 412, 400], fill=(180, 100, 80), outline=(120, 60, 50), width=3)
        img.save(p, quality=90)
    return p


def run_baseline() -> None:
    model_id = DEFAULT_MODEL_ID
    scheduler_name = "EulerA"
    seed = 42
    steps = 30
    guidance = 7.5

    prompt = (
        "ultra-realistic product photo of a backpack on a white background, "
        "studio lighting, soft shadow, very sharp"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    pipe = load_text2img(model_id, scheduler_name)
    device = get_device()
    g = make_generator(seed, device)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512,
        width=512,
        generator=g,
    )

    img = out.images[0]
    save(img, ROOT / "outputs" / "baseline.png")

    print("OK saved outputs/baseline.png")
    print(
        "CONFIG:",
        {
            "model_id": model_id,
            "scheduler": scheduler_name,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
        },
    )


def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    prompt = ECOMMERCE_PROMPT
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15", "EulerA", 15, 7.5),
        ("run03_steps50", "EulerA", 50, 7.5),
        ("run04_guid4", "EulerA", 30, 4.0),
        ("run05_guid12", "EulerA", 30, 12.0),
        ("run06_ddim", "DDIM", 30, 7.5),
    ]

    device = get_device()
    for name, scheduler_name, steps, guidance in plan:
        pipe = load_text2img(model_id, scheduler_name)
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        save(img, ROOT / "outputs" / f"t2i_{name}.png")
        print(
            "T2I",
            name,
            {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance},
        )


def run_img2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    scheduler_name = "EulerA"
    steps = 30
    guidance = 7.5

    init_path = ensure_product_input_image()

    prompt = (
        "premium e-commerce product shot, same object, white seamless background, "
        "soft shadow, high-end catalog lighting"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    strengths = [
        ("run07_strength035", 0.35),
        ("run08_strength060", 0.60),
        ("run09_strength085", 0.85),
    ]

    pipe_t2i = load_text2img(model_id, scheduler_name)
    pipe_i2i = to_img2img(pipe_t2i)

    device = get_device()

    init_image = Image.open(init_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    for name, strength in strengths:
        g = make_generator(seed, device)
        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        save(img, ROOT / "outputs" / f"i2i_{name}.png")
        print(
            "I2I",
            name,
            {
                "scheduler": scheduler_name,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "strength": strength,
            },
        )


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "t2i":
            run_text2img_experiments()
        elif mode == "i2i":
            run_img2img_experiments()
        elif mode == "all":
            run_baseline()
            run_text2img_experiments()
            run_img2img_experiments()
        else:
            print("Usage: python experiments.py [t2i|i2i|all]")
            sys.exit(1)
    else:
        main()
