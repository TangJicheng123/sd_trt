python demo_txt2img.py "a beautiful girl" --denoising-steps 20

python demo_txt2img.py "masterpiece, best quality, 1girl, vivid_paina, blue hair, brown eyes, short hair" --negative-prompt "ugly, bad face, fused hand, fused feet, worst quality, low quality, bad hands, missing fingers, weapon, sword, holding, text, signature" --height 768 --width 512 --seed 91652449 --denoising-steps 20 --scheduler "EulerA" --version 1.5 --build-static-batch --num-warmup-runs 0

python demo_txt2img.py "a sleep girl" --negative-prompt "ugly, bad face, fused hand, fused feet, worst quality, low quality, bad hands, missing fingers, weapon, sword, holding, text, signature" --height 768 --width 512 --seed 91652449 --denoising-steps 20 --scheduler "EulerA" --version 1.5 --build-static-batch --num-warmup-runs 0