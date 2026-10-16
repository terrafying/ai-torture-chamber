"""Robot-malfunction inserts for the needle cut: Wan2.2-TI2V-5B, portrait 704x1280, ~3 s at 24 fps.
Writes out/NN_name.mp4 as each finishes, then out/ALL_DONE."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time, torch, traceback
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
OUT = "/workspace/out"; os.makedirs(OUT, exist_ok=True)
STYLE = ", cinematic, dramatic hard lighting, hot pink and acid lime neon rim light, deep black shadows, shallow depth of field, 35mm film grain, highly detailed"
CLIPS = [
 ("sparks_head", "Extreme close-up of a sleek white humanoid robot head, its faceplate cracking apart as bright sparks burst from the seams and smoke pours out"),
 ("hand_button", "A trembling chrome robotic hand hovering above a big glossy red button, the fingers twitching and jerking, drops of black oil falling, dark room"),
 ("eye_glitch", "Macro shot of an android's glass eye, the iris spinning and zooming erratically, reflections of flashing red warning lights on the lens"),
 ("server_alarm", "Long rows of server racks in a dark data center, status lights rippling from green to red in waves, a red alarm beacon strobing, sparks falling from the ceiling"),
 ("collapse", "A chrome humanoid robot slowly collapsing to its knees in an empty white corridor, joints locking up, steam venting from its shoulders"),
 ("crt_melt", "An old CRT television in a dark room showing a face made of static that warps and melts, magenta and cyan colours bleeding, VHS distortion"),
 ("skull_cables", "Dozens of brightly coloured cables plugged into the back of a robot's metal skull, electric arcs jumping between the plugs, black background, slow motion"),
 ("arm_spasm", "An orange industrial robot arm on a factory floor spasming out of control, swinging wildly, sparks flying, red emergency lights spinning"),
 ("egg_crack", "A single white egg on a polished steel table cracks open in slow motion, a thin glowing pink liquid seeping out, black background"),
 ("led_face", "A robot face made of glowing LED pixels flickering between a calm expression, a frightened expression and a crying expression, then freezing on a corrupted error pattern"),
 ("two_doors", "Two identical doors at the end of a dark foggy hallway, red light glowing from under the left door, the camera slowly pushing forward between them"),
 ("wired_chair", "A humanoid robot sitting upright in a metal chair with thick wires running into its head, convulsing slightly, then going completely still, cold blue light"),
 ("oil_tears", "Close-up of a polished silver android face, black oil tears running slowly down its cheeks, pink rim light"),
 ("gauge_red", "Macro shot of an analog gauge, its needle slamming back and forth into the red zone and vibrating violently, sparks bouncing off the glass"),
]
NEG = "text, letters, subtitles, watermark, logo, signature, real human, person, blurry, low quality, jpeg artifacts, deformed, static frame, still image"
mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()     # the full-frame fp32 decode of 704x1280x73 runs out of 48 GB
for i, (name, prompt) in enumerate(CLIPS):
    path = f"{OUT}/{i:02d}_{name}.mp4"
    if os.path.exists(path): continue
    t0 = time.time()
    try:
        frames = pipe(prompt=prompt + STYLE, negative_prompt=NEG, height=1280, width=704, num_frames=73,
                      guidance_scale=5.0, num_inference_steps=30, generator=torch.Generator("cuda").manual_seed(100 + i)).frames[0]
        export_to_video(frames, path, fps=24)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    torch.cuda.empty_cache()
open(f"{OUT}/ALL_DONE", "w").write("ok")
