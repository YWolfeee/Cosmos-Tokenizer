# @title In this step, load the required checkpoints, and perform video reconstruction. {"run":"auto"}
import os
import cv2
import numpy as np
import torch

import importlib
import cosmos_tokenizer.video_lib
import mediapy as media

importlib.reload(cosmos_tokenizer.video_lib)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

# 1) Specify the model name, and the paths to the encoder/decoder checkpoints.
model_name = 'Cosmos-1.0-Tokenizer-DV8x16x16' # @param ["Cosmos-0.1-Tokenizer-CV4x8x8", "Cosmos-0.1-Tokenizer-CV8x8x8", "Cosmos-0.1-Tokenizer-CV8x16x16", "Cosmos-0.1-Tokenizer-DV4x8x8", "Cosmos-0.1-Tokenizer-DV8x8x8", "Cosmos-0.1-Tokenizer-DV8x16x16", "Cosmos-1.0-Tokenizer-CV8x8x8", "Cosmos-1.0-Tokenizer-DV8x16x16"]
temporal_window = 49 # @param {type:"slider", min:1, max:121, step:8}

encoder_ckpt = f"pretrained_ckpts/{model_name}/encoder.jit"
decoder_ckpt = f"pretrained_ckpts/{model_name}/decoder.jit"

# 2) Load or provide the video filename you want to tokenize & reconstruct.
input_filepath = "test_data/video.mp4"

# 3) Read the video from disk (shape = T x H x W x 3 in BGR).
input_video = media.read_video(input_filepath)[..., :3]
assert input_video.ndim == 4 and input_video.shape[-1] == 3, "Frames must have shape T x H x W x 3"

# 4) Expand dimensions to B x Tx H x W x C, since the CausalVideoTokenizer expects a batch dimension
#    in the input. (Batch size = 1 in this example.)
batched_input_video = np.expand_dims(input_video, axis=0)

# 5) Create the CausalVideoTokenizer instance with the encoder & decoder.
#    - device="cuda" uses the GPU
#    - dtype="bfloat16" expects Ampere or newer GPU (A100, RTX 30xx, etc.)
tokenizer = CausalVideoTokenizer(
    checkpoint_enc=encoder_ckpt,
    checkpoint_dec=decoder_ckpt,
    device="cuda",
    dtype="bfloat16",
)

# 6) Use the tokenizer to autoencode (encode & decode) the video.
#    The output is a NumPy array with shape = B x T x H x W x C, range [0..255].
batched_output_video = tokenizer(batched_input_video,
                                 temporal_window=temporal_window)

# 7) Extract the single video from the batch (index 0).
output_video = batched_output_video[0]

# 9) Save the reconstructed video to disk.
input_dir, input_filename = os.path.split(input_filepath)
filename, ext = os.path.splitext(input_filename)
output_filepath = f"{input_dir}/{filename}_{model_name.split('-')[-1]}{ext}"
media.write_video(output_filepath, output_video)
print("Input video read from:\t", f"{os.getcwd()}/{input_filepath}")
print("Reconstruction saved:\t", f"{os.getcwd()}/{output_filepath}")

# 10) Visualization of the input video (left) and the reconstruction (right).
# Save side-by-side comparison video
comparison_filepath = f"{input_dir}/{filename}_comparison_{model_name.split('-')[-1]}{ext}"
combined_video = np.concatenate([input_video, output_video], axis=2)  # Concatenate horizontally
media.write_video(comparison_filepath, combined_video)
print("Comparison video saved:\t", f"{os.getcwd()}/{comparison_filepath}")