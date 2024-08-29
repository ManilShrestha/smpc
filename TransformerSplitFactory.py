"""
This script splits a transformer model from the Stable Diffusion 3 pipeline into specific blocks
and saves the resulting model to a specified output file. The script is configurable via command-line
arguments for flexibility in selecting the model, the range of transformer blocks to split, and the
output file path.

Usage:
------
Run the script from the command line with the following options:
    --model_id          : Path to the pre-trained model or model ID (default: a specific SD3 model path).
    --split_mode        : Split for 'client' or 'server' (default: 'server')
    --split_start       : Starting index for the transformer block split (default: 0).
    --split_end         : Ending index for the transformer block split (default: 23).
    --output_file       : Path to the output file where the split model will be saved (required).
    --last_block        : Does this block split contain the last block of the transformer (optional)

Example:
--------
For server split:
    python TransformerSplitFactory.py --split_mode "server" --output_file "/path/to/output/file.pth" --split_start 1 --split_end 11
For client split:
    python TransformerSplitFactory.py --split_mode "client" --output_file "/path/to/output/file.pth" --has_last_block True
"""

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3_split import (SD3Transformer2DModelServerSplit,    
                                                                 SD3Transformer2DModelClientSplit)
import argparse
import os


def save_split_transformer_client(model_id, output_file, has_last_block=False):
    # Load the pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Get the combined transformer model
    combined_model = pipe.transformer

    if not has_last_block:
        blocks = combined_model.transformer_blocks[0]
        time_text_embed_state_dict = combined_model.time_text_embed.state_dict()
        context_embedder_state_dict = combined_model.context_embedder.state_dict()
        pos_embed_state_dict = combined_model.pos_embed.state_dict()

        transformer_split = SD3Transformer2DModelClientSplit(
            pipe.transformer.config, 
            blocks,
            has_last_block,
            time_text_embed_state_dict, 
            context_embedder_state_dict,
            pos_embed_state_dict
        )

    else:
        blocks = combined_model.transformer_blocks[-1]
        norm_out_state_dict = combined_model.norm_out.state_dict()
        proj_out_state_dict = combined_model.proj_out.state_dict()
        
        transformer_split = SD3Transformer2DModelClientSplit(
            pipe.transformer.config, 
            blocks,
            has_last_block,
            norm_out_state_dict, 
            proj_out_state_dict
        )
    
    # Save the split transformer model to a file
    torch.save(transformer_split, output_file)
    print(f"Split transformer saved to {output_file}")

def save_split_transformer_server(model_id, split_start_idx, split_end_idx, output_file):
    # Load the pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Get the combined transformer model
    combined_model = pipe.transformer

    part2_blocks = combined_model.transformer_blocks[split_start_idx:split_end_idx+1]
    # Create the split transformer model
    transformer_split = SD3Transformer2DModelServerSplit(
        pipe.transformer.config, 
        part2_blocks
    )
    # Save the split transformer model to a file
    torch.save(transformer_split, output_file)
    print(f"Split transformer saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Split a transformer model and save the parts to a file.")
    default_sd3_model = "/storage/ms5267@drexel.edu/models/stable-diffusion-3-medium-diffusers/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"

    # Arguments for model path, split indices, and output file
    parser.add_argument("--model_id", type=str, default=default_sd3_model, help="Path to the pre-trained model or model ID")
    parser.add_argument("--split_mode", type=str, default="server", help="Either server or client side model")
    parser.add_argument("--split_start", type=int, default=0, help="Starting index for the transformer block split (default: 0)")
    parser.add_argument("--split_end", type=int, default=23, help="Ending index for the transformer block split (default: 23)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file where the split model will be saved")
    parser.add_argument("--has_last_block", type=bool, default=False, help="Path to the output file where the split model will be saved")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Call the function to save the split transformer
    if args.split_mode=='server':
        save_split_transformer_server(args.model_id, args.split_start, args.split_end, args.output_file)
    else:
        save_split_transformer_client(args.model_id, args.output_file, args.has_last_block)

if __name__ == "__main__":
    main()
