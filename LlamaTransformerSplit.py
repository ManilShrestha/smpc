import torch
import argparse
import os

from transformers import AutoTokenizer, LlamaForCausalLM

def save_split_transformer_layers(model_id, split_idx, output_file_1, output_file_2):
    
    # Load the llama model 
    pretrained_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    # Extract the model and config
    config = pretrained_model.config
    model = pretrained_model.model

    layers = model.layers[1:-1]

    # Split the number of decoder layers
    part_1_layers = layers[ : split_idx]
    part_2_layers = layers[ split_idx: ]

    # print(part_1_layers)

    # Save the split transformer model to file
    # layers_state_dicts_1 = {f'layer_{i}': layer.state_dict() for i, layer in enumerate(part_1_layers)}
    # torch.save(layers_state_dicts_1, output_file_1)
    torch.save(part_1_layers, output_file_1)

    print(f"Split transformer saved to {output_file_1}")
          
    # layers_state_dicts_2 = {f'layer_{i}': layer.state_dict() for i, layer in enumerate(part_2_layers)}
    # torch.save(layers_state_dicts_1, output_file_2)
    torch.save(part_2_layers, output_file_2)

    print(f"Split transformer saved to {output_file_2}")


def main():
    parser = argparse.ArgumentParser(description="Split the llama transformer decoder layers and save the parts to a file.")
    default_llama_model = "/storage/yr82@drexel.edu/Meta-Llama-3.1-8B"
    # Arguments for model path, split indices, and output file
    parser.add_argument("--model_id", type=str, default=default_llama_model, help="Path to the pre-trained model or model ID")
    parser.add_argument("--split_idx", type=int, default=15, help="Index to indicate how many decoder layer each split needs to contain.")
    parser.add_argument("--output_file_1", type=str, required=True, help="Path to the output file where the first split model will be saved")
    parser.add_argument("--output_file_2", type=str, required=True, help="Path to the output file where the second split model will be saved")

    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file_2), exist_ok=True)
    
    # Call the function to save the split transformer
    save_split_transformer_layers(args.model_id, args.split_idx, args.output_file_1, args.output_file_2)

if __name__ == "__main__":
    main()



# model_id = "/storage/yr82@drexel.edu/Meta-Llama-3.1-8B"

# save_split_transformer_layers(model_id, 16)

