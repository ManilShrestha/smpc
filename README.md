# smpc

## Pre-requisite
To run the scripts, the diffusers library has been customized to be compatible with the split transformer logic. Follow these steps:

1. Uninstall base diffusers: `pip uninstall diffusers`
2. Install customized diffuser from: https://github.com/ManilShrestha/diffusers
   - `git clone [https://github.com/ManilShrestha/diffusers](https://github.com/ManilShrestha/diffusers.git)`
   - `cd diffusers`
   - `pip install -e .`


## TransformerSplitFactory.py
**Overview**

TransformerSplitFactory.py is a script designed to split a transformer model from the Stable Diffusion 3 pipeline into specific blocks and save the resulting model to a specified output file. This allows you to distribute parts of the transformer model across different machines or processes.

This is run by the owner of model to distribute the blocks of transformers to the distributed servers.

**Usage**
The script can be executed from the command line with configurable options:

- --model_id: Path to the pre-trained model or model ID (default: a specific Stable Diffusion 3 model path).
- --split_start: The starting index for the transformer block split (default: 0).
- --split_end: The ending index for the transformer block split (default: 23).
- --output_file: The path where the split model will be saved (required).
- --has_last_block: Indicates whether this block split contains the last block of the transformer.

`python TransformerSplitFactory.py --output_file "/path/to/output/file.pth" --split_start 1 --split_end 11`


## TransformerSplitServer.py
**Overview**

TransformerSplitServer.py is a script that implements a server to load a pre-split transformer model and process client requests for inference. 

This is intended to be run by the distribute hosts with the model file provided to them.


**Usage**
The script can be executed from the command line with configurable options:
- --host: IP/Hostname where the model will be reachable
- --port: Port that hosts this model
- --device: The cuda device (default: cuda).


`python TransformerSplitServer.py --split_model_file_path "/path/to/model.pth" --host "0.0.0.0" --port 8765 --device "cuda"`
