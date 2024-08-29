# smpc

## Pre-requisite
To run the scripts, the diffusers library has been customized to be compatible with the split transformer logic. Follow these steps:

1. Uninstall base diffusers: `pip uninstall diffusers`
2. Install customized diffuser from: https://github.com/ManilShrestha/diffusers
   - `git clone https://github.com/ManilShrestha/diffusers.git`
   - `cd diffusers`
   - `pip install -e .`


## TransformerSplitFactory.py
**Overview**

TransformerSplitFactory.py is a script designed to split a transformer model from the Stable Diffusion 3 pipeline into specific blocks and save the resulting model to a specified output file. This allows you to distribute parts of the transformer model across different machines or processes.

_**This is run by the owner of model to distribute the blocks of transformers to the distributed servers.**_

**Usage**
The script can be executed from the command line with configurable options:

- --model_id          : Path to the pre-trained model or model ID (default: a specific SD3 model path).
- --split_mode        : Split for 'client' or 'server' (default: 'server')
- --split_start       : Starting index for the transformer block split (default: 0).
- --split_end         : Ending index for the transformer block split (default: 23).
- --output_file       : Path to the output file where the split model will be saved (required).
- --last_block        : Does this block split contain the last block of the transformer (optional)


Example:
--------
For server split:

    python TransformerSplitFactory.py --split_mode "server" --output_file "/path/to/output/file.pth" --split_start 1 --split_end 11
    
For client split:

    python TransformerSplitFactory.py --split_mode "client" --output_file "/path/to/output/file.pth" --has_last_block True



## TransformerSplitServer.py
**Overview**

TransformerSplitServer.py is a script that implements a server to load a pre-split transformer model and process client requests for inference. 

_**This is intended to be run by the distributed hosts using the model file provided to them.**_


**Usage**
The script can be executed from the command line with configurable options:
- --host: IP/Hostname where the model will be reachable
- --port: Port that hosts this model
- --device: The cuda device (default: cuda).


`python TransformerSplitServer.py --split_model_file_path "/path/to/model.pth" --host "0.0.0.0" --port 8765 --device "cuda"`
