# Fine-tuning-Mistral-7B-ORPO

## Project Summary
- The provided project notebook focuses on fine-tuning a causal language model using the Optimal Rejection Perturbation Optimization (ORPO) method with Parameter-Efficient Fine-Tuning (PEFT) techniques. Here's a concise summary of the project:

## Environment Setup:

- Installs necessary libraries: bitsandbytes, transformers, peft, accelerate, datasets, and the latest version of trl from GitHub.
- Checks the GPU compatibility for FlashAttention and sets the appropriate torch data type (bfloat16 or float16).
  
## Model and Tokenizer Initialization:

- Loads the mistralai/Mistral-7B-v0.1 model and tokenizer.
- Configures the tokenizer to add an end-of-sequence (EOS) token.
  
## Dataset Preparation:

- Loads the HuggingFaceH4/ultrafeedback_binarized dataset, splitting it into training and testing preferences.
- Processes the dataset by applying a chat template to the "chosen" and "rejected" fields and tokenizes the processed text.
  
## Quantization Configuration:

- Sets up BitsAndBytesConfig for 4-bit quantization using nf4 type and specifies the compute data type and double quantization usage.
  
## Model Preparation:

- Loads the pre-trained causal language model with the specified quantization configuration and maps it to the appropriate device.
  
## PEFT Configuration:

- Defines a LoraConfig with parameters for LoRA (Low-Rank Adaptation) such as r, lora_alpha, lora_dropout, and target modules to be fine-tuned.
  
## ORPO Configuration:

- Sets up the ORPO training arguments including learning rate, beta, scheduler type, max sequence lengths, batch sizes, optimization method, number of epochs, evaluation strategy, logging steps, warmup steps, and output directory.
  
## Training and Evaluation:

- Initializes the ORPOTrainer with the model, training arguments, datasets, PEFT configuration, and tokenizer.
- Trains the model and saves the fine-tuned model to a specified directory.

The project combines state-of-the-art techniques in model fine-tuning, leveraging both PEFT and ORPO methods to enhance the performance of a large language model for a specific task. ​
