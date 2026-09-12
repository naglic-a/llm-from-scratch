import argparse
from pathlib import Path
import torch

from config import ModelConfig
from data.tokenizer import load_tokenizer, LLMTokenizer
from generation import generate_tokens
from model.gpt import GPTModel
from training import resolve_device

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Chatbot")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gpt.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()

def standalone_decodable_token_ids(tokenizer: object) -> list[int]:
    valid_ids = []
    for token_id in range(tokenizer.vocab_size()):
        try:
            tokenizer.decode([token_id])
        except UnicodeError:
            continue
        valid_ids.append(token_id)
    if not valid_ids:
        raise ValueError("tokenizer has no independently decodable tokens")
    return valid_ids

def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    
    print(f"Loading model from {arguments.checkpoint}...")
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPTModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = LLMTokenizer(load_tokenizer(checkpoint["tokenizer_path"]))
    allowed_token_ids = torch.tensor(
        standalone_decodable_token_ids(tokenizer),
        dtype=torch.long,
        device=device,
    )
    
    import random
    
    print("Model loaded! Type 'quit' or 'exit' to stop.")
    print("-" * 50)
    print("\n*** The AI is now running in Chatbot mode! ***\n")

    conversation_history = ""
    stop_words = ["<|endoftext|>", "<|user|>"]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Farewell!")
            break
            
        # Add the user's turn to the context using the special tokens!
        prompt = conversation_history + f"<|user|>{user_input}<|bot|>"
        
        prompt_ids = tokenizer.encode(prompt)
        
        # Prevent the context from exceeding the model's max_context_length
        max_prompt_length = model_config.max_context_length - arguments.max_new_tokens
        if max_prompt_length <= 0:
            print("Error: max_context_length is too small to generate that many tokens.")
            break
            
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
            
        token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        generated_ids = generate_tokens(
            model=model,
            token_ids=token_ids,
            max_new_tokens=arguments.max_new_tokens,
            context_length=model_config.max_context_length,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            allowed_token_ids=allowed_token_ids,
        )
        
        new_token_ids = generated_ids[0][len(prompt_ids):].tolist()
        response = tokenizer.decode(new_token_ids)
        
        for stop_word in stop_words:
            if stop_word in response:
                response = response.split(stop_word)[0]
                
        response = response.strip()
        print(f"AI: {response}")
        
        # Save to history with the proper special tokens
        conversation_history += f"<|user|>{user_input}<|bot|>{response}<|endoftext|>"

if __name__ == "__main__":
    main()
