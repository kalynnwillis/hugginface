from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    print("Hi, I'm your chatbot! Type 'quit' to exit.")
    chat_history_ids = None

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'quit':
            break

        # Encode user input and append the end-of-string token
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history (if it exists)
        bot_input_ids = new_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_input_ids], dim=-1)

        # Generate a response from the model
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        print("Bot:", response)

if __name__ == "__main__":
    main()
