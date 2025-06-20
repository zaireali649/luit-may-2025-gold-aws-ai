from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize conversation history
chat_history_ids = None
attention_mask = None

print("Chatbot is ready! Type 'quit' to exit or 'reset' to clear conversation history.")
for step in range(5):  # Adjust as needed
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    if user_input.lower() == "reset":
        chat_history_ids = None
        attention_mask = None
        print("Chat history cleared.")
        continue

    # Encode input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    new_attention_mask = torch.ones_like(new_input_ids)

    # Combine with chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
    else:
        bot_input_ids = new_input_ids
        attention_mask = new_attention_mask

    # Generate response with attention mask
    output_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

    # Update history
    chat_history_ids = output_ids
    attention_mask = torch.ones_like(chat_history_ids)

    # Decode and print reply
    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
