from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from data import PROMPT, SENTIMENTS, TEST_CASES
text = TEST_CASES[0]['text']
sentiment = TEST_CASES[0]['expected_sentiment']
prompt = PROMPT.format(text)
print(prompt)
# Load pre-trained model and tokenizer
model_name = "MBZUAI/LaMini-GPT-1.5B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name,
                                        load_in_8bit=True,
                                        )

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt")

def generate_single_token(input_text):
    # Encode the input text and get the last token's id
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    last_token_id = input_ids[0, -1].unsqueeze(0).unsqueeze(0)
    
    # Generate a single token
    output = text_generator(input_text,
                            max_new_tokens=1,
                            use_cache=True,
                            )

    generated_text = output[0]['generated_text']
    
    # Extract and return the generated token
    new_token = generated_text[len(input_text):].strip()
    return new_token

# Example usage
generated_token = generate_single_token(prompt)
print(f"Generated token: {generated_token}")

