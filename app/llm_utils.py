import os
import openai

# Load Azure OpenAI credentials from environment variables
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"  # Update if needed

def get_claim_llm_review(extracted_fields, classification_result, raw_text=None, extra_prompt=None):
    """
    Send claim data to Azure OpenAI LLM and get a review/explanation.
    """
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
        return "[LLM not configured: Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT]"

    # Format the prompt
    prompt = (
        "You are an expert insurance claim analyst AI.\n"
        "Here is the extracted claim data:\n"
        f"{extracted_fields}\n"
        "Classification result:\n"
        f"{classification_result}\n"
    )
    if raw_text:
        prompt += f"\nRaw OCR text:\n{raw_text[:1000]}...\n"  # Limit raw text for brevity
    if extra_prompt:
        prompt += f"\n{extra_prompt}\n"
    prompt += (
        "\nPlease do the following:\n"
        "- Summarize the claim in plain English.\n"
        "- Explain why the claim was classified as it was.\n"
        "- Suggest any missing information or next steps.\n"
        "- If you see any risk or fraud indicators, mention them.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful insurance claim assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"[LLM error: {str(e)}]" 