"""
Agentic Claim Processing Agent (LangChain)
-----------------------------------------
This module defines a LangChain agent that can:
- Accept a claim file (PDF/image) or raw text
- Use tools for OCR, field extraction, and classification (wrapping your existing functions)
- Reason step-by-step to process a claim, ask for missing info, and return a decision/explanation
- Can be called from a new UI page or API endpoint (does not change current navigation)
"""

from langchain.agents import initialize_agent, Tool
import os
import ocr_parser as ocr_parser_mod
import field_extractor as field_extractor_mod
import classifier as classifier_mod
import ast
import json
# Initialize LLM (Azure OpenAI for LangChain)
from langchain_community.llms import AzureOpenAI

# Initialize your existing components
ocr_parser = ocr_parser_mod.OCRParser()
field_extractor = field_extractor_mod.FieldExtractor()
classifier = classifier_mod.ClaimClassifier()

# Define tool wrappers

def ocr_tool(file_path: str) -> str:
    """Extract text from a PDF or image file."""
    if file_path.lower().endswith('.pdf'):
        data = ocr_parser.extract_text_from_pdf(file_path)
        return data['text'] + " " + data['images_text']
    else:
        return ocr_parser.extract_text_from_image(file_path)

def extract_fields_tool(text: str) -> dict:
    """Extract claim fields from text."""
    return field_extractor.extract_fields(text)

def classify_claim_tool(fields, raw_text: str = "") -> dict:
    """Classify a claim and return decision/explanation."""
    # Debug: print the input
    print("classify_claim_tool input:", fields)
    # If already a dict, use as is
    if isinstance(fields, dict):
        parsed_fields = fields
    # Try JSON first
    elif isinstance(fields, str):
        try:
            parsed_fields = json.loads(fields)
        except Exception:
            try:
                parsed_fields = ast.literal_eval(fields)
            except Exception:
                # Fallback: try to extract key fields from string
                parsed_fields = {}
                # Simple heuristics for common fields
                if 'name' in fields.lower():
                    parsed_fields['name'] = fields
                if 'date' in fields.lower():
                    parsed_fields['date'] = fields
                if 'amount' in fields.lower():
                    parsed_fields['amount'] = fields
                if not parsed_fields:
                    # If still nothing, just wrap as name
                    parsed_fields = {"name": fields}
    else:
        return {"error": f"Could not parse fields input as dict. Got: {fields}"}
    return classifier.classify_claim(parsed_fields, raw_text)

# Define LangChain tools
ocr_langchain_tool = Tool(
    name="OCR Extraction",
    func=ocr_tool,
    description="Extract text from a PDF or image file. Input: file path. Output: extracted text."
)
fields_langchain_tool = Tool(
    name="Field Extraction",
    func=extract_fields_tool,
    description="Extract claim fields from raw text. Input: text. Output: dict of fields."
)
classify_langchain_tool = Tool(
    name="Claim Classification",
    func=classify_claim_tool,
    description="Classify a claim using extracted fields and raw text. Input: dict of fields and raw text. Output: decision and explanation."
)


llm = AzureOpenAI(
    deployment_name="gpt-35-turbo-instruct",  #os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    temperature=0,
)

# Create the agent
agent = initialize_agent(
    tools=[ocr_langchain_tool, fields_langchain_tool, classify_langchain_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

def run_agentic_claim_workflow(file_path: str = None, raw_text: str = None) -> str:
    """
    Run the agentic claim workflow on a file or text.
    Returns the agent's reasoning and final decision/explanation.
    """
    if file_path:
        prompt = f"Process this claim file step by step: {file_path}. Use OCR, extract fields, classify, and explain."
    elif raw_text:
        prompt = f"Process this claim text step by step: {raw_text[:500]}... Use field extraction, classify, and explain."
    else:
        return "No input provided."
    return agent.run(prompt)

# Example usage (for testing, not UI):
# result = run_agentic_claim_workflow(file_path="/path/to/claim.pdf")
# print(result) 