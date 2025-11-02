import app.agentic_claim_agent as agentic_claim_agent

# Example: Run agent on a claim file (update the path to a real file)
file_path = "sample_claim.pdf"  # Replace with your actual file path
result_file = agentic_claim_agent.run_agentic_claim_workflow(file_path=file_path)
print("--- Agent Result for File ---")
print(result_file)

# Example: Run agent on raw claim text
raw_text = "Patient: John Doe\nDate: 2024-06-01\nDiagnosis: Fractured arm\nAmount: $3200\n..."
result_text = agentic_claim_agent.run_agentic_claim_workflow(raw_text=raw_text)
print("--- Agent Result for Raw Text ---")
print(result_text) 