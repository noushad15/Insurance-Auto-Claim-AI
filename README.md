# Auto Claim AI System

An intelligent system for automated insurance claim processing using OCR, NLP, machine learning, and large language models (LLMs) — now with agentic (LangChain) AI workflows.

## 🏥 Overview

The Auto Claim AI system processes insurance claim documents (PDFs and images) to:

1.  **Upload Claim File** - Accept PDF/Image uploads via a FastAPI endpoint.
2.  **OCR & NLP Parsing** - Extract text and structured data from documents.
3.  **Field Extraction** - Identify key claim information like patient name, diagnosis, and claimed amount.
4.  **AI Classification** - Determine the claim's approval status with confidence scores.
5.  **LLM & Agentic Workflows** - Use Azure OpenAI and LangChain agents for natural language explanations, Q&A, and step-by-step claim reasoning.
6.  **Database Storage** - Store claim data and processing results in a local database.

## 🚀 Features

- **Multi-format Support**: PDF and image file processing.
- **Advanced OCR**: Text extraction with image preprocessing.
- **NLP Field Extraction**: Intelligent field identification.
- **AI Classification**: Rule-based and ML-powered claim assessment.
- **LLM-Powered Workflows**: Natural language claim review and analytics.
- **Agentic Claim Processing**: Step-by-step, tool-using agent (LangChain) for robust, explainable claim decisions.
- **FastAPI Backend**: A robust backend to handle requests and serve the AI models.
- **Database Integration**: Store and retrieve claim information.

## 📁 Project Structure

```
auto_claim_ai/
│
├── app/
│   ├── __init__.py
│   ├── agentic_claim_agent.py    # Agentic claim processing logic
│   ├── app.py                    # FastAPI application
│   ├── classifier.py             # AI classification model
│   ├── database.py               # Database models and sessions
│   ├── field_extractor.py        # NLP field extraction
│   ├── llm_utils.py              # Utilities for LLM interactions
│   └── ocr_parser.py             # OCR and text extraction
│
├── data/
│   └── sample_claims.pdf         # Sample claim documents
│
├── models/
│   └── claim_classifier.pkl      # Trained classification model
│
├── notebook/
│   └── claim_model_training.ipynb # Notebook for model training
│
├── scripts/
│   └── populate_sample_data.py   # Script to populate the database
│
├── test/
│   └── ...                       # Test files
│
├── agent_test.py                 # Script for testing the agent
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR engine

### Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd auto_claim_ai
    ```

2.  **Install Tesseract OCR**

    **macOS:**
    ```bash
    brew install tesseract
    ```

    **Ubuntu/Debian:**
    ```bash
    sudo apt-get install tesseract-ocr
    ```

    **Windows:**
    Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

3.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the Application

1.  **Start the FastAPI app**
    ```bash
    cd app
    uvicorn app:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

### Running the Agentic Claim Agent

You can test the agentic claim processing by running the `agent_test.py` script:

```bash
python agent_test.py
```

This script will simulate a claim processing request and show the agent's step-by-step reasoning.

## 📝 API Endpoints

The following are the main endpoints available:

- `POST /upload/`: Upload a claim document (PDF or image) for processing.
- `GET /claims/`: Retrieve a list of all claims.
- `GET /claims/{claim_id}`: Retrieve the details of a specific claim.

For more details, you can access the auto-generated API documentation at `http://127.0.0.1:8000/docs`.
