
# LegalInsight: Comprehensive Legal Strategy and Judgment Analysis

LegalInsight is an advanced Streamlit web application designed to simplify legal document analysis for professionals, researchers, and students. By uploading a legal document in PDF format, users can generate summaries, retrieve relevant past judgments, and receive strategic insights to guide judicial decisions. The app also includes a "Chat with PDF" feature that allows users to interact with the document through a conversational interface.


https://github.com/user-attachments/assets/9982ad66-a3b0-4442-bf8f-aa425f1fc160

## Features Overview

### PDF Upload
Upload any legal document in PDF format. LegalInsight supports chargesheets, case files, legal briefs, and more.

### Document Summary Generation
Automatically generate concise summaries of uploaded legal documents using state-of-the-art Natural Language Processing (NLP).

### Judgment Retrieval
Retrieve past judgments from similar cases, aiding in understanding precedents and preparing for court proceedings.

### Strategic Insights
LegalInsight suggests potential strategies and actions based on the uploaded document, leveraging AI to analyze legal principles and past judgments.

### Chat with PDF
The "Chat with PDF" feature allows users to interactively ask questions about the uploaded document, providing specific details and clarifications in real-time.

## Updates: Local LLM and Demo Access
LegalInsight now supports running processes locally to maintain privacy. Users can download Ollama for local LLM use, ensuring that data stays private. A demo is also available at: [LegalInsight Demo](https://legal-insight.streamlit.app/).

## Installation Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Ollama (optional, for local use)
- Together API Key (for using external APIs)

### Step-by-Step Installation for Local Use

1. **Clone the Repository**
   Clone the LegalInsight repository to your local machine:
   ```bash
   git clone https://github.com/KeshavG69/LegalInsight.git
   cd LegalInsight/local_code
   ```

**Activate virtual environment:**
   ```bash
   python3 -m venv virtual
   ```

   ```bash
   source virtual/bin/activate
   ```

   or for Windows-based machines -
   ```bash
   .\virtual\Scripts\activate
   ```

**Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Dependencies**
   Go back to the root folder and install the required Python packages:
   ```bash
   cd ..
   pip install -r requirements.txt
   ```

3. **Download and Set Up Ollama (for Local Use)**
   If you want to run LegalInsight with local LLM models, download and install [Ollama](https://ollama.com/download).

4. **Create a `.env` File**
   Create a `.env` file in the root directory and add your Together API key:
   ```
   TOGETHER_API_KEY=your_together_api_key_here
   ```
   This key is necessary for using external APIs to retrieve additional information or for document analysis.

5. **Run the Streamlit Application Locally**
   Run the application locally by executing:
   ```bash
   cd local_code
   streamlit run app.py
   ```

   The application will launch in your default web browser. If it doesn't, manually open `http://localhost:8501`.

### For Online Use (No Local Setup Required)
If you just want to test the app without setting it up locally, you can access it directly via the hosted version here: [LegalInsight Demo](https://legal-insight.streamlit.app/).

## File Structure

- **`src/`**: This folder hosts the code for the online demo of LegalInsight. **You do not need to interact with this folder** unless you are the maintainer hosting the web version.
- **`local_code/`**: Contains scripts for running LegalInsight entirely locally, ensuring privacy and local processing. This is the folder users should use for running the app on their local machine.
- **`requirements.txt`**: Located in the root directory, this file contains all the required dependencies for running the application.

## Technical Details

### Natural Language Processing (NLP)
LegalInsight uses pre-trained language models from the `transformers` library for legal document analysis. For local use, Ollama's locally hosted LLM models can be used.

### Streamlit Framework
Built using Streamlit, LegalInsight provides a seamless and interactive experience, allowing for rapid development and deployment of data-driven web applications.

### PDF Processing
The app utilizes `PyPDF2` to extract text from PDF files, which is then fed into the NLP pipeline for analysis.

### AI-Powered Strategy Generator
LegalInsight suggests potential strategies by comparing the current case with past judgments, providing actionable insights that could influence judicial decisions.

## Contribution Guidelines

LegalInsight is an open-source project, and contributions are highly encouraged. Follow these steps to contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Commit your changes to the branch.
4. Open a Pull Request with a detailed description of your changes.

Ensure your code adheres to the project's coding standards, and all tests pass before submitting the Pull Request.

## License

LegalInsight is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license. For more information, see the `LICENSE` file in the repository.

## Contact Information

For questions, feedback, or support inquiries, feel free to reach out:

- **Email**: gargkeshav504@gmail.com
- **LinkedIn**: [Keshav Garg](https://www.linkedin.com/in/keshav-garg-7760b1232/)

We look forward to your contributions and hope LegalInsight becomes an invaluable tool in your legal research and practice.













