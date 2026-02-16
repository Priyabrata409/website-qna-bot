# Website Q&A Bot - Verification Walkthrough

## 1. Environment Setup
You need to set up your API keys in the `.env` file.
1.  Navigate to the `website_qa_bot` directory:
    ```bash
    cd website_qa_bot
    ```
2.  Copy the example env file:
    ```bash
    cp .env.example .env
    ```
3.  Edit `.env` and add your keys:
    - `OPENAI_API_KEY`: Your OpenAI API key.
    - `PINECONE_API_KEY`: Your Pinecone API key.
    - `PINECONE_INDEX_NAME`: Your desired index name (e.g., "website-qa").

## 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt


## 3. Running the App
Start the Streamlit application:

```bash
streamlit run app.py