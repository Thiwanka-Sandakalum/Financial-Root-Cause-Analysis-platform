import asyncio
import os
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Import the main orchestration function
from ingestion.pipeline.graph_writer import ingest_document

# Load environment variables (from your .env file)
load_dotenv()

def main():
    # 1. Setup Neo4j Driver
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # 2. Setup Gemini LLM and Embedder
    # Make sure GOOGLE_API_KEY is in your .env
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or gemini-2.0-pro-exp based on your setup
        temperature=0.0
    )
    
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    # 3. Define the document details
    file_path = "/home/thiwanka/Downloads/2020-annual-report1.pdf"
    
    print(f"Starting ingestion for {file_path}...")
    
    try:
        # 4. Run the ingestion pipeline
        doc_id = ingest_document(
            file_path=file_path,
            company_ticker="Dialog",
            company_name="Dialog Axiata PLC",
            doc_type="Annual Report",
            fiscal_period="2020",
            driver=driver,
            embedder=embedder,
            llm=llm
        )
        print(f"Success! Document ingested with ID: {doc_id}")
        
    finally:
        # Always close the neo4j connection
        driver.close()

if __name__ == "__main__":
    main()
