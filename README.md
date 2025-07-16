# Attocube Support AI Chatbot

## Overview:
The **Attocube Support AI Chatbot** is an intelligent chatbot designed to provide instant, accurate answers about Attocube equipment and documentation. Built using advanced Retrieval-Augmented Generation (RAG) technology, this assistant can understand complex technical queries and provide detailed responses with relevant source citations and visual context.

## Key Features

### 🤖 **Intelligent Document Understanding**
- **Dual-Granularity Search**: Uses both fine-grained and coarse-grained document chunking for optimal context retrieval
- **Smart Query Processing**: Automatically detects procedural vs. informational queries and adjusts search strategy accordingly
- **Follow-up Context Expansion**: Increases context window for follow-up questions to provide more comprehensive answers

### 📄 **Advanced PDF Processing**
- **Multi-Modal Content Extraction**: Extracts and processes both text and images from PDF documentation
- **Image Integration**: Displays relevant diagrams, schematics, and visual aids alongside text responses
- **Source Attribution**: Provides precise citations with document names, page numbers, and document types

### 🔍 **Enhanced Search Capabilities**
- **Context-Aware Retrieval**: Maintains conversation history to understand follow-up questions
- **Document Type Filtering**: Can focus searches on specific types of documentation
- **Semantic Search**: Uses Google Vertex AI embeddings for intelligent content matching

### 🛡️ **Enterprise-Ready Security**
- **Google OAuth Integration**: Secure authentication with domain restrictions (@lbl.gov)
- **Session Management**: Maintains separate conversation contexts per user session
- **Cloud-Native Architecture**: Designed for Google Cloud Run with automatic scaling

### 🎯 **User Experience Features**
- **Markdown Support**: Rich text formatting in responses with code blocks, lists, and emphasis
- **Debug Mode**: Technical users can view retrieved chunks and search metadata
- **Responsive Design**: Clean, modern interface optimized for desktop and mobile
- **Real-time Interaction**: Fast response times with loading indicators

## Technical Architecture

- **Frontend**: Flask web application with responsive HTML/CSS/JavaScript interface
- **Backend**: Python-based RAG system using LangChain and Google Vertex AI
- **Vector Database**: ChromaDB for efficient semantic search and retrieval
- **Cloud Infrastructure**: Deployed on Google Cloud Run with Cloud Storage integration
- **AI Models**: Google Gemini for language generation and Vertex AI for embeddings

## How to run locally:
This app was built on a Windows system and is made to run locally on a Windows system for local development testing. Therefore setting up a Windows environment is necessary to run the `run_local.bat` batch file. Set up file system as shown below, the GCS bucket is necessary to retrieve PDFs. Then run the command below in project directory.
```
./run_local.bat
```

## File Path
```
attocube-support\
├──.env
├──rag.py
├──templates\
   ├──index.html
   ├──login.html
├──app.py
├──.gcloudignore
├──Dockerfile
├──requirements.txt
├──run_local.bat
├──run_local.py
├──test_config.py
├──verify_deployment.py
├──check-dependencies.py
```

## GCS Bucket
```
attocube-support-pdfs\
├──pdfs\
   ├──[Place PDFs here]
```