# Current State:
The bot is able to handle high chunk cases such as lengthy procedures but also small chunk cases such as searching for a serial number.  
No context or memory is implemented and bot is designed around metadata based on manuals.  
Bot has only been tested on manuals.  
Bot is considered to be in a working alpha state.  

# Future States:
Implement feedback logging and response optimization using response ratings.  
Test for long term conversational performance  

# How to run:
```
python rag.py
```

# File Path
```
Project Folder
|___chroma_db
    |___[Content will be populated on first run. Creating the 'chroma_db' folder is not necessary]
|___pdfs
    |___[Place all PDFs here]
|___.env
|___rag.py
```