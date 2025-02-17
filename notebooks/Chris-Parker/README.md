# Welcome to the SuperDataScience Community Project!
Welcome to the Sentiment Analysis using YouTube repository! 🎉

This project is a intended as a learning experience, and hence has not undergone full development (eg: no edit checks). 

# Project Scope of Works:

## Project Overview

This task makes use of the Youtube commentThreads api to retrieve the 100 most recent comments for a specified video, Didd add
a bit extra just to get some more details into the output, and play with multi-column display.

The comments are then extracted into a list, before having emojis converted to text (simple analysis showed unconverted 
emojis always resulted in a negative sentiment, whereas their text equivalents yielded better outcomes eg: thumbs-up and 
red-heart return a positive sentiment).

Comments are then passed to a BERT based classifier (it does seem to be biased towards negative)in order to obtain a 
sentiment rating POSITIVE or NEGATIVE, and a score. For this exercise a threshhold of 70% was applied with anything lower 
being reclassed as Neutral.  For simplicity model inference is kept as CPU.

Streamlit is being used for simple UI.


Execution:

After activating your .venv

Add the following to your .env:
SECRET_KEY="your YT API key"

pip install -r requirements.txt

streamlit run YT_comment_analysis.py

ctrl-c to exit.