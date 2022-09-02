# AI-Lab
This is my AI-Lab code. 
To make it run you first need besides this files to download the Hashtag2Vec - Project:
https://github.com/hezhichenghzc/Hashtag2Vec/blob/65393b58f0866326d514200bd1bb6ee3ca0d8680/Hashtag2Vec.zip

Then you need to replace the Hashtag2Vector file with the one in this repo.
Run the AI Lab Code. After a while (full dataset needs 8h ! ),  the message "Include data now" appears. You need to copy the matrices files (M_h_h.txt for 
example, 10 files) into the data folder of Hashtag2Vec. Make sure Hashtag2Vec uses the right folder. Run it. It creates 4 files (W_h_5_5_0.txt for example), 
which need to be copied into the LaborAI/Test folder. Now press enter. 2 diagrams are created with 4 subplots each and silhoutte and ami are calculated.

# Possible Improvements
Performance:
- Try using sentences with min. number of words/hashtags
- Check formulas of the Hashtag2Vec paper and how they are implemented

Runtime:
- Currently each sentence has 64 embeddings, make max_length of sentence and use it for bert (also for hashtags), has major impact on runtime
- Use Sparse matrices, currently dictionaries (not that impactful i feel like)

General:
- Make it run automated
- Coding style improvements

