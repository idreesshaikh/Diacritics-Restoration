# Diacritics-Restoration

**A character-level recurrent neural-network based model applied to diacritics restoration.**

  In the past and still at present, people often replace characters with diacritics with their ASCII counterparts. Even though the resulting text is usually easy to understand for humans, it is much harder for further computational processing. When writing emails, tweets or texts in certain languages, people for various reasons sometimes write without diacritics. When using Latin script, they replace characters with diacritics (e.g., c with acute or caron) by the underlying basic character without diacritics. Practically speaking, they write in ASCII.

## About the Task

  Almost half of the words in Hungarian Language Contains diacritics. So, restoring diacritics became an important task in particulars with Hungarian Language.
We make use of the preprocessed data of Hungarian Language into three train, dev, and test splits. Each one with and without their corresponding diacritics separated by a tab. That helps us read data eventually without the use of any other external library. In this very project we are trying to map our input sequence on the output sequence with a ‘character level embedding’ implementation. a Character-Word Long Short-Term Memory Language Model which both reduces the perplexity with respect to a baseline word-level language model and reduces the number of parameters of the model. Character information can reveal structural (dis)similarities between words and can even be used when a word is out-of-vocabulary, thus improving the modeling of infrequent and unknown words. We will use a parallel dataset of the following form where each tuple represents a pair of (non_diacritized, diacritized) input and output sequences.

  **(‘koszonom szepen’, ‘köszönöm szépen’)**

  
#### Requirement:    
  **Extract diacritic_data.rar to 'diacritic_data' folder which includes train and test files. The dataset is 'non diacritized data' and 'diacritized data' separated by tab. You can use your own data/language, but in my case I'm restoring diacritics in Hungarian language.**

  
![](image/BiLSTM.png)

#### Note:
  **Load_Model variable in mainfile.py can be set to True to load the trained data, but keep it False if you're running the program for the first time. It's set to False by default.**
