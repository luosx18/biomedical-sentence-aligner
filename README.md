# biomedical_sentence_aligner
 
codes for the paper Sentence Alignment with Parallel Documents Facilitates Biomedical Machine Translation.

The pseudo_documents.py generates and conmbines all pseudo_docoments into one txt for w2v training with the original document pairs.

The w2v.py trains the BEWs.

The sentence_aligner.py generate the aligned-sentence pairs.

The eval.py calculate the pre rec and f1 according to the results of sentence_aligner.py and manually aligned pairs.
