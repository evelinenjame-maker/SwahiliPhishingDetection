# Error Analysis Report

Generated from `combined_keep` configuration predictions on:
- Held-out test set: 482 messages
- BongoSCAM-only subset: 264 messages

## Error counts per algorithm

| Algorithm | Test set errors | BongoSCAM errors | Test acc | BongoSCAM acc |
|---|---|---|---|---|
| Naïve Bayes | 3 | 2 | 0.9938 | 0.9924 |
| Linear SVM | 1 | 1 | 0.9979 | 0.9962 |
| Random Forest | 0 | 0 | 1.0000 | 1.0000 |

## Error type breakdown

False Negatives (FN) = phishing missed (most dangerous in deployment).
False Positives (FP) = legitimate flagged as phishing (recoverable).

| Algorithm | False Negatives | False Positives |
|---|---|---|
| Naïve Bayes | 0 | 5 |
| Linear SVM | 2 | 0 |

## Examined examples (BongoSCAM test set)

### False Negatives (missed phishing)

- **Missed by:** Linear SVM, Linear SVM
  - **Message:** Mpigie Mzee NASHONI MBIRIBI kwa tiba asili miliki mali,utajiri pete,kazi,kesi,masomo mapenzi,kilimo,ufugaji,biashara Piga 0787-406-889


### False Positives (legitimate flagged as phishing)

- **Flagged by:** Naïve Bayes, Naïve Bayes
  - **Message:** Samahani, sikuweza kupokea simu yako.

- **Flagged by:** Naïve Bayes, Naïve Bayes
  - **Message:** Naomba unitumie namba ya huyo fundi.


## Discussion for dissertation

Operationally, false negatives (missed phishing) are more dangerous than 
false positives, since the latter can be filtered through user review. 
The error rate for each algorithm gives an upper bound on real-world 
deployment risk. Error overlap across algorithms suggests certain 
messages are genuinely ambiguous; these warrant qualitative discussion 
in the dissertation Limitations section.
