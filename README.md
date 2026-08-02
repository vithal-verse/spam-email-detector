# Spam Email Detector

A beginner AI project based on the "Spam Email Detector" idea from the PDF list. The project is a small web app that classifies a message as spam or safe using a text classifier trained directly in JavaScript.

## Project Overview

The app lets a user paste an email or SMS-style message and checks whether it looks like spam. It also shows:

- The predicted label
- A confidence score
- The words that influenced the result
- Model statistics such as training sample count and vocabulary size

This project is intentionally dependency-free. You can run it by opening `index.html` in a browser.

## Folder Structure

```text
spam-email-detector/
  index.html                  Main web page
  styles.css                  User interface styling
  app.js                      Browser event handling and result rendering
  src/spamDetector.js         Dataset, preprocessing, model training, prediction
  tests/spamDetector.test.js  Simple Node.js tests for the classifier
  assets/email-shield.svg     Small visual asset used by the interface
  package.json                Project metadata and scripts
```

## How To Run

Open `index.html` in your browser.

You can also run a local server from this folder:

```bash
python3 -m http.server 8000
```

Then visit:

```text
http://localhost:8000
```

## How To Test

If Node.js is installed, run:

```bash
npm test
```

The test checks that obvious spam and normal messages are classified correctly.

## How The Code Works

1. `src/spamDetector.js` stores a small labeled dataset with `spam` and `ham` examples.
2. The `tokenize` function cleans text, lowercases it, removes punctuation, and splits it into useful words.
3. The model builds a vocabulary from the training data.
4. Each message is converted into TF-IDF features, which give more value to meaningful words.
5. A linear SVM-style classifier is trained with hinge loss.
6. `predict` converts new text into features, scores it, and returns the predicted label.
7. `app.js` connects the model to the page, reads user input, and displays the result.

## Why This Is An AI Project

The classifier learns from labeled examples instead of using fixed if-else rules. During training, it adjusts word weights so that spam-like words such as `winner`, `claim`, and `free` push the score toward spam, while normal words such as `meeting`, `report`, and `appointment` push the score toward safe.

## Limitations

This is a beginner learning project, not a production spam filter. The training dataset is small, so unusual messages can be misclassified. A real spam detector would use a much larger dataset, stronger evaluation, and ongoing retraining.
