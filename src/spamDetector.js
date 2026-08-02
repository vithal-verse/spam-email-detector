(function attachSpamDetector(root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SpamDetector = factory();
  }
})(typeof self !== "undefined" ? self : this, function createSpamDetector() {
  "use strict";

  const STOP_WORDS = new Set([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you"
  ]);

  const rawMessages = [
    { label: "spam", text: "Congratulations you have won a free lottery prize claim your cash reward now" },
    { label: "spam", text: "Urgent your account has a bonus click this link to claim the reward" },
    { label: "spam", text: "Win a brand new phone now click to collect your free gift" },
    { label: "spam", text: "Limited offer get cheap medicine without prescription order today" },
    { label: "spam", text: "You were selected for a 1000 dollar shopping voucher call now" },
    { label: "spam", text: "Claim your free vacation tickets winner confirmation required" },
    { label: "spam", text: "Act now your loan has been approved with zero credit check" },
    { label: "spam", text: "Earn money fast working from home no experience needed" },
    { label: "spam", text: "Exclusive deal buy one get one free only today click here" },
    { label: "spam", text: "Your parcel fee is pending pay now to avoid delivery failure" },
    { label: "spam", text: "Lowest price casino bonus deposit now and double your money" },
    { label: "spam", text: "Final notice verify your bank details to unlock your account" },
    { label: "spam", text: "Get rich quick with our crypto investment plan guaranteed returns" },
    { label: "spam", text: "Free ringtone subscription reply yes to activate today" },
    { label: "spam", text: "You won a luxury car send your address to receive the prize" },
    { label: "spam", text: "Urgent security alert update password using this link now" },
    { label: "spam", text: "Congratulations cash winner call this number immediately" },
    { label: "spam", text: "Offer ends tonight claim discount coupon before midnight" },
    { label: "spam", text: "Your tax refund is ready submit card details now" },
    { label: "spam", text: "Miracle weight loss pills free trial available today" },
    { label: "ham", text: "Hi Alex can we move the project meeting to tomorrow afternoon" },
    { label: "ham", text: "Please review the report and send your feedback before Friday" },
    { label: "ham", text: "Your appointment with Doctor Mehta is confirmed for 10 am Monday" },
    { label: "ham", text: "Lunch at the cafe today sounds good see you at one" },
    { label: "ham", text: "The invoice for last month has been attached for your records" },
    { label: "ham", text: "Team reminder standup starts in ten minutes" },
    { label: "ham", text: "Can you pick up milk on the way home" },
    { label: "ham", text: "Thank you for submitting the assignment on time" },
    { label: "ham", text: "Here are the notes from the machine learning class" },
    { label: "ham", text: "The delivery arrived at reception please collect it after work" },
    { label: "ham", text: "I updated the budget spreadsheet and shared the drive link" },
    { label: "ham", text: "Your library book is due next week" },
    { label: "ham", text: "Let us schedule a call to discuss the design changes" },
    { label: "ham", text: "Happy birthday hope you have a wonderful day" },
    { label: "ham", text: "The train leaves at seven thirty please reach early" },
    { label: "ham", text: "Attached are the minutes from yesterday meeting" },
    { label: "ham", text: "Could you confirm your availability for the workshop" },
    { label: "ham", text: "Dinner is ready when you get back" },
    { label: "ham", text: "We need to finalize the presentation slides by noon" },
    { label: "ham", text: "Your payment receipt for the course is attached" }
  ];

  function tokenize(text) {
    return String(text)
      .toLowerCase()
      .replace(/https?:\/\/\S+/g, " urltoken ")
      .replace(/\b\d+([.,]\d+)?\b/g, " numbertoken ")
      .replace(/[^a-z0-9$]+/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 2 && !STOP_WORDS.has(token));
  }

  function countTokens(tokens) {
    return tokens.reduce((counts, token) => {
      counts[token] = (counts[token] || 0) + 1;
      return counts;
    }, Object.create(null));
  }

  function buildVocabulary(samples) {
    const documentFrequency = Object.create(null);

    samples.forEach((sample) => {
      const uniqueTokens = new Set(tokenize(sample.text));
      uniqueTokens.forEach((token) => {
        documentFrequency[token] = (documentFrequency[token] || 0) + 1;
      });
    });

    const vocabulary = Object.keys(documentFrequency).sort();
    const tokenToIndex = Object.create(null);
    vocabulary.forEach((token, index) => {
      tokenToIndex[token] = index;
    });

    const idf = vocabulary.map((token) => {
      const frequency = documentFrequency[token];
      return Math.log((1 + samples.length) / (1 + frequency)) + 1;
    });

    return { vocabulary, tokenToIndex, idf };
  }

  function vectorize(text, modelParts) {
    const tokens = tokenize(text);
    const counts = countTokens(tokens);
    const vector = [];
    let squaredLength = 0;

    Object.keys(counts).forEach((token) => {
      const index = modelParts.tokenToIndex[token];
      if (index === undefined) {
        return;
      }

      const termFrequency = counts[token] / tokens.length;
      const value = termFrequency * modelParts.idf[index];
      vector.push([index, value, token]);
      squaredLength += value * value;
    });

    const length = Math.sqrt(squaredLength) || 1;
    return vector.map(([index, value, token]) => [index, value / length, token]);
  }

  function dot(weights, vector) {
    return vector.reduce((total, [index, value]) => total + weights[index] * value, 0);
  }

  function trainSvm(samples = rawMessages, options = {}) {
    const modelParts = buildVocabulary(samples);
    const weights = new Array(modelParts.vocabulary.length).fill(0);
    let bias = 0;

    const epochs = options.epochs || 120;
    const learningRate = options.learningRate || 0.65;
    const regularization = options.regularization || 0.0015;

    const trainingRows = samples.map((sample) => ({
      label: sample.label,
      target: sample.label === "spam" ? 1 : -1,
      vector: vectorize(sample.text, modelParts)
    }));

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const rate = learningRate / (1 + epoch * 0.04);

      trainingRows.forEach((row) => {
        for (let index = 0; index < weights.length; index += 1) {
          weights[index] *= 1 - rate * regularization;
        }

        const margin = row.target * (dot(weights, row.vector) + bias);
        if (margin < 1) {
          row.vector.forEach(([index, value]) => {
            weights[index] += rate * row.target * value;
          });
          bias += rate * row.target * 0.05;
        }
      });
    }

    return {
      ...modelParts,
      weights,
      bias,
      labels: ["ham", "spam"],
      trainingSize: samples.length,
      spamCount: samples.filter((sample) => sample.label === "spam").length,
      hamCount: samples.filter((sample) => sample.label === "ham").length
    };
  }

  function scoreToProbability(score) {
    const clippedScore = Math.max(-8, Math.min(8, score));
    return 1 / (1 + Math.exp(-clippedScore));
  }

  function getSignals(text, model, limit = 6) {
    return vectorize(text, model)
      .map(([index, value, token]) => ({
        token,
        weight: model.weights[index],
        contribution: model.weights[index] * value,
        direction: model.weights[index] >= 0 ? "spam" : "ham"
      }))
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, limit);
  }

  function predict(text, model = defaultModel) {
    const vector = vectorize(text, model);
    const rawScore = dot(model.weights, vector) + model.bias;
    const spamProbability = scoreToProbability(rawScore);
    const label = spamProbability >= 0.5 ? "spam" : "ham";
    const confidence = label === "spam" ? spamProbability : 1 - spamProbability;

    return {
      label,
      rawScore,
      spamProbability,
      confidence,
      tokens: tokenize(text),
      matchedTokenCount: vector.length,
      signals: getSignals(text, model)
    };
  }

  function getModelStats(model = defaultModel) {
    return {
      trainingSize: model.trainingSize,
      spamCount: model.spamCount,
      hamCount: model.hamCount,
      vocabularySize: model.vocabulary.length
    };
  }

  const defaultModel = trainSvm(rawMessages);

  return {
    rawMessages,
    tokenize,
    trainSvm,
    predict,
    getSignals,
    getModelStats,
    defaultModel
  };
});
