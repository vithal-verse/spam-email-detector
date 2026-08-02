const detector = window.SpamDetector;
const model = detector.defaultModel;

const emailInput = document.querySelector("#email-input");
const analyzeButton = document.querySelector("#analyze-button");
const clearButton = document.querySelector("#clear-button");
const resultPanel = document.querySelector("#result-panel");
const verdictText = document.querySelector("#verdict-text");
const confidenceText = document.querySelector("#confidence-text");
const probabilityText = document.querySelector("#probability-text");
const meterFill = document.querySelector("#meter-fill");
const signalList = document.querySelector("#signal-list");
const tokenCount = document.querySelector("#token-count");
const sampleButtons = document.querySelectorAll("[data-sample]");

const samples = {
  spam: "Congratulations! You won a free cash prize. Claim your reward now using this link.",
  ham: "Hi, can we move the project meeting to Friday afternoon? I will send the report before lunch.",
  mixed: "Your payment receipt is attached. Please review it before the workshop tomorrow."
};

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderSignals(signals) {
  signalList.innerHTML = "";

  if (signals.length === 0) {
    const item = document.createElement("li");
    item.className = "empty-state";
    item.textContent = "No known training words matched.";
    signalList.appendChild(item);
    return;
  }

  signals.forEach((signal) => {
    const item = document.createElement("li");
    item.className = `signal signal-${signal.direction}`;
    item.innerHTML = `
      <span>${signal.token}</span>
      <strong>${signal.direction}</strong>
    `;
    signalList.appendChild(item);
  });
}

function renderPrediction() {
  const text = emailInput.value.trim();

  if (!text) {
    resultPanel.dataset.result = "empty";
    verdictText.textContent = "Waiting for text";
    confidenceText.textContent = "0% confidence";
    probabilityText.textContent = "Spam probability: 0%";
    meterFill.style.width = "0%";
    tokenCount.textContent = "0 matched words";
    renderSignals([]);
    return;
  }

  const prediction = detector.predict(text, model);
  const isSpam = prediction.label === "spam";

  resultPanel.dataset.result = prediction.label;
  verdictText.textContent = isSpam ? "Likely spam" : "Looks safe";
  confidenceText.textContent = `${percent(prediction.confidence)} confidence`;
  probabilityText.textContent = `Spam probability: ${percent(prediction.spamProbability)}`;
  meterFill.style.width = percent(prediction.spamProbability);
  tokenCount.textContent = `${prediction.matchedTokenCount} matched words`;
  renderSignals(prediction.signals);
}

function renderStats() {
  const stats = detector.getModelStats(model);
  document.querySelector("#training-size").textContent = stats.trainingSize;
  document.querySelector("#spam-count").textContent = stats.spamCount;
  document.querySelector("#ham-count").textContent = stats.hamCount;
  document.querySelector("#vocabulary-size").textContent = stats.vocabularySize;
}

analyzeButton.addEventListener("click", renderPrediction);

clearButton.addEventListener("click", () => {
  emailInput.value = "";
  emailInput.focus();
  renderPrediction();
});

emailInput.addEventListener("input", renderPrediction);

sampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    emailInput.value = samples[button.dataset.sample];
    emailInput.focus();
    renderPrediction();
  });
});

renderStats();
renderPrediction();
