const form = document.querySelector("#predict-form");
const urlInput = document.querySelector("#url");
const submitButton = form.querySelector("button");
const result = document.querySelector("#result");
const label = document.querySelector("#label");
const phishingScore = document.querySelector("#phishing-score");
const scoreLabel = document.querySelector("#score-label");
const modelSource = document.querySelector("#model-source");
const explanation = document.querySelector("#explanation");
const explanationText = document.querySelector("#explanation-text");
const riskFactors = document.querySelector("#risk-factors");
const features = document.querySelector("#features");

const visibleFeatures = [
  ["url_length", "URL length"],
  ["domain_length", "Domain length"],
  ["num_dots", "Dots"],
  ["num_hyphens", "Hyphens"],
  ["num_digits", "Digits"],
  ["has_ip", "Has IP"],
  ["has_at", "Has @"],
  ["num_subdomains", "Subdomains"],
  ["has_suspicious_tld", "Risky TLD"],
  ["has_url_shortener", "Shortener"],
  ["has_redirect_param", "Redirect param"],
  ["has_embedded_url", "Embedded URL"],
  ["has_long_token", "Long token"],
  ["has_public_hosting_platform", "Public hosting"],
  ["has_brand_impersonation", "Brand signal"],
  ["suspicious_word_count", "Suspicious words"],
  ["html_available", "HTML available"],
  ["num_forms", "Forms"],
  ["num_password_inputs", "Password inputs"],
  ["num_iframes", "Iframes"],
  ["has_login_form", "Login form"],
  ["has_meta_refresh", "Meta refresh"],
  ["has_javascript_redirect", "JS redirect"],
];

function statusText(value) {
  return value ? "Detected" : "Not detected";
}

function buildRiskFactors(data) {
  const f = data.features || {};
  return [
    ["Public Hosting", statusText(f.has_public_hosting_platform)],
    ["Brand Impersonation", statusText(f.has_brand_impersonation)],
    ["URL Shortener", statusText(f.has_url_shortener)],
    ["Redirect Parameter", statusText(f.has_redirect_param)],
    ["Embedded URL", statusText(f.has_embedded_url)],
    ["Risky TLD", statusText(f.has_suspicious_tld)],
    ["IP Address", statusText(f.has_ip)],
    ["At Symbol", statusText(f.has_at)],
    ["Long Token", statusText(f.has_long_token)],
    ["Encoded Characters", statusText(f.has_encoded_chars)],
    ["HTML Collection", f.html_available ? "Available" : "Unavailable"],
    ["Login Form", statusText(f.has_login_form)],
    ["Password Input", statusText(f.num_password_inputs)],
    ["Meta Refresh", statusText(f.has_meta_refresh)],
    ["JavaScript Redirect", statusText(f.has_javascript_redirect)],
  ];
}

function buildExplanation(data) {
  const percent = Math.round(data.phishing_probability * 100);
  if (data.model_source !== "phish360_tri_branch_cnn") {
    return "The Tri-Branch CNN could not complete all three inputs. Heuristic checks are not a calibrated probability, so no safety conclusion can be made. Manual review is required.";
  }
  if (data.risk_level === "phishing") {
    return `Phishing score ${percent}%. High risk: do not enter personal information or continue to this website.`;
  }
  if (data.risk_level === "suspicious") {
    return `Phishing score ${percent}%. The result is uncertain and requires manual review.`;
  }
  return `Phishing score ${percent}%. Few phishing patterns were detected, but this is not proof that the website is safe.`;
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function renderRiskFactors(rows) {
  clearChildren(riskFactors);
  rows.forEach(([feature, status]) => {
    const row = document.createElement("tr");
    const featureCell = document.createElement("td");
    const statusCell = document.createElement("td");

    featureCell.textContent = feature;
    statusCell.textContent = status;
    row.append(featureCell, statusCell);
    riskFactors.appendChild(row);
  });
}

function renderFeatureCards(data) {
  clearChildren(features);

  visibleFeatures.forEach(([key, title]) => {
    features.appendChild(createFeatureCard(title, data.features[key]));
  });
}

function createFeatureCard(title, value) {
  const card = document.createElement("article");
  const label = document.createElement("span");
  const strong = document.createElement("strong");

  card.className = "feature";
  label.textContent = title;
  strong.textContent = value ?? "";
  card.append(label, strong);
  return card;
}

function showError(message) {
  result.classList.remove("hidden", "phishing", "suspicious", "legitimate");
  result.classList.add("phishing");
  label.textContent = "error";
  phishingScore.textContent = "0%";
  modelSource.textContent = message;
  explanation.classList.add("hidden");
  features.classList.add("hidden");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const url = urlInput.value.trim();
  if (!url) return;

  submitButton.disabled = true;
  submitButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join(", ")
        : data.detail || "Request failed";
      throw new Error(detail);
    }

    const percent = Math.round(data.phishing_probability * 100);
    const limitedAnalysis = data.model_source !== "phish360_tri_branch_cnn";
    const labels = {
      phishing: "High phishing risk",
      suspicious: limitedAnalysis ? "Limited analysis - manual review" : "Suspicious - manual review",
      legitimate: "Low phishing likelihood",
    };

    result.classList.remove("hidden", "phishing", "suspicious", "legitimate");
    result.classList.add(data.risk_level);
    label.textContent = labels[data.risk_level] || data.risk_level;
    phishingScore.textContent = limitedAnalysis ? "N/A" : `${percent}%`;
    scoreLabel.textContent = limitedAnalysis ? "model score unavailable" : "model phishing score";
    modelSource.textContent = limitedAnalysis ? "Heuristic fallback (Tri-Branch unavailable)" : "Tri-Branch CNN";

    explanationText.textContent = buildExplanation(data);
    renderRiskFactors(buildRiskFactors(data));
    explanation.classList.remove("hidden");

    renderFeatureCards(data);
    features.classList.remove("hidden");
  } catch (error) {
    showError(error.message || "Unable to analyze this URL.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Analyze";
  }
});
