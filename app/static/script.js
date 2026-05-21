const form = document.querySelector("#predict-form");
const urlInput = document.querySelector("#url");
const submitButton = form.querySelector("button");
const result = document.querySelector("#result");
const label = document.querySelector("#label");
const confidence = document.querySelector("#confidence");
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

function riskLevel(value, highWhenDetected = true) {
  if (!value) return highWhenDetected ? "Safe" : "Low";
  return highWhenDetected ? "High" : "Medium";
}

function statusText(value) {
  return value ? "Detected" : "Not Found";
}

function buildRiskFactors(data) {
  const f = data.features || {};
  return [
    ["Public Hosting", statusText(f.has_public_hosting_platform), riskLevel(f.has_public_hosting_platform)],
    ["Brand Impersonation", statusText(f.has_brand_impersonation), riskLevel(f.has_brand_impersonation)],
    ["URL Shortener", statusText(f.has_url_shortener), riskLevel(f.has_url_shortener)],
    ["Redirect Parameter", statusText(f.has_redirect_param), riskLevel(f.has_redirect_param)],
    ["Embedded URL", statusText(f.has_embedded_url), riskLevel(f.has_embedded_url)],
    ["Risky TLD", statusText(f.has_suspicious_tld), riskLevel(f.has_suspicious_tld)],
    ["IP Address", statusText(f.has_ip), riskLevel(f.has_ip)],
    ["At Symbol", statusText(f.has_at), riskLevel(f.has_at)],
    ["Long Token", statusText(f.has_long_token), riskLevel(f.has_long_token)],
    ["Encoded Characters", statusText(f.has_encoded_chars), riskLevel(f.has_encoded_chars, false)],
    ["HTML Available", statusText(f.html_available), f.html_available ? "Low" : "Medium"],
    ["Login Form", statusText(f.has_login_form), riskLevel(f.has_login_form)],
    ["Password Input", statusText(f.num_password_inputs), riskLevel(f.num_password_inputs)],
    ["Meta Refresh", statusText(f.has_meta_refresh), riskLevel(f.has_meta_refresh)],
    ["JavaScript Redirect", statusText(f.has_javascript_redirect), riskLevel(f.has_javascript_redirect)],
  ];
}

function buildExplanation(data) {
  const f = data.features || {};
  const reasons = [];

  if (f.has_public_hosting_platform) reasons.push("website su dung public hosting");
  if (f.has_brand_impersonation) reasons.push("URL co dau hieu gia mao brand/crypto");
  if (f.has_url_shortener) reasons.push("URL dung shortener");
  if (f.has_redirect_param) reasons.push("URL co tham so redirect");
  if (f.has_embedded_url) reasons.push("URL long URL khac ben trong");
  if (f.has_ip) reasons.push("URL dung dia chi IP");
  if (f.has_at) reasons.push("URL co ky tu @");
  if (f.has_long_token) reasons.push("URL co token dai bat thuong");
  if (f.has_login_form) reasons.push("HTML co form dang nhap");
  if (f.num_password_inputs) reasons.push("HTML co password input");
  if (f.has_meta_refresh) reasons.push("HTML co meta refresh");
  if (f.has_javascript_redirect) reasons.push("HTML co JavaScript redirect");

  if (data.label === "phishing") {
    const detail = reasons.length ? reasons.join(", ") : "xac suat phishing vuot nguong cua mo hinh";
    return `Mo hinh danh gia phishing chu yeu do ${detail}.`;
  }

  if (reasons.length) {
    return `Mo hinh van phan loai legitimate, nhung can luu y: ${reasons.join(", ")}.`;
  }

  return "Mo hinh phan loai legitimate vi chua thay cac dau hieu rui ro manh trong URL.";
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function renderRiskFactors(rows) {
  clearChildren(riskFactors);
  rows.forEach(([feature, status, risk]) => {
    const row = document.createElement("tr");
    const featureCell = document.createElement("td");
    const statusCell = document.createElement("td");
    const riskCell = document.createElement("td");
    const riskBadge = document.createElement("span");

    featureCell.textContent = feature;
    statusCell.textContent = status;
    riskBadge.className = `risk ${risk.toLowerCase()}`;
    riskBadge.textContent = risk;
    riskCell.appendChild(riskBadge);
    row.append(featureCell, statusCell, riskCell);
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
  result.classList.remove("hidden", "phishing", "legitimate");
  result.classList.add("phishing");
  label.textContent = "error";
  confidence.textContent = "0%";
  modelSource.textContent = message;
  explanation.classList.add("hidden");
  features.classList.add("hidden");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const url = urlInput.value.trim();
  if (!url) return;

  submitButton.disabled = true;
  submitButton.textContent = "Dang phan tich";

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

    const percent = Math.round(data.confidence * 100);

    result.classList.remove("hidden", "phishing", "legitimate");
    result.classList.add(data.label);
    label.textContent = data.label;
    confidence.textContent = `${percent}%`;
    modelSource.textContent = data.model_source;

    explanationText.textContent = buildExplanation(data);
    renderRiskFactors(buildRiskFactors(data));
    explanation.classList.remove("hidden");

    renderFeatureCards(data);
    features.classList.remove("hidden");
  } catch (error) {
    showError(error.message || "Khong the phan tich URL nay.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Phan tich";
  }
});
