const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const statusCard = document.getElementById("status-card");
const statusText = document.getElementById("status-text");
const submitButton = document.getElementById("submit-button");
const resultSection = document.getElementById("result");
const transcriptOutput = document.getElementById("transcript-output");
const downloadLink = document.getElementById("download-link");

function setStatus(message, tone = "muted") {
  statusText.textContent = message;
  statusCard.className = `status-card ${tone}`;
}

function renderSegments(segments) {
  return segments.map(([start, end, text, speaker]) => {
    const prefix = speaker ? `${speaker}: ` : "";
    return `[${start.toFixed(2)}s -> ${end.toFixed(2)}s] ${prefix}${text}`;
  }).join("\n");
}

function attachFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setStatus(`Ready to transcribe ${file.name}.`);
}

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    attachFile(file);
  }
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) {
    setStatus(`Ready to transcribe ${file.name}.`);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = fileInput.files;
  if (!file) {
    setStatus("Choose a file first.", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus(`Uploading ${file.name}...`);
  resultSection.classList.add("hidden");
  downloadLink.hidden = true;

  const data = new FormData(form);

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }

    transcriptOutput.textContent = renderSegments(payload.segments);
    resultSection.classList.remove("hidden");
    if (payload.download_url) {
      downloadLink.href = payload.download_url;
      downloadLink.hidden = false;
    }
    setStatus(`Completed in ${payload.processing_time.toFixed(1)} seconds.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

fetch("/api/status")
  .then((response) => response.json())
  .then((payload) => {
    const model = payload.model?.model_size || "unknown";
    setStatus(`Server ready. Model: ${model}.`);
  })
  .catch(() => {
    setStatus("Server status unavailable. Start the model server first.", "error");
  });
