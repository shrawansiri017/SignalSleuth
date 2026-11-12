
const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");

newsText.addEventListener("input", () => {
    charCount.textContent = `${newsText.value.length} characters`;
});

async function analyzeNews() {
    const text = newsText.value.trim();

    const loading = document.getElementById("loading");
    const errorMessage = document.getElementById("errorMessage");
    const resultSection = document.getElementById("resultSection");

    errorMessage.classList.add("hidden");
    resultSection.classList.add("hidden");

    if (!text) {
        errorMessage.textContent = "Please paste a news article before analyzing.";
        errorMessage.classList.remove("hidden");
        return;
    }

    analyzeBtn.disabled = true;
    loading.classList.remove("hidden");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Analysis failed.");
        }

        displayResult(result);

    } catch (error) {
        errorMessage.textContent = error.message;
        errorMessage.classList.remove("hidden");

    } finally {
        loading.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
}

function displayResult(result) {
    const resultSection = document.getElementById("resultSection");
    const predictionText = document.getElementById("predictionText");
    const resultBadge = document.getElementById("resultBadge");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");
    const explanationType = document.getElementById("explanationType");
    const wordList = document.getElementById("wordList");

    const isFake = result.prediction === "FAKE";

    predictionText.textContent = isFake ? "FAKE NEWS" : "REAL NEWS";
    resultBadge.textContent = result.prediction;

    resultBadge.classList.toggle("fake", isFake);

    confidenceValue.textContent = `${result.confidence}%`;
    confidenceBar.style.width = `${result.confidence}%`;

    explanationType.textContent = result.explanation_type || "";

    wordList.innerHTML = "";

    if (result.explanation && result.explanation.length > 0) {
        result.explanation.forEach(item => {
            const chip = document.createElement("span");
            chip.className = "word-chip";
            chip.textContent = `${item.word} · ${item.score}`;
            wordList.appendChild(chip);
        });
    } else {
        wordList.textContent = "No significant contributing words found.";
    }

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

// IMAGE UPLOAD AND PREVIEW

const imageInput = document.getElementById("imageInput");
const chooseImageBtn = document.getElementById("chooseImageBtn");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");
const imageFileName = document.getElementById("imageFileName");
const removeImageBtn = document.getElementById("removeImageBtn");
const inspectImageBtn = document.getElementById("inspectImageBtn");
const imageMessage = document.getElementById("imageMessage");

let selectedImage = null;

chooseImageBtn.addEventListener("click", () => {
    imageInput.click();
});

imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
        showImagePreview(imageInput.files[0]);
    }
});

function showImagePreview(file) {
    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        showImageMessage("Please select a JPG, PNG, or WEBP image.");
        imageInput.value = "";
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showImageMessage("Image size must be less than 10 MB.");
        imageInput.value = "";
        return;
    }

    selectedImage = file;

    imagePreview.src = URL.createObjectURL(file);
    imageFileName.textContent = file.name;

    previewContainer.classList.remove("hidden");
    imageMessage.classList.add("hidden");
}

removeImageBtn.addEventListener("click", () => {
    if (imagePreview.src.startsWith("blob:")) {
        URL.revokeObjectURL(imagePreview.src);
    }

    selectedImage = null;
    imageInput.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("hidden");
    imageMessage.classList.add("hidden");
});

uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        showImagePreview(file);
    }
});

function showImageMessage(message) {
    imageMessage.textContent = message;
    imageMessage.classList.remove("hidden");
}


inspectImageBtn.addEventListener("click", async () => {
    if (!selectedImage) {
        showImageMessage("Please choose an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);

    inspectImageBtn.disabled = true;
    inspectImageBtn.textContent = "Inspecting image...";

    try {
        const response = await fetch("/api/inspect-image", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Image inspection failed.");
        }

        showImageMessage(
    `AI ANALYSIS COMPLETE\n\n` +
    `Prediction: ${result.prediction}\n` +
    `Confidence: ${result.confidence}%\n\n` +
    `${result.message}`
);

    } catch (error) {
        showImageMessage(error.message);

    } finally {
        inspectImageBtn.disabled = false;
        inspectImageBtn.textContent = "Inspect Image →";
    }
});

// VIDEO DEEPFAKE DETECTION

const videoInput = document.getElementById("videoInput");
const chooseVideoBtn = document.getElementById("chooseVideoBtn");
const videoUploadArea = document.getElementById("videoUploadArea");
const videoPreviewContainer = document.getElementById("videoPreviewContainer");
const videoPreview = document.getElementById("videoPreview");
const videoFileName = document.getElementById("videoFileName");
const removeVideoBtn = document.getElementById("removeVideoBtn");
const inspectVideoBtn = document.getElementById("inspectVideoBtn");
const videoMessage = document.getElementById("videoMessage");

let selectedVideo = null;

chooseVideoBtn.addEventListener("click", () => videoInput.click());

videoInput.addEventListener("change", () => {
    if (videoInput.files.length > 0) {
        showVideoPreview(videoInput.files[0]);
    }
});


function showVideoPreview(file) {
    const allowedExtensions = [
        "mp4", "mov", "avi", "mkv", "webm"
    ];

    const extension = file.name.split(".").pop().toLowerCase();

    if (!allowedExtensions.includes(extension)) {
        showVideoMessage("Please select an MP4, MOV, AVI, MKV, or WEBM video.");
        videoInput.value = "";
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showVideoMessage("Video size must be less than 50 MB.");
        videoInput.value = "";
        return;
    }

    selectedVideo = file;
    videoPreview.src = URL.createObjectURL(file);
    videoFileName.textContent = file.name;

    videoPreviewContainer.classList.remove("hidden");
    videoMessage.classList.add("hidden");
}


removeVideoBtn.addEventListener("click", () => {
    if (videoPreview.src.startsWith("blob:")) {
        URL.revokeObjectURL(videoPreview.src);
    }

    selectedVideo = null;
    videoInput.value = "";
    videoPreview.src = "";

    videoPreviewContainer.classList.add("hidden");
    videoMessage.classList.add("hidden");
});

videoUploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    videoUploadArea.classList.add("drag-over");
});

videoUploadArea.addEventListener("dragleave", () => {
    videoUploadArea.classList.remove("drag-over");
});

videoUploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    videoUploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        showVideoPreview(file);
    }
});

function showVideoMessage(message) {
    videoMessage.textContent = message;
    videoMessage.classList.remove("hidden");
}

inspectVideoBtn.addEventListener("click", async () => {
    if (!selectedVideo) {
        showVideoMessage("Please choose a video first.");
        return;
    }

    const formData = new FormData();
    formData.append("video", selectedVideo);

    inspectVideoBtn.disabled = true;
    inspectVideoBtn.textContent = "Analyzing video...";

    showVideoMessage("AI is analyzing the video. Please wait...");

    try {
        const response = await fetch("/api/inspect-video", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Video analysis failed.");
        }

        showVideoMessage(
            `ANALYSIS COMPLETE\n\n` +
            `Prediction: ${result.prediction}\n` +
            `Confidence: ${result.confidence}%\n\n` +
            `${result.message}`
        );

    } catch (error) {
        showVideoMessage(error.message);
    } finally {
        inspectVideoBtn.disabled = false;
        inspectVideoBtn.textContent = "Analyze Video →";
    }
});
