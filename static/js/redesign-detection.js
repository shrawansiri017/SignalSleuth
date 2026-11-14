
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // COMMON FUNCTIONS
    // ==========================================

    function showMessage(element, message, isError = true) {
        if (!element) return;

        element.textContent = message;
        element.classList.remove("hidden", "error", "success");
        element.classList.add(isError ? "error" : "success");
    }

    function hideMessage(element) {
        if (element) {
            element.textContent = "";
            element.classList.add("hidden");
        }
    }

    async function getError(response) {
        const data = await response.json();

        if (data.limit_reached && data.register_url) {
            return `${data.error} Register here: ${window.location.origin}${data.register_url}`;
        }

        return data.error || "Something went wrong. Please try again.";
    }

    function showMediaResult(messageElement, result) {
        if (!messageElement) return;

        const prediction = document.createElement("strong");
        prediction.textContent =
            `${result.prediction} — ${result.confidence}% confidence`;

        const description = document.createElement("p");
        description.textContent =
            result.message || "AI analysis completed.";

        messageElement.replaceChildren(prediction, description);
        messageElement.classList.remove("hidden", "error");
        messageElement.classList.add("success");
    }

    // ==========================================
    // 1. FAKE NEWS DETECTION
    // ==========================================

    const newsText = document.getElementById("newsText");
    const charCount = document.getElementById("charCount");
    const analyzeBtn = document.getElementById("analyzeBtn");

    if (newsText && analyzeBtn) {

        newsText.addEventListener("input", () => {
            if (charCount) {
                charCount.textContent = `${newsText.value.length} characters`;
            }
        });

        analyzeBtn.addEventListener("click", async () => {

            const text = newsText.value.trim();

            const errorMessage = document.getElementById("errorMessage");
            const loading = document.getElementById("loading");
            const resultSection = document.getElementById("resultSection");

            hideMessage(errorMessage);

            if (!text) {
                showMessage(errorMessage, "Please enter a news article.");
                return;
            }

            analyzeBtn.disabled = true;

            if (loading) loading.classList.remove("hidden");
            if (resultSection) resultSection.classList.add("hidden");

            try {

                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ text })
                });

                if (!response.ok) {
                    throw new Error(await getError(response));
                }

                const result = await response.json();

                const predictionText =
                    document.getElementById("predictionText");

                const resultBadge =
                    document.getElementById("resultBadge");

                const confidenceValue =
                    document.getElementById("confidenceValue");

                const confidenceBar =
                    document.getElementById("confidenceBar");

                const explanationType =
                    document.getElementById("explanationType");

                const wordList =
                    document.getElementById("wordList");

                // Display prediction
                if (predictionText) {
                    predictionText.textContent = result.prediction;
                }

                if (resultBadge) {
                    resultBadge.textContent = result.prediction;

                    resultBadge.classList.toggle(
                        "fake",
                        result.prediction === "FAKE"
                    );
                }

                // Display confidence
                const confidence = Math.max(
                    0,
                    Math.min(100, Number(result.confidence) || 0)
                );

                if (confidenceValue) {
                    confidenceValue.textContent = `${confidence}%`;
                }

                if (confidenceBar) {
                    confidenceBar.style.width = `${confidence}%`;
                }

                if (explanationType) {
                    explanationType.textContent =
                        result.explanation_type || "";
                }

                // ==========================================
                // EXPLAINABLE AI
                // ==========================================

                if (wordList) {

                    wordList.innerHTML = "";

                    const article = text;

                    // Patterns to identify in the article.
                    // These are language observations, not fact verification.

                    const patterns = [
                        {
                            title: "Sensational or exaggerated claims",

                            description:
                                "The article contains dramatic language or extraordinary promises that may require additional evidence.",

                            regex:
                                /\b(breaking news|shocking|miracle|secret discovery|cure permanently|instantly|overnight|guaranteed)\b/gi
                        },

                        {
                            title: "Unsupported medical claims",

                            description:
                                "The article makes health-related claims that should be checked against reliable medical evidence.",

                            regex:
                                /\b(cure|cancer|diabetes|heart disease|without medicine|stop taking medicines|permanently)\b/gi
                        },

                        {
                            title: "Unverified or anonymous information",

                            description:
                                "The article uses uncertain attribution or claims without clearly identifying their source.",

                            regex:
                                /\b(allegedly|reportedly|sources claim|experts say|it is believed|rumours?)\b/gi
                        },

                        {
                            title: "Conspiracy-style language",

                            description:
                                "The article suggests that information is being hidden or deliberately suppressed.",

                            regex:
                                /\b(hide from the public|government is hiding|cover[- ]?up|they don't want you to know|secretly)\b/gi
                        },

                        {
                            title: "Urgent sharing language",

                            description:
                                "The article pressures readers to share information immediately rather than encouraging verification.",

                            regex:
                                /\b(share this|before it gets deleted|share immediately|spread the word|don't ignore)\b/gi
                        }
                    ];

                    const matchedPatterns = [];

                    patterns.forEach(pattern => {

                        const matches = article.match(pattern.regex);

                        if (matches) {
                            matchedPatterns.push({
                                title: pattern.title,
                                description: pattern.description,

                                phrases: [
                                    ...new Set(
                                        matches.map(word => word.toLowerCase())
                                    )
                                ]
                            });
                        }
                    });

                    // Main explanation heading
                    const reasonHeading =
                        document.createElement("h3");

                    reasonHeading.textContent =
                        "Why was this article flagged?";

                    wordList.appendChild(reasonHeading);

                    // Display detected language patterns
                    if (matchedPatterns.length > 0) {

                        matchedPatterns.forEach((item, index) => {

                            const card =
                                document.createElement("div");

                            card.style.cssText = `
                                padding: 14px;
                                margin: 10px 0;
                                background: #EAF1FF;
                                border: 1px solid #BFD3FF;
                                border-radius: 10px;
                                color: #1743B5;
                            `;

                            const title =
                                document.createElement("strong");

                            title.textContent =
                                `${index + 1}. ${item.title}`;

                            const description =
                                document.createElement("p");

                            description.textContent =
                                item.description;

                            const phrases =
                                document.createElement("p");

                            phrases.textContent =
                                "Detected phrases: " +
                                item.phrases.join(", ");

                            card.append(
                                title,
                                description,
                                phrases
                            );

                            wordList.appendChild(card);
                        });

                    } else {

                        const message =
                            document.createElement("p");

                        message.textContent =
                            "No specific language patterns were detected. The model may have used other word patterns to make its prediction.";

                        wordList.appendChild(message);
                    }

                    // Existing model word contribution scores
                    const scoreHeading =
                        document.createElement("h3");

                    scoreHeading.textContent =
                        "Model Word Contributions";

                    scoreHeading.style.marginTop = "22px";

                    wordList.appendChild(scoreHeading);

                    const words = result.explanation || [];

                    if (words.length > 0) {

                        words.forEach(item => {

                            const chip =
                                document.createElement("div");

                            chip.textContent =
                                `${item.word} — Score: ${Number(item.score).toFixed(4)}`;

                            chip.style.cssText = `
                                display: block;
                                padding: 12px;
                                margin: 8px 0;
                                background: #EAF1FF;
                                color: #1743B5;
                                border: 1px solid #BFD3FF;
                                border-radius: 8px;
                                font-size: 14px;
                                font-weight: 600;
                            `;

                            wordList.appendChild(chip);
                        });

                    } else {

                        const noWords =
                            document.createElement("p");

                        noWords.textContent =
                            "No model word contributions are available.";

                        wordList.appendChild(noWords);
                    }

                    // Disclaimer
                    const disclaimer =
                        document.createElement("p");

                    disclaimer.textContent =
                        "These language patterns are observations, not verified facts. Word contribution scores show model influence, not proof of misinformation. Independent fact-checking is required to establish whether claims are true.";

                    disclaimer.style.cssText = `
                        font-size: 12px;
                        color: #64748B;
                        margin-top: 14px;
                        line-height: 1.6;
                    `;

                    wordList.appendChild(disclaimer);
                }

                // Show results
                if (resultSection) {
                    resultSection.classList.remove("hidden");
                    resultSection.style.display = "block";
                    resultSection.style.visibility = "visible";
                    resultSection.style.opacity = "1";
                }

            } catch (error) {

                showMessage(errorMessage, error.message);

            } finally {

                analyzeBtn.disabled = false;

                if (loading) {
                    loading.classList.add("hidden");
                }
            }
        });
    }

    // ==========================================
    // 2. IMAGE DEEPFAKE DETECTION
    // ==========================================

    const imageInput = document.getElementById("imageInput");

    if (imageInput) {

        const chooseButton =
            document.getElementById("chooseImageBtn");

        const uploadArea =
            document.getElementById("uploadArea");

        const previewContainer =
            document.getElementById("imagePreviewContainer");

        const preview =
            document.getElementById("imagePreview");

        const fileName =
            document.getElementById("imageFileName");

        const removeButton =
            document.getElementById("removeImageBtn");

        const inspectButton =
            document.getElementById("inspectImageBtn");

        const message =
            document.getElementById("imageMessage");

        let selectedFile = null;

        function selectImage(file) {

            if (!file) return;

            const allowed = ["jpg", "jpeg", "png", "webp"];

            const extension =
                file.name.split(".").pop().toLowerCase();

            if (!allowed.includes(extension)) {
                showMessage(
                    message,
                    "Please select a JPG, JPEG, PNG or WEBP image."
                );
                return;
            }

            selectedFile = file;

            hideMessage(message);

            preview.src = URL.createObjectURL(file);

            fileName.textContent = file.name;

            previewContainer.classList.remove("hidden");
            uploadArea.classList.add("hidden");
        }

        chooseButton?.addEventListener("click", () => {
            imageInput.click();
        });

        imageInput.addEventListener("change", () => {
            selectImage(imageInput.files[0]);
        });

        ["dragenter", "dragover"].forEach(eventName => {

            uploadArea?.addEventListener(eventName, event => {

                event.preventDefault();

                uploadArea.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {

            uploadArea?.addEventListener(eventName, event => {

                event.preventDefault();

                uploadArea.classList.remove("drag-over");
            });
        });

        uploadArea?.addEventListener("drop", event => {

            selectImage(event.dataTransfer.files[0]);
        });

        removeButton?.addEventListener("click", () => {

            selectedFile = null;

            imageInput.value = "";
            preview.src = "";

            previewContainer.classList.add("hidden");
            uploadArea.classList.remove("hidden");

            hideMessage(message);
        });

        inspectButton?.addEventListener("click", async () => {

            if (!selectedFile) {
                showMessage(message, "Please select an image first.");
                return;
            }

            const formData = new FormData();

            formData.append("image", selectedFile);

            inspectButton.disabled = true;
            inspectButton.textContent = "Analyzing...";

            hideMessage(message);

            try {

                const response = await fetch("/api/inspect-image", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(await getError(response));
                }

                const result = await response.json();

                showMediaResult(message, result);

            } catch (error) {

                showMessage(message, error.message);

            } finally {

                inspectButton.disabled = false;
                inspectButton.textContent = "Analyze Image →";
            }
        });
    }

    // ==========================================
    // 3. VIDEO DEEPFAKE DETECTION
    // ==========================================

    const videoInput = document.getElementById("videoInput");

    if (videoInput) {

        const chooseButton =
            document.getElementById("chooseVideoBtn");

        const uploadArea =
            document.getElementById("videoUploadArea");

        const previewContainer =
            document.getElementById("videoPreviewContainer");

        const preview =
            document.getElementById("videoPreview");

        const fileName =
            document.getElementById("videoFileName");

        const removeButton =
            document.getElementById("removeVideoBtn");

        const inspectButton =
            document.getElementById("inspectVideoBtn");

        const message =
            document.getElementById("videoMessage");

        let selectedFile = null;

        function selectVideo(file) {

            if (!file) return;

            const allowed = [
                "mp4",
                "mov",
                "avi",
                "mkv",
                "webm"
            ];

            const extension =
                file.name.split(".").pop().toLowerCase();

            if (!allowed.includes(extension)) {
                showMessage(message, "Please select a supported video file.");
                return;
            }

            selectedFile = file;

            hideMessage(message);

            preview.src = URL.createObjectURL(file);
            preview.load();

            fileName.textContent = file.name;

            previewContainer.classList.remove("hidden");
            uploadArea.classList.add("hidden");
        }

        chooseButton?.addEventListener("click", () => {
            videoInput.click();
        });

        videoInput.addEventListener("change", () => {
            selectVideo(videoInput.files[0]);
        });

        ["dragenter", "dragover"].forEach(eventName => {

            uploadArea?.addEventListener(eventName, event => {

                event.preventDefault();

                uploadArea.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {

            uploadArea?.addEventListener(eventName, event => {

                event.preventDefault();

                uploadArea.classList.remove("drag-over");
            });
        });

        uploadArea?.addEventListener("drop", event => {

            selectVideo(event.dataTransfer.files[0]);
        });

        removeButton?.addEventListener("click", () => {

            selectedFile = null;

            videoInput.value = "";

            preview.pause();
            preview.removeAttribute("src");
            preview.load();

            previewContainer.classList.add("hidden");
            uploadArea.classList.remove("hidden");

            hideMessage(message);
        });

        inspectButton?.addEventListener("click", async () => {

            if (!selectedFile) {
                showMessage(message, "Please select a video first.");
                return;
            }

            const formData = new FormData();

            formData.append("video", selectedFile);

            inspectButton.disabled = true;
            inspectButton.textContent = "Analyzing...";

            hideMessage(message);

            try {

                const response = await fetch("/api/inspect-video", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(await getError(response));
                }

                const result = await response.json();

                showMediaResult(message, result);

            } catch (error) {

                showMessage(message, error.message);

            } finally {

                inspectButton.disabled = false;
                inspectButton.textContent = "Analyze Video →";
            }
        });
    }

});