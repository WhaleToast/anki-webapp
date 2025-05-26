import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";

let selectedFile = null;
let currentDeckId = null; // Store the deck ID for downloads
let apiUrl = "http://localhost:9999/";

document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("pdfInput");
    const generateBtn = document.getElementById("generateBtn");
    const downloadBtn = document.getElementById("downloadBtn"); // Add this button to your HTML
    const responseBox = document.getElementById("response");

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            selectedFile = fileInput.files[0];
            responseBox.textContent = `Selected file: ${selectedFile.name}`;
            // Hide download button when new file is selected
            if (downloadBtn) downloadBtn.style.display = "none";
            currentDeckId = null;
        }
    });

    generateBtn.addEventListener("click", async () => {
        if (downloadBtn) downloadBtn.style.display = "none";
        if (!selectedFile) {
            responseBox.textContent = "Please select a PDF file first.";
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);
        
        responseBox.textContent = "Generating...";
        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";

        try {
            const res = await fetch(`${apiUrl}upload_pdf`, {
                method: "POST",
                body: formData
            });
            
            const data = await res.json();
            
            // Display the response
            responseBox.innerHTML = "Success!" 
            
            // Store deck ID and show download button if deck was created
            if (data.deck_id) {
                currentDeckId = data.deck_id;
                if (downloadBtn) {
                    downloadBtn.style.display = "inline-block";
                }
            }
            
        } catch (err) {
            responseBox.textContent = "Error: " + err;
        } finally {
            // Reset button state
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Cards";
        }
    });

    // Download functionality
    if (downloadBtn) {
        downloadBtn.addEventListener("click", async () => {
            if (!currentDeckId) {
                alert("No deck available for download. Please generate cards first.");
                return;
            }

            downloadBtn.disabled = true;
            downloadBtn.textContent = "Downloading...";

            try {
                const response = await fetch(`${apiUrl}api/download-deck/${currentDeckId}`);
                
                if (!response.ok) {
                    throw new Error(`Download failed: ${response.statusText}`);
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `anki-deck-${currentDeckId}.apkg`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                // Clean up
                URL.revokeObjectURL(url);
                
            } catch (error) {
                alert('Download failed. Please try again.');
                console.error(error);
            } finally {
                downloadBtn.disabled = false;
                downloadBtn.textContent = "Download Deck";
            }
        });
    }
});
