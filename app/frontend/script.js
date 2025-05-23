import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";

let selectedFile = null;
let apiUrl = "http://45.148.31.50:9999/" 

document.addEventListener("DOMContentLoaded", () => {
	const fileInput = document.getElementById("pdfInput");
	const button = document.getElementById("generateBtn");
	const responseBox = document.getElementById("response")


	fileInput.addEventListener("change", () => {
		if (fileInput.files.length > 0) {
			selectedFile = fileInput.files[0];
			responseBox.textContent = `Selected file: ${selectedFile.name}`;
		}
	});

	button.addEventListener("click", async () => {
		if (!selectedFile) {
			responseBox.textContent = "Please select a PDF file first.";
			return;
		}

		const formData = new FormData();
		formData.append("file", selectedFile);
		
		responseBox.textContent = "Generating...";

		try {
			const res = await fetch(`${apiUrl}upload_pdf`, {
				method: "POST",
				body: formData
			});

			const data = await res.json()
			responseBox.innerHTML = marked.parse(data.response);
		} catch (err) {
			responseBox.textContent = "Error: " + err;
		}
	});
});

// async function generate(prompt){
// 	const responseBox = document.getElementById("response");
//
// 	responseBox.textContent = "Generating...";
// 	try {
// 		const res = await fetch("http://45.148.31.50:9999/generate", {
// 			method: "POST",
// 			headers: { "Content-Type": "application/json" },
// 			body: JSON.stringify({ prompt: prompt })
// 		});
//
// 		const data = await res.json();
// 		responseBox.innerHTML = marked.parse(data.response);
// 	} catch (err) {
// 		responseBox.textContent = "Error: " + err;
// 	
// 	}
// }
