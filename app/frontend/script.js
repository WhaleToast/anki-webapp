document.addEventListener("DOMContentLoaded", () => {
	const button = document.getElementById("generateBtn");

	button.addEventListener("click", () => {
		const prompt = document.getElementById("prompt").value;
		generate(prompt);

	});
});

async function generate(prompt){
	const responseBox = document.getElementById("response");

	responseBox.textContent = "Generating...";

	// try {
	// 	const res = await fetch("http://localhost:8000/generate", {
	// 		method: "POST",
	// 		headers: { "Content-Type": "application/json" },
	// 		body: JSON.stringify({ prompt: prompt })
	// 	});
	//
	// 	const data = await res.json();
	// 	responseBox.textContent = data.response;
	// } catch (err) {
	// 	responseBox.textContent = "Error: " + err;
	// }

}
