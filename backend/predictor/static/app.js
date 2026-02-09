document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("churnForm");
    const resultDiv = document.getElementById("result");
    const confidenceBar = document.getElementById("confidenceBar");

    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        // Prepare payload
        const payload = {
            tenure: parseInt(document.getElementById("tenure").value),
            MonthlyCharges: parseFloat(document.getElementById("monthlyCharges").value),
            TotalCharges: parseFloat(document.getElementById("totalCharges").value),
            Contract: parseInt(document.getElementById("contract").value),
            PaymentMethod: parseInt(document.getElementById("paymentMethod").value)
        };

        // Frontend validation
        if (
            isNaN(payload.tenure) || payload.tenure < 0 ||
            isNaN(payload.MonthlyCharges) || payload.MonthlyCharges <= 0 ||
            isNaN(payload.TotalCharges) || payload.TotalCharges <= 0
        ) {
            resultDiv.className = "alert alert-warning mt-4";
            resultDiv.innerHTML = "❌ Please enter valid positive values.";
            resultDiv.classList.remove("d-none");
            return;
        }

        resultDiv.classList.add("d-none");

        try {
            // Use relative URL for production compatibility
            const response = await fetch("/predict/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                if (response.status === 503) {
                    throw new Error("Service temporarily unavailable");
                }
                throw new Error(data.error || "Server error");
            }

            // Show result
            resultDiv.classList.remove("d-none");

            const probabilityText = `Probability: ${data.churn_probability}%`;

            if (data.churn_prediction === "Yes") {
                resultDiv.className = "alert alert-danger mt-4";
                resultDiv.innerHTML = `
                    ⚠️ <strong>Customer is likely to churn</strong><br>
                    ${probabilityText}
                `;
                confidenceBar.className = "progress-bar bg-danger";
            } else {
                resultDiv.className = "alert alert-success mt-4";
                resultDiv.innerHTML = `
                    ✅ <strong>Customer is likely to stay</strong><br>
                    ${probabilityText}
                `;
                confidenceBar.className = "progress-bar bg-success";
            }

            // Update confidence bar
            confidenceBar.style.width = `${data.churn_probability}%`;
            confidenceBar.innerText = `${data.churn_probability}%`;

            // Show top influencing factors
            if (data.top_factors && data.top_factors.length > 0) {
                const reasonsList = data.top_factors
                    .map(factor => `<li>${factor}</li>`)
                    .join("");

                resultDiv.innerHTML += `
                    <hr>
                    <strong>Top influencing factors:</strong>
                    <ul>${reasonsList}</ul>
                `;
            }

        } catch (error) {
            resultDiv.className = "alert alert-warning mt-4";
            resultDiv.innerHTML = `❌ ${error.message || "Error connecting to server."}`;
            resultDiv.classList.remove("d-none");
        }
    });

});
