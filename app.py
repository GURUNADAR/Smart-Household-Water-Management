import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = "models/water_demand_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"

if not os.path.exists(MODEL_PATH):
    import train_model
    train_model.train()

pipeline = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    model_metadata = json.load(f)

BENCHMARK_LPCD = 135.0

def generate_recommendations(inputs, daily_demand, per_capita):
    tips = []
    if inputs['washing_machine'] == 1:
        tips.append("Run the washing machine with full loads only to save up to 40 litres per cycle.")
    if inputs['garden'] == 1:
        tips.append("Water outdoor plants in the early morning or late evening to avoid evaporative loss.")
    if inputs['water_source'] == 'Tanker Supplemental':
        tips.append("Install an overflow alarm on overhead storage tanks to prevent refill wastage.")
    if inputs['rooms'] >= 4:
        tips.append("Perform a monthly check for dripping taps or cistern leaks across all bathrooms.")
    if per_capita > BENCHMARK_LPCD:
        diff = round(per_capita - BENCHMARK_LPCD, 1)
        tips.append(f"Per-person demand ({per_capita} L) exceeds the Indian national norm (135 L) by {diff} L. Consider installing tap aerators.")
    else:
        tips.append("Your per-person water consumption aligns efficiently within national urban benchmarks.")
    return tips

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metadata=model_metadata, benchmark=BENCHMARK_LPCD)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        household_size = max(1, min(int(request.form.get("household_size", 4)), 20))
        dwelling_type = request.form.get("dwelling_type", "Apartment")
        rooms = max(1.0, min(float(request.form.get("rooms", 3)), 15.0))
        washing_machine = int(request.form.get("washing_machine", 0))
        garden = int(request.form.get("garden", 0))
        water_source = request.form.get("water_source", "Municipal Piped")

        input_df = pd.DataFrame([{
            'household_size': household_size,
            'dwelling_type': dwelling_type,
            'rooms': rooms,
            'washing_machine': washing_machine,
            'garden': garden,
            'water_source': water_source
        }])

        raw_pred = pipeline.predict(input_df)[0]
        daily_litres = max(80.0, round(float(raw_pred), 1))
        monthly_litres = round(daily_litres * 30.0, 1)
        per_capita = round(daily_litres / household_size, 1)

        scenarios = {
            "5%": {"daily_saved": round(daily_litres * 0.05, 1), "monthly_saved": round(monthly_litres * 0.05, 1), "target_daily": round(daily_litres * 0.95, 1)},
            "10%": {"daily_saved": round(daily_litres * 0.10, 1), "monthly_saved": round(monthly_litres * 0.10, 1), "target_daily": round(daily_litres * 0.90, 1)},
            "15%": {"daily_saved": round(daily_litres * 0.15, 1), "monthly_saved": round(monthly_litres * 0.15, 1), "target_daily": round(daily_litres * 0.85, 1)}
        }

        benchmark_comparison = {
            "benchmark_lpcd": BENCHMARK_LPCD,
            "status": "Above Benchmark" if per_capita > BENCHMARK_LPCD else "Within Benchmark",
            "difference": round(per_capita - BENCHMARK_LPCD, 1)
        }

        recommendations = generate_recommendations({
            'washing_machine': washing_machine, 'garden': garden,
            'water_source': water_source, 'rooms': rooms
        }, daily_litres, per_capita)

        return render_template(
            "result.html",
            inputs=request.form, daily_litres=daily_litres, monthly_litres=monthly_litres,
            per_capita=per_capita, benchmark=benchmark_comparison,
            scenarios=scenarios, recommendations=recommendations, metadata=model_metadata
        )
    except Exception as e:
        return render_template("index.html", error=str(e), metadata=model_metadata, benchmark=BENCHMARK_LPCD)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
