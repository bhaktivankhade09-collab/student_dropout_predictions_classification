# Student Dropout Prediction — Streamlit UI

This folder contains your existing project artifacts plus an upgraded Streamlit interface.

## What changed in `app.py`
- Cleaner dashboard layout with three tabs: Predict outcome, Model performance, and Feature guide.
- Dropdowns for coded fields such as marital status, gender, attendance schedule, nationality, course, admission mode, and qualification.
- Yes/No selectors for binary features.
- Numeric controls for grades, semester counts, age, and economic indicators.
- Keeps the exact numeric feature codes and original feature order expected by the trained model.
- Shows the prediction and the model's estimated dropout score, when available.
- Shows your saved model-comparison metrics.

The labels are based on the UCI dataset documentation. For some high-cardinality coded fields such as parents' occupations, the app displays the code and points to the UCI data dictionary where a complete label mapping is not included in the app.

## Run on Windows
1. Extract the ZIP.
2. Open the extracted folder in File Explorer.
3. Click the address bar, type `cmd`, and press Enter.
4. Run:

```bat
pip install -r requirements.txt
streamlit run app.py
```

The ZIP includes your saved model, scaler, feature columns, dataset, and model comparison file, so you should not need to retrain the model. Streamlit will print a local URL, usually `http://localhost:8501`.

If `pip` is not recognized, try:
```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Important
- Keep all project files in the same folder.
- This is a student project prototype. Its prediction may be wrong and should not be used as the sole basis for decisions about students.
- The current project maps both `Graduate` and `Enrolled` to `Not Dropout`, so it does not separately predict graduation vs continued enrollment.
- Official dataset documentation: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
