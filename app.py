

import os
import csv
import warnings
from datetime import datetime
from threading import Lock

import numpy as np
import joblib
from flask import Flask, jsonify, request, send_file, make_response
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Cancer_Data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'cancer_ensemble_model.pkl')
FEATURE_NAMES = [
    'worst_texture',
    'se_radius',
    'worst_symmetry',
    'mean_concave_points',
    'worst_concavity',
    'se_area',
    'worst_radius',
    'worst_concave_points',
]


FEATURE_TO_CSV = {
    'worst_texture': 'texture_worst',
    'se_radius': 'radius_se',
    'worst_symmetry': 'symmetry_worst',
    'mean_concave_points': 'concave points_mean',
    'worst_concavity': 'concavity_worst',
    'se_area': 'area_se',
    'worst_radius': 'radius_worst',
    'worst_concave_points': 'concave points_worst',
}

CLASS_NAMES = ['Benign', 'Malignant']
CLASS_MAP = {'M': 1, 'B': 0}

history_lock = Lock()
PREDICTION_HISTORY = []

print("[INFO] Loading trained ensemble model...")
try:
    model_bundle = joblib.load(MODEL_PATH)
    XGB_MODEL = model_bundle['xgboost']
    LGB_MODEL = model_bundle['lightgbm']
    CAT_MODEL = model_bundle['catboost']
    SCALER = model_bundle['scaler']
    MODEL_FEATURE_COLUMNS = model_bundle['feature_columns']
    print(f"[INFO] Model loaded successfully. Features: {MODEL_FEATURE_COLUMNS}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("[HINT] Run 'python train_model.py' first to train and save the model.")
    raise

N_FEATURES = len(FEATURE_NAMES)

def read_csv_rows():
    """Read all rows from Cancer_Data.csv as list of dicts."""
    rows = []
    with open(CSV_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_feature_stats():
    """Compute per-feature min/max/mean/step from Cancer_Data.csv."""
    rows = read_csv_rows()
    stats = {}
    for feat_name in FEATURE_NAMES:
        csv_col = FEATURE_TO_CSV[feat_name]
        values = []
        for row in rows:
            try:
                values.append(float(row[csv_col]))
            except (ValueError, KeyError):
                continue
        if values:
            vmin = min(values)
            vmax = max(values)
            vmean = sum(values) / len(values)
            vstep = (vmax - vmin) / 100.0 if vmax > vmin else 0.01
            stats[feat_name] = {
                'min': round(vmin, 6),
                'max': round(vmax, 6),
                'mean': round(vmean, 6),
                'step': round(vstep, 6),
            }
    return stats


def get_random_sample():
    """Draw a random real patient row from Cancer_Data.csv."""
    rows = read_csv_rows()
    if not rows:
        return None
    row = np.random.choice(rows)
    features = []
    for feat_name in FEATURE_NAMES:
        csv_col = FEATURE_TO_CSV[feat_name]
        try:
            features.append(float(row[csv_col]))
        except (ValueError, KeyError):
            features.append(0.0)
    true_dx = row.get('diagnosis', 'B')
    return {
        'features': features,
        'true_diagnosis': true_dx,
        'true_class': CLASS_MAP.get(true_dx, 0),
    }


FEATURE_STATS = compute_feature_stats()
print(f"[INFO] Feature stats computed for {len(FEATURE_STATS)} features")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': 'Ensemble (XGBoost + LightGBM + CatBoost)',
        'features': N_FEATURES,
        'classes': CLASS_NAMES,
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Return feature names and statistics for slider configuration."""
    return jsonify({
        'feature_names': FEATURE_NAMES,
        'feature_stats': FEATURE_STATS,
    })


@app.route('/random_sample', methods=['GET'])
def random_sample():
    """Return a real patient sample from Cancer_Data.csv."""
    sample = get_random_sample()
    if sample is None:
        return jsonify({'error': 'No data available'}), 500
    return jsonify(sample)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict malignancy from 8 features.
    Body: { name, age, id, features: [8 floats] }
    Returns: { patient, prediction }
    """
    try:
        data = request.get_json(force=True)
        name = data.get('name', 'Unknown')
        age = data.get('age', 0)
        patient_id = data.get('id', '')
        features = data.get('features', [])

        if len(features) != N_FEATURES:
            return jsonify({
                'error': f'Expected {N_FEATURES} features, got {len(features)}'
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        xgb_prob = XGB_MODEL.predict_proba(X_scaled)[0, 1]
        lgb_prob = LGB_MODEL.predict_proba(X_scaled)[0, 1]
        cat_prob = CAT_MODEL.predict_proba(X_scaled)[0, 1]

        ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / 3.0

        pred_class = 1 if ensemble_prob >= 0.5 else 0
        confidence = ensemble_prob if pred_class == 1 else (1.0 - ensemble_prob)

        prob_benign = 1.0 - ensemble_prob
        prob_malignant = ensemble_prob
        result_label = CLASS_NAMES[pred_class]

        prediction = {
            'class': pred_class,
            'confidence': float(confidence),
            'probability_benign': float(prob_benign),
            'probability_malignant': float(prob_malignant),
            'result': result_label,
        }

        patient = {
            'name': name,
            'age': age,
            'id': patient_id,
        }

        history_record = {
            'patient_name': name,
            'patient_age': age,
            'patient_id': patient_id,
            'prediction_result': result_label,
            'confidence': float(confidence),
            'probability_benign': float(prob_benign),
            'probability_malignant': float(prob_malignant),
            'created_at': datetime.now().isoformat(),
        }
        with history_lock:
            PREDICTION_HISTORY.insert(0, history_record)
            # Keep only last 100 records
            if len(PREDICTION_HISTORY) > 100:
                PREDICTION_HISTORY.pop()

        return jsonify({
            'patient': patient,
            'prediction': prediction,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Return in-memory prediction history."""
    with history_lock:
        history_copy = list(PREDICTION_HISTORY)
    return jsonify({'history': history_copy})


@app.route('/', methods=['GET'])
def index():
    """Serve the frontend index.html."""
    return send_file('index.html')

@app.route('/style.css')
def serve_css():
    """Serve the CSS stylesheet."""
    return send_file('style.css')


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """Generate and download a PDF report of all prediction history."""
    with history_lock:
        history_copy = list(PREDICTION_HISTORY)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=18, textColor=colors.HexColor('#e94560'),
        alignment=1, spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', parent=styles['Normal'],
        fontSize=12, textColor=colors.HexColor('#666666'),
        alignment=1, spaceAfter=10
    )

    elements = []
    elements.append(Paragraph('Cancer Risk Prediction Report', title_style))
    elements.append(Paragraph(
        f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        subtitle_style
    ))
    elements.append(Spacer(1, 20))

    if not history_copy:
        elements.append(Paragraph(
            'No prediction records available.',
            styles['Normal']
        ))
    else:
        total = len(history_copy)
        benign = sum(1 for r in history_copy if r['prediction_result'] == 'Benign')
        malignant = sum(1 for r in history_copy if r['prediction_result'] == 'Malignant')

        elements.append(Paragraph('Summary', styles['Heading2']))
        summary_data = [
            ['Total Predictions', str(total)],
            ['Benign', str(benign)],
            ['Malignant', str(malignant)],
        ]
        summary_table = Table(summary_data, colWidths=[2.5 * inch, 1.5 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#444444')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#2a2a4a')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Detailed records
        elements.append(Paragraph('Prediction History', styles['Heading2']))

        table_data = [['Patient', 'Age', 'ID', 'Result', 'Confidence',
                       'Prob Benign', 'Prob Malignant', 'Date']]
        for r in history_copy:
            table_data.append([
                r.get('patient_name', 'N/A'),
                str(r.get('patient_age', 'N/A')),
                r.get('patient_id', 'N/A'),
                r.get('prediction_result', 'N/A'),
                f"{r.get('confidence', 0):.2%}",
                f"{r.get('probability_benign', 0):.2%}",
                f"{r.get('probability_malignant', 0):.2%}",
                r.get('created_at', 'N/A')[:16],
            ])

        history_table = Table(table_data, repeatRows=1)
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e94560')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#444444')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#2a2a4a')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#2a2a4a'), colors.HexColor('#1f1f3a')]),
        ]))
        elements.append(history_table)

    doc.build(elements)
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = \
        f'attachment; filename=cancer_prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    return response


if __name__ == '__main__':
    print("[INFO] Starting Flask backend on http://127.0.0.1:5000")
    print("[INFO] Open http://127.0.0.1:5000/ in your browser to use the frontend")
    app.run(host='127.0.0.1', port=5000, debug=True)
