# TCO Streamlit App

Einfache Streamlit-App zum Vergleich von Baustellenenergie:
- Dieselgenerator
- Mobile Sodium-Ion Batterie

## Voraussetzungen
- Python 3.10+
- `pip`

## Installation
```bash
cd /Users/oskar/Downloads/TCO_Model
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Start
```bash
streamlit run app.py
```

## Häufiger Fehler: `import matplotlib.pyplot as plt`
Wenn hier ein Fehler kommt, ist `matplotlib` meist nicht in der aktiven Umgebung installiert.

Prüfen:
```bash
which python
python -m pip show matplotlib
```

Fix:
```bash
python -m pip install -r requirements.txt
```

Danach erneut starten:
```bash
streamlit run app.py
```
