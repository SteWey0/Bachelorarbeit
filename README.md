## Analyse-Framework für meine Bachelorarbeit mit folgenden Files:

- corrections.py: Code für pattern_correction und lowpass Filter. Übernommen von https://github.com/AndiZm/curcor/blob/main/programs/analysis/2023/corrections.py
- routines.py: Enthält die meisten Analyseschritte für die Datenauswertung. Darunter fallen:
    - Einlesen von Daten
    - Anwenden der pattern_correction, Bildung eines gewichteten Mittelwerts und Anwendung eines lowpass Filters
    - Speichern der gemittelten $g^{(2)}$-Funktion
    - Berechnung einer FFT der gemittelten $g^{(2)}$-Funktion

    Für die Zukunft geplant sind:
    - Korrelation der mean-pulseshapes, Interpolation und Fit
    -  Bestimmung des Integrals und Fehlers darauf
- analysis_tests.ipynb: Jupyter-Notebook mit allen Tests die ich bereits für die Datenanalyse gemacht habe. 
- analysis.ipynb: Jupyter-Notebook für die finale Datenanalyse (bisher ungenutzt)