<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>Universiteti i Prishtines</p>
      <p>Fakulteti i Inxhinierise Elektrike dhe Kompjuterike</p>
      <p>Programi Master</p>
      <p>Profesori: Prof. Dr. Lule Ahmeti </p>
      <p>Mentori: Mentor: Dr. Sc. Mërgim H. HOTI</p>
      <p>Lenda: Machine Learning</p>
    </td>
 </tr>
</table>

# Projekti i Mësimit të Makinerisë

Ky projekt është i strukturuar për të zhvilluar, trajnuar dhe vlerësuar modele të mësimit të makinerisë duke përdorur datasetet e dhëna.

## Struktura e Projektit

```
machine-learning-project/
│── data/                  # Datasetet e papërpunuara dhe të përpunuara
│   ├── raw/               # Dataseti origjinal (i paprekur)
│   ├── processed/         # Dataseti i përpunuar (pas pastrimit dhe transformimit)
│── models/                # Modelet e ruajtura (checkpoint-e, modeli final)
│── notebooks/             # Jupyter notebooks për eksplorim dhe analizë
│── reports/               # Raporte, vizualizime, dhe log-e
│── scripts/               # Skripte ndihmëse për automatizim (p.sh., shkarkimi i të dhënave)
│── src/                   # Kodi burimor për trajnim, vlerësim dhe inferencë
│   ├── data_preprocessing.py  # Skripti për pastrimin dhe transformimin e të dhënave
│   ├── train.py           # Skripti për trajnim të modelit
│   ├── evaluate.py        # Skripti për vlerësim të modelit
│   ├── inference.py       # Skripti për parashikime
│── tests/                 # Teste njësish (unit tests) për skriptet dhe funksionet
│── requirements.txt       # Paketat e nevojshme për projektin
│── README.md              # Dokumentacioni i projektit
│── .gitignore             # Skedarët që duhet të injorohen nga Git
```

## Përpunimi i të Dhënave

### Të dhënat para përpunimit
Dataseti origjinal ndodhet në `data/raw/` dhe mund të ketë:
- Vlera të munguara
- Të dhëna të dyfishta
- Kolona me formate të pasakta
- Vlera jashtë intervaleve të lejuara (p.sh., vitet jashtë intervalit 2019-2024)
- Vlera kategorike të pa koduara

Shembuj të të dhënave para përpunimit:
```
Viti (int), Muaji (int), Përshkrimi i Sektorit (string), Komuna (string), Statusi i Regjistrimit (string), Përshkrimi i Gjobave në bazë të Ligjit (string), Numri i Tatimpaguesve (int), Numri i Gjobave të Lëshuara (int), Vlera e Gjobave të Lëshuara (float)
2019,1,Arsimi,PRIZREN,Individual,Nd.Neni.53.6-Law 03/L-222,1,1,250.0
2019,1,Arsimi,PRIZREN,Individual,Nd.Neni.53(2.1)-Law 03/L-222,1,1,250.0
```

### Përpunimi i të Dhënave
Skripti `src/data_preprocessing.py` kryen hapat e mëposhtëm:
1. **Heqja e vlerave të munguara** - Të gjitha rreshtat me vlera të munguara hiqen nga dataseti.
2. **Heqja e vlerave të dyfishta** - Rreshtat e dyfishta eliminohen për të shmangur tepricat.
3. **Filtrimi i vlerave të pavlefshme**
   - Kolona `Viti` lejohet vetëm me vlera midis 2019 dhe 2024, të tjerat hiqen.
   - Kolona `Muaji` lejohet vetëm me vlera nga 1 deri në 12.
   - Kolona `Komuna` duhet të jetë midis 1 dhe 38.
4. **Kodimi i të dhënave kategorike**
   - Kolonat kategorike kodohen përmes `Label Encoding` ose `One-Hot Encoding`.

### Të dhënat pas përpunimit
Pas përpunimit, dataseti i përpunuar ruhet në `data/processed/` dhe ka:
- Nuk ka vlera të munguara apo të dyfishta
- Të gjitha kolonat numerike janë brenda intervaleve të lejuara
- Vlerat kategorike janë të koduara

Shembull i të dhënave pas përpunimit:
```
Viti (int), Muaji (int), Përshkrimi i Sektorit (int - Label Encoded), Komuna (int - Label Encoded), Statusi i Regjistrimit (int - Label Encoded), Përshkrimi i Gjobave (int - Label Encoded), Numri i Tatimpaguesve (int), Numri i Gjobave të Lëshuara (int), Vlera e Gjobave të Lëshuara (float)
2019,1,0,23,1,2,1,1,250.0
2019,1,0,23,1,3,1,1,250.0
```
(Ku kolonat kategorike janë konvertuar në numra përmes `Label Encoding`.)

## Instalimi dhe Përdorimi

### Kërkesat
Për të ekzekutuar projektin, sigurohuni që keni paketat e nevojshme:
```
pip install -r requirements.txt
```

### Ekzekutimi i Skripteve
1. **Përpunimi i të dhënave**
   ```
   python src/data_preprocessing.py
   ```
2. **Trajnimi i modelit**
   ```
   python src/train.py
   ```
3. **Vlerësimi i modelit**
   ```
   python src/evaluate.py
   ```
4. **Inferenca mbi të dhënat e reja**
   ```
   python src/inference.py
   ```

## Autorët
- Projekti u zhvillua për qëllime studimore dhe praktike.

---

Ky README përshkruan të gjitha aspektet e projektit dhe përpunimin e të dhënave. Ju lutemi, ndiqni udhëzimet për të përpunuar datasetin dhe trajnuar modelin tuaj.
