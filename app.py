# ============================================================
# PROJEKTS: Sirds slimību prognozēšana ar mašīnmācīšanos
# ============================================================

# ============================================================
# 1. BIBLIOTĒKU IMPORTS
# ============================================================

# warnings bibliotēka ļauj kontrolēt brīdinājumus.
# Šeit mēs izslēdzam brīdinājumus, lai izvade būtu tīra
# un prezentācijas laikā nerādītos lieki paziņojumi.
import warnings
warnings.filterwarnings('ignore')

# NumPy ir bibliotēka darbam ar skaitliskiem masīviem.
# Tā nodrošina ātrus matemātiskos aprēķinus un tiek izmantota
# gandrīz visos mašīnmācīšanās projektos.
import numpy as np

# Pandas ir galvenā bibliotēka darbam ar tabulveida datiem.
# Ar to var:
# - ielādēt CSV failus
# - apstrādāt kolonnas
# - filtrēt datus
# - veidot DataFrame objektus
import pandas as pd

# Matplotlib ir pamata grafiku zīmēšanas bibliotēka.
import matplotlib.pyplot as plt

# Seaborn ir uzlabota grafiku bibliotēka,
# kas balstīta uz matplotlib un nodrošina
# skaistākus un informatīvākus grafikus.
import seaborn as sns

# Scikit-learn ir galvenā mašīnmācīšanās bibliotēka Python vidē.
# Tā satur klasifikācijas algoritmus un datu apstrādes rīkus.
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Mašīnmācīšanās algoritmi:
# Decision Tree – lēmumu koks
# Random Forest – vairāku koku ansamblis
# KNN – tuvāko kaimiņu metode
# SVM – atbalsta vektoru mašīna
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# SciPy bibliotēka – izmantojam Box-Cox transformācijai
from scipy.stats import boxcox


# ============================================================
# 2. DATU IELĀDE UN DIREKTORIJA
# ============================================================

# Šeit tiek ielādēts fails "heart.csv".
# Tā kā mēs izmantojam relatīvo ceļu,
# fails atrodas tajā pašā direktorijā (mapē),
# kur atrodas šis Python fails.
#
# Ja fails būtu citā mapē, vajadzētu norādīt pilnu ceļu,
# piemēram: "dati/heart.csv"
df = pd.read_csv("heart.csv")


# ============================================================
# PAPILDU LABOJUMS: TARGET KOLONNAS TIPA LABOŠANA
# ============================================================

# Problēma:
# Zemāk esošajā ciklā visas kolonnas,
# kas nav continuous_features sarakstā,
# tiek pārveidotas par "object" tipu.
#
# Tas nozīmē, ka arī "target" kolonna kļūst par tekstveida tipu.
# Taču klasifikācijas algoritmi (Decision Tree, Random Forest utt.)
# pieprasa, lai mērķa mainīgais (y) būtu diskrētas skaitliskas klases (0 vai 1).
#
# Ja target ir object tipa, rodas kļūda:
# "ValueError: Unknown label type: unknown"
#
# Risinājums:
# Piespiedu kārtā pārveidojam target kolonnu atpakaļ par veselu skaitli (int).

df["target"] = df["target"].astype(int)


# ============================================================
# 3. DATU SAGATAVOŠANA
# ============================================================

# Definējam nepārtrauktos (skaitliskos) mainīgos.
# Šīs kolonnas satur reālus skaitļus.
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# CIKLS:
# Šis for cikls iziet cauri katrai kolonnai datu kopā.
# Ja kolonna nav nepārtrauktais mainīgais,
# tā tiek pārveidota par kategorisku tipu.
for col in df.columns:
    if col not in continuous_features:
        df[col] = df[col].astype("object")

# One-hot encoding:
# Kategoriskie dati tiek pārveidoti binārā formā (0/1),
# jo mašīnmācīšanās algoritmi strādā tikai ar skaitļiem.
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)

# Definējam ievades datus (X) un mērķa mainīgo (y).
# X – pazīmes
# y – vai pacientam ir sirds slimība
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Papildu drošība:
# Vēlreiz nodrošinām, ka mērķa mainīgais ir vesels skaitlis.
# Tas garantē, ka klasifikatori saņem pareizu datu tipu.
y = y.astype(int)

# Datu sadalīšana:
# 80% apmācībai
# 20% testēšanai
# Stratify nodrošina, ka klašu proporcija saglabājas vienāda.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


# ============================================================
# 4. DATU TRANSFORMĀCIJA (Box-Cox)
# ============================================================

# Box-Cox transformācija palīdz padarīt datus
# tuvākus normālajam sadalījumam.
# Tas uzlabo dažu algoritmu precizitāti.

# Pievienojam mazu konstanti, jo Box-Cox
# darbojas tikai ar pozitīvām vērtībām.
X_train = X_train.copy()
X_test = X_test.copy()

X_train["oldpeak"] += 0.001
X_test["oldpeak"] += 0.001

lambdas = {}

# CIKLS:
# Iziet cauri katram nepārtrauktajam mainīgajam
# un pielieto transformāciju.
for col in continuous_features:
    if (X_train[col] > 0).all():
        X_train[col], lambdas[col] = boxcox(X_train[col])
        X_test[col] = boxcox(X_test[col], lmbda=lambdas[col])


# ============================================================
# 5. MAŠĪNMĀCĪŠANĀS MODEĻU APRAKSTS
# ============================================================

# Decision Tree:
# Veido lēmumu struktūru "ja–tad".
# Pluss – viegli interpretējams.
# Mīnuss – var pārāk pielāgoties datiem (overfitting).
dt = DecisionTreeClassifier(random_state=0)

# Random Forest:
# Apvieno vairākus lēmumu kokus.
# Katrs koks balso par rezultātu.
# Parasti precīzāks un stabilāks modelis.
rf = RandomForestClassifier(random_state=0)

# KNN (K-Nearest Neighbors):
# Nosaka klasi, balstoties uz tuvākajiem kaimiņiem.
# Nepieciešama datu standartizācija.
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# SVM (Support Vector Machine):
# Meklē optimālu robežlīniju starp klasēm.
# Ļoti efektīvs sarežģītos datos.
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])


# ============================================================
# 6. MODEĻU APMĀCĪBA UN NOVĒRTĒŠANA
# ============================================================

# Izveidojam vārdnīcu ar visiem modeļiem.
models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "KNN": knn_pipeline,
    "SVM": svm_pipeline
}

# CIKLS:
# Šis cikls automātiski:
# - apmāca katru modeli
# - veic prognozi
# - izvada rezultātus
for name, model in models.items():
    
    # Modeļa apmācība
    model.fit(X_train, y_train)
    
    # Prognoze uz testēšanas datiem
    y_pred = model.predict(X_test)
    
    # Rezultātu izdrukāšana
    print(f"\n===== {name} =====")
    print("Precizitāte (Accuracy):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))