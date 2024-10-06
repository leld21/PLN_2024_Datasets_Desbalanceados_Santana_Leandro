import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report

file_path = "SHOPEE_REVIEWS_Filtrado.csv"
df = pd.read_csv(file_path, low_memory=False)
# 3)
# a), b) 
df.info()
print(df.head())

#c) e e)
# O dataset escolhido possui apenas uma label por entrada/linha. Alem disso nao possui linha sem label.

#d)
print("quantidade por labels:")
rating_distribution = df['review_rating'].value_counts().sort_index()
print(rating_distribution)

# 4) ...

# 7) remove duplicadas ( reviews iguais )
df.drop_duplicates(subset=['review_text'], inplace=True)
rating_distribution = df['review_rating'].value_counts().sort_index()
print(rating_distribution)

# 8) gráfico para mostrar a distribuição de quantidade de palavras por registro

df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribuição de Quantidade de Palavras por Registro')
plt.xlabel('Número de Palavras')
plt.ylabel('Quantidade de Registros')
plt.xlim(0, 100)
plt.show()

# 9)
#tirar valores nulos
df['review_text'] = df['review_text'].fillna("")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review_text'])
y = df['review_rating']

# Dividindo o dataset em treino, validação e teste (70% treino, 15% validação e 15% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}, Test set: {X_test.shape}")

# 10) slices

slices = [0.1, 0.3, 0.5, 0.7]
training_slices = {}

for slice_size in slices:
    X_slice, _, y_slice, _ = train_test_split(X_train, y_train, train_size=slice_size, random_state=42)
    training_slices[f"slice_{int(slice_size*100)}%"] = (X_slice, y_slice)

for slice_name, (X_slice, y_slice) in training_slices.items():
    print(f"{slice_name}: {X_slice.shape}")

# 11) Naive Bayes e Micro e Macro F1 score

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# predições
y_pred_valid = nb_classifier.predict(X_valid)

# Calculando Micro e Macro F1 score
micro_f1 = f1_score(y_valid, y_pred_valid, average='micro')
macro_f1 = f1_score(y_valid, y_pred_valid, average='macro')

# Apresentando os scores
print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

# Relatório de Classificação
print("\nRelatório de Classificação:\n", classification_report(y_valid, y_pred_valid))