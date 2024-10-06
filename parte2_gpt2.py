import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("SHOPEE_REVIEWS_Filtrado.csv")
df = df.dropna(subset=['review_text'])

# gpt-2
tokenizer = AutoTokenizer.from_pretrained("miguelvictor/python-gpt2-large")
model = AutoModel.from_pretrained("miguelvictor/python-gpt2-large")

tokenizer.pad_token = tokenizer.eos_token

# geracao de embeddings
def get_gpt2_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Gerar embeddings para cada review no dataset
print("Gerando embeddings para o dataset...")
df['embeddings'] = df['review_text'].apply(lambda x: get_gpt2_embeddings(str(x), tokenizer, model))

# Criar uma matriz de embeddings para FAISS
embeddings_matrix = np.vstack(df['embeddings'].values)
embedding_dimension = embeddings_matrix.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings_matrix) 

# splits
X_train, X_test, y_train, y_test = train_test_split(
    embeddings_matrix, df['review_rating'], test_size=0.3, random_state=42)

# Treinar um classificador simples de regressao
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Prever
y_pred = clf.predict(X_test)

# Calcular Micro e Macro F1 Scores
micro_f1 = f1_score(y_test, y_pred, average='micro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
