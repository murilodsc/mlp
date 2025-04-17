import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os

# Carregando o dataset
df = pd.read_csv('movies_metadata.csv', sep=';', low_memory=False, on_bad_lines='skip')

# Extrair o ano da data de lançamento
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Codificar a linguagem original com one-hot encoding
language_dummies = pd.get_dummies(df['original_language'], prefix='lang')
df = pd.concat([df, language_dummies], axis=1)

# Selecionando colunas relevantes
cols = ['popularity', 'runtime', 'revenue', 'vote_average', 'release_year']
df = df[cols + list(language_dummies.columns)]

# Convertendo colunas para numérico (forçando erros para NaN e depois limpando)
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover filmes com receita igual a 0
df = df[df['revenue'] > 0]

# Removendo valores ausentes
df = df.dropna()

# Separando features (X) e target (y)
X = df[['popularity', 'runtime', 'revenue', 'release_year'] + list(language_dummies.columns)]
y = df['vote_average']

# Normalizando os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criando o modelo MLP
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))     # define a forma da entrada
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))        # camada oculta 1
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))         # camada oculta 2
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))         # camada oculta 3
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))         # camada oculta 4
model.add(Dense(1))                             # saída (regressão)

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Treinando o modelo
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2)

# Histórico de treinamento
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2)

# Plotando o MAE
plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label='MAE - Treino')
plt.plot(history.history['val_mae'], label='MAE - Validação')
plt.title('Erro Médio Absoluto (MAE) por Época')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('docs/mae_vs_epochs.png')

# Plotando o Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss - Treino')
plt.plot(history.history['val_loss'], label='Loss - Validação')
plt.title('Loss (MSE) por Época')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('docs/loss_vs_epochs.png')

os.makedirs('docs', exist_ok=True)

# Avaliando o modelo
loss, mae = model.evaluate(X_test, y_test)
print(f'Erro médio absoluto (MAE) no teste: {mae:.2f}')
