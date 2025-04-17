import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carregando o dataset
df = pd.read_csv('movies_metadata.csv', sep=';', low_memory=False, on_bad_lines='skip')

# Selecionando colunas relevantes
cols = ['popularity', 'runtime', 'revenue', 'vote_average']
df = df[cols]

# Convertendo colunas para numérico (forçando erros para NaN e depois limpando)
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Removendo valores ausentes
df = df.dropna()

# Separando features (X) e target (y)
X = df[['popularity', 'runtime', 'revenue']]
y = df['vote_average']

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criando o modelo MLP
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))  # camada oculta 1
model.add(Dense(32, activation='relu'))               # camada oculta 2
model.add(Dense(1))                                         # saída (regressão)

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Treinando o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Avaliando o modelo
loss, mae = model.evaluate(X_test, y_test)
print(f'Erro médio absoluto (MAE) no teste: {mae:.2f}')
