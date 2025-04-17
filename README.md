### README

# Implementação da rede neural MLP para recomendação de filmes

Este projeto utiliza Python para realizar análises e treinamentos de modelos de Machine Learning. Ele implementa uma Rede Neural Perceptron Multicamadas (MLP) para prever a avaliação média de filmes com base em características específicas.

## Detalhes do Algoritmo

A rede neural foi projetada para realizar uma tarefa de regressão, utilizando as seguintes colunas do dataset `movies_metadata.csv` como entrada:

- **popularity**: Popularidade do filme.
- **runtime**: Duração do filme (em minutos).
- **revenue**: Receita gerada pelo filme.
- **vote_average**: Avaliação média do filme (usada como variável alvo).

### Pré-processamento dos Dados

1. **Conversão de Dados**: As colunas são convertidas para valores numéricos, substituindo valores inválidos por `NaN`.
2. **Remoção de Valores Ausentes**: Linhas com valores ausentes são removidas.
3. **Normalização**: As colunas de entrada (`popularity`, `runtime`, `revenue`) são normalizadas utilizando `StandardScaler` para melhorar o desempenho do modelo.
4. **Divisão dos Dados**: Os dados são divididos em conjuntos de treino (80%) e teste (20%).

### Estrutura da Rede Neural

A rede neural foi implementada utilizando a biblioteca `tensorflow.keras` com a seguinte arquitetura:

- **Camada de Entrada**: 3 neurônios (uma para cada feature: `popularity`, `runtime`, `revenue`).
- **Camada Oculta 1**: 64 neurônios com função de ativação ReLU.
- **Camada Oculta 2**: 32 neurônios com função de ativação ReLU.
- **Camada de Saída**: 1 neurônio para prever a avaliação média (`vote_average`), sem função de ativação (regressão).

### Treinamento

- **Função de Perda**: Mean Squared Error (MSE).
- **Otimizador**: Adam.
- **Métricas**: Mean Absolute Error (MAE).
- **Épocas**: 10.
- **Tamanho do Batch**: 32.
- **Validação**: 20% dos dados de treino são usados para validação durante o treinamento.

### Avaliação

O modelo é avaliado no conjunto de teste utilizando o erro médio absoluto (MAE) como métrica principal.

## Como executar o projeto

Siga os passos abaixo para configurar e executar o projeto:

### 1. Clonar o repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar e ativar um ambiente virtual
No macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

No Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar as dependências
Com o ambiente virtual ativado, instale as dependências listadas no arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Executar o projeto
Após a instalação das dependências, execute o arquivo principal do projeto:
```bash
python main.py
```

### 5. Desativar o ambiente virtual
Após finalizar, você pode desativar o ambiente virtual com:
```bash
deactivate
```

### Observação
Certifique-se de ter o Python 3.7 ou superior instalado em sua máquina.
