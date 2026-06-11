# Poda Estruturada em Modelos CIFAR-10

Este projeto implementa o treino de modelos *baseline* (**MLP** e **VGG-11**) no dataset **CIFAR-10** e avalia diferentes estratégias de **poda estruturada (*Structured Pruning*)**.

O objetivo é analisar o impacto da compressão dos modelos no desempenho preditivo e na eficiência computacional.

---

## 1. Configuração do Ambiente Virtual

Recomenda-se a utilização de um ambiente virtual (`venv`) para isolar as dependências do projeto.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## 2. Instalação de Dependências

Com o ambiente virtual ativo, instale as dependências necessárias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Estrutura do Projeto

O projeto encontra-se organizado nos seguintes notebooks:

### `Model_Dataset_Base.ipynb`

- Descarrega e prepara o dataset **CIFAR-10**
- Cria a divisão treino/validação
- Treina e guarda os modelos *baseline* (**MLP** e **VGG-11**)

### `Poda_Pipeline_MLP.ipynb`

- Carrega o modelo **MLP** treinado
- Aplica diferentes estratégias de poda estruturada:
  - L1
  - L2
  - Taylor
  - OBD
- Realiza *fine-tuning* e avalia os resultados

### `Poda_Pipeline_VGG.ipynb`

- Carrega o modelo **VGG-11** treinado
- Aplica as mesmas estratégias de poda adaptadas à arquitetura convolucional
- Realiza *fine-tuning* e compara métricas de desempenho

---

## 4. Ordem de Execução

Os notebooks devem ser executados **pela seguinte ordem**:

1. `Model_Dataset_Base.ipynb`  
2. `Poda_Pipeline_MLP.ipynb`  
3. `Poda_Pipeline_VGG.ipynb`

Esta sequência garante que os modelos *baseline* são treinados e guardados antes da aplicação da poda estruturada.

---

## 5. Dependências Principais

O projeto utiliza principalmente:

- **PyTorch**
- **TorchVision**
- **Torch-Pruning**
- **Scikit-learn**
- **Matplotlib**
- **NumPy**