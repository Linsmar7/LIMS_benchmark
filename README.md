# LIMS - Sistema de Indexação de Espaços Métricos

LIMS é uma estrutura de índice inovadora que integra métodos de particionamento compacto, técnicas baseadas em pivôs e índices aprendidos para suportar processamento eficiente de consultas de similaridade em espaços métricos.

## Começando

### 1. Preparar Datasets

Você precisará baixar os datasets e prepará-los usando o script Python fornecido.

*   **ImageNet**: [Download](https://image-net.org/download-images.php)
*   **Forest Cover Type**: [Download Kaggle](https://www.kaggle.com/c/forest-cover-type-prediction/data)
*   **CoPHIR**: [Download Data Mendeley](https://data.mendeley.com/datasets/8wp73zxr47/12)

Após baixar os arquivos, extraia-os para a pasta `datasets/` (crie-a se não existir). Em seguida, execute o script de geração de dados para formatá-los corretamente:

```bash
python gen_data.py
```

### 2. Configurar para seu Próprio Dataset

Para rodar o LIMS com seu próprio dataset, você precisa ajustar alguns parâmetros no código e seguir um fluxo específico de preparação.

#### A. Alterações no Código

1.  **Definir a Dimensão**:
    Abra o arquivo `common/Constants.cpp` e altere o valor de `COEFFS` para corresponder à dimensionalidade do seu dataset.
    ```cpp
    // Exemplo para dimensão 128
    const int Constants::COEFFS = 128; 
    ```

2.  **Configurar Caminho de Entrada para Clusterização**:
    Abra `main.cpp`. Na seção de avaliação de clusterização (dentro do `if(flag_eva)`), localize a variável `data_path` (aproximadamente linha 451) e aponte para o seu arquivo de dados formatado (TXT/CSV sem cabeçalho).
    ```cpp
    string data_path = "./datasets_processed/seu_dataset/seu_arquivo.txt";
    ```
    > **Nota**: O format esperado é um arquivo texto onde cada linha é um ponto, e as coordenadas são separadas por vírgula.

#### B. Fluxo de Execução

O processo LIMS é dividido em: Avaliação (Clusterização) -> Construção do Índice -> Benchmark/Validação.

**Passo 1: Compilar**
```bash
make clean
make all
```

**Passo 2: Avaliar e Gerar Clusters**
Execute o `main` com a flag de avaliação ativada (`1`). Isso irá gerar os centróides e arquivos de cluster iniciais.
```bash
# Uso: ./main [num_pivots] [num_clusters] [dimensao] 1 [min_clusters] [max_clusters] [passo]
./main 100 100 282 1 10 100 10
```
*   Isso criará arquivos em `outputFiles/K_[num_clusters]/` (ex: `outputFiles/K_100/`).
*   Os arquivos gerados serão: `ref.txt` (pivôs principais), `clu_X.txt` (dados clusterizados) e `ref_X.txt` (pivôs secundários).

**Passo 3: Mover Arquivos para Entrada**
Para construir o índice final, você deve mover os arquivos gerados para a pasta de entrada padrão (`inputFiles`).
```bash
# Exemplo: movendo os arquivos gerados no passo anterior
cp outputFiles/K_100/* inputFiles/
```
*   Certifique-se de que `inputFiles/` contenha `ref.txt`, `clu_0.txt`... e `ref_0.txt`...
Também é necessário que tenha os pontos que você quer buscar em KNN e range querie em `inputFiles/KNN.txt` e `inputFiles/range.txt`.

**Passo 4: Construir o Índice**
Agora execute o `main` com a flag de avaliação desativada (`0`). Isso lerá de `inputFiles`, calculará as distâncias de referência e salvará as páginas de dados binários em `data/`.
```bash
# Uso: ./main [num_pivots] [num_clusters] [dimensao] 0
./main 100 100 282 0
```

### 3. Executar Benchmark e Validação

Após a construção do índice (arquivos em `data/` e estrutura em memória pronta), você pode rodar os benchmarks.

Certifique-se de que os arquivos de consulta (`_knn.csv` e `_range.csv`) estejam em `r_tree/queries/`.

**Benchmark de Performance**
Mede tempo de resposta, I/O e uso de RAM.
```bash
# Uso: ./benchmark_lims [nome_dataset] [num_pivots] [num_clusters] [dimensao] [dir_indice_opcional]
./benchmark_lims cophir 100 100 282
```
Os resultados serão salvos em `results/benchmark_lims_[nome].csv`.

**Validação de Acurácia**
Compara os resultados do LIMS com uma busca linear (Ground Truth) para calcular o Recall.
```bash
# Uso: ./validar_lims [nome_dataset] [num_pivots] [num_clusters] [dimensao]
./validar_lims cophir 100 100 282
```
Os resultados serão salvos em `results/validacao_lims_[nome].csv`.