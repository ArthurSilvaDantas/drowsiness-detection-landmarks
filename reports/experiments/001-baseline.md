# 001. Baseline: ResNet-18 com heatmaps e soft-argmax

## 1. Objetivo

Ter um primeiro modelo **simples e correto**, que aprenda desde as primeiras épocas e sirva de
referência para os experimentos com olhos fechados. Este modelo **não tem nenhum tratamento especial
para olhos fechados**: a expectativa é que ele acerte bem os olhos abertos, mas desenhe um olho
médio aberto quando o olho está fechado. Este experimento mede esse efeito.

## 2. Configuração

| Item | Valor |
|---|---|
| Backbone | ResNet-18 com pesos da ImageNet, sem `avgpool` e `fc` |
| Ajuste do backbone | fine-tuning completo, com taxa de aprendizagem 5 vezes menor que a da cabeça |
| Cabeça | 3 × (convolução transposta 4 × 4, passo 2, 256 canais, BatchNorm, ReLU) e convolução 1 × 1 com 98 saídas; pesos iniciais pequenos (desvio 0,001) |
| Parâmetros | 15.397.538, todos treináveis (backbone 11.176.512; cabeça 4.221.026) |
| Entrada | recorte de 256 × 256, margem de 1,25 |
| Decodificação | mapas de 64 × 64, softmax espacial e **soft-argmax** (posição esperada, subpixel) |
| Perda | L1 nas coordenadas decodificadas, dividida por 256 |
| Peso dos olhos / sobreamostragem | nenhum (1) |
| Augmentation | espelhamento com Mirror98, rotação ±15°, escala 0,85 a 1,2, deslocamento ±6 %, brilho e contraste |
| Divisão | 6.737 treino e 763 validação (por imagem, do treino oficial); 2.500 teste |
| Lote / épocas | 32 / 40 |
| Otimizador | AdamW (weight decay 1e-4); taxa de 1e-3 na cabeça e 2e-4 no backbone; 1 época de warmup e decaimento em cosseno |
| Escolha da época | menor NME na validação |

**Por que soft-argmax com L1 e não mapas gaussianos com MSE:** com MSE pixel a pixel, um mapa todo
zero já tem perda baixa (a armadilha dos mapas nulos, demonstrada no notebook 03), e a rede pode
ficar presa nessa solução. Com soft-argmax a perda é calculada na coordenada, não existe saída
trivial, e a decodificação é subpixel, o que importa porque o olho aberto tem só uns 10 px de altura.

**Por que fine-tuning completo:** congelar o backbone funciona bem para classificar, mas localizar
pontos com precisão exige adaptar também as camadas iniciais, que guardam a informação espacial. A
taxa menor no backbone e o warmup protegem os pesos da ImageNet dos gradientes grandes da cabeça
ainda aleatória no início do treino.

## 3. Reprodutibilidade

A mesma configuração foi treinada três vezes, em máquinas e implementações diferentes:

| Execução | Melhor época | NME de validação |
|---|---|---|
| `train.py`, Apple M5 (MPS) | 38 | 4,11 % |
| notebook `04_treino`, Apple M5 (MPS) | 40 | 4,09 % |
| `train.py`, RTX A6000 | 38 | 4,13 % |

As três chegam ao mesmo nível. A execução na RTX A6000 é a **referência** dos experimentos
seguintes, porque todas as variantes foram treinadas nessa máquina.

Curvas de treino (notebook 04): o NME de validação cai desde a primeira época (11,8 % → 7,4 % →
6,2 % → 5,8 % na época 4) e se estabiliza perto de 4,1 % depois da época 30. A correlação do EAR
sobe mais devagar: oscila em torno de 0,3 até a época 20, passa de 0,58 na época 28 e termina em
0,63. A abertura da pálpebra é a última coisa que o modelo aprende.

Treino contra validação, sem augmentation: NME de 3,35 % no treino e 4,09 % na validação, um
overfitting leve.

## 4. Resultados no teste (WFLW, 2.500 faces)

Modelo do notebook 04 avaliado no notebook 05, com o recorte pela caixa anotada e sem TTA:

| Métrica | Valor |
|---|---|
| NME | **5,21 %** |
| Taxa de falha (NME > 10 %) | 7,20 % |
| NME dos olhos | 3,68 % |
| Taxa de falha dos olhos | 3,76 % |

| Pose | Expressão | Iluminação | Maquiagem | Oclusão | Desfoque |
|---|---|---|---|---|---|
| 9,39 | 5,72 | 5,08 | 5,40 | 6,40 | 5,95 |

| Contorno | Sobrancelhas | Nariz | Olhos | Boca | Pupilas |
|---|---|---|---|---|---|
| 6,96 | 4,89 | 3,95 | 3,68 | 4,51 | 4,10 |

A execução da RTX A6000, com a média da face espelhada (TTA), dá **5,05 %** de NME e 6,08 % de
falha (notebook 06); esse é o número usado nas comparações com o modelo final.

**Caminho completo** (YuNet acha as faces, recorte, modelo), associando cada face anotada à face
prevista de menor NME: 97,8 % das faces detectadas, NME de 5,18 % nas faces detectadas e mediana de
97 ms por imagem na CPU. Contando as faces não detectadas como falha, a taxa de falha fica em
9,60 %. O NME quase não muda em relação à caixa anotada (5,21 %), então o recorte feito na aplicação
é compatível com o do treino; a diferença na taxa de falha vem das faces que o detector não
encontra.

### A medida dos olhos

| Medida | Valor |
|---|---|
| Correlação do EAR previsto com o anotado | 0,658 |
| Erro médio do EAR | 0,054 |
| Inclinação da reta EAR previsto × anotado | 0,42 |
| Olhos fechados (EAR anotado < 0,15; 274 faces): EAR previsto médio | 0,198 (o anotado é 0,082) |
| Olhos fechados reconhecidos (EAR previsto < 0,15) | 15,0 % |
| Olhos abertos (EAR anotado ≥ 0,25) previstos como fechados | 0,1 % |

## 5. Exemplos

As figuras de acertos, falhas e olhos fechados estão no notebook 05. As piores falhas são faces com
pose e oclusão (touca e óculos de natação; capacete com um trombone na frente do rosto), duas faces
sobrepostas no mesmo recorte (o modelo mistura as faces) e faces muito desfocadas. Nos olhos
fechados, o modelo acerta **onde** o olho está, mas deixa uma pequena abertura entre as pálpebras.

## 6. Análise

- **Precisão geral boa para um baseline:** 5,2 % de NME no teste com ResNet-18 e entrada de 256,
  no nível de métodos publicados como o LAB (5,27 %). A diferença entre validação e teste (4,1 % →
  5,2 %) é esperada, porque o teste tem muito mais pose e expressão.
- **Pose é o ponto fraco geral** (9,4 %), seguida de oclusão; o contorno é a região mais difícil.
- **O modelo desenha um olho médio aberto:** a inclinação de 0,42 mostra que ele comprime o EAR em
  direção à média. Olhos fechados saem com EAR perto de 0,20, valor de um olho semiaberto, e só 15 %
  ficam abaixo de 0,15. A causa provável é o desbalanceamento (só 9 % das faces de treino com olho
  fechado) somado à perda: com pesos iguais, os 16 pontos dos olhos são só 16 % da perda, e um erro
  de 1 a 2 px entre as pálpebras quase não muda a L1.

## 7. Conclusão

O baseline é a referência de precisão geral. O problema dos olhos fechados é tratado no experimento
002 (`002-olhos-fechados.md`). Na época, o critério previsto para o sucesso era a correlação do EAR;
no 002 ele foi trocado pela fração de olhos fechados reconhecidos, que é o que decide o alerta (a
troca está explicada lá).
