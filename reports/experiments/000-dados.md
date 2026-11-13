# 000. Dados: análise do WFLW e decisões de pré-processamento

## 1. Objetivo

Entender o WFLW antes de treinar e fixar as decisões de dados: leitura das anotações, recorte,
resolução, divisão entre treino e validação e data augmentation.

## 2. O dataset

| | Treino | Teste |
|---|---|---|
| Faces | 7.500 | 2.500 |
| Imagens | 5.316 | 2.118 |
| Máximo de faces por imagem | 18 | 6 |
| Lado da caixa (mediana; p5 a p95, px) | 136; 75 a 480 | 130; 74 a 460 |
| Faces com algum atributo difícil | 48,9 % | 78,6 % |

Cada linha das anotações descreve uma **face**, então o dataset é indexado por face. 883 imagens
têm faces nos dois conjuntos oficiais; a divisão oficial é por face e foi respeitada.

**Atributos (% das faces):**

| | Pose | Expressão | Iluminação | Maquiagem | Oclusão | Desfoque |
|---|---|---|---|---|---|---|
| Treino | 3,5 | 2,4 | 20,4 | 1,1 | 17,9 | 18,8 |
| Teste | 13,0 | 12,6 | 27,9 | 8,2 | 29,4 | 30,9 |

O teste é bem mais difícil que o treino: a pose aparece quase 4 vezes mais e a expressão mais de 5 vezes. Por isso o
erro na validação (tirada do treino) subestima o erro no teste.

## 3. Olhos fechados

EAR de 6 pontos calculado com os landmarks **anotados**:

| | Mediana | EAR < 0,10 | EAR < 0,15 | EAR < 0,20 |
|---|---|---|---|---|
| Treino | 0,258 | 4,1 % | 8,8 % | 21,0 % |
| Teste | 0,265 | 5,7 % | 11,0 % | 22,1 % |

Cerca de **9 % das faces de treino** têm EAR abaixo de 0,15 (olhos semicerrados ou fechados). Com
tão poucos exemplos, um modelo sem tratamento tende a prever um olho médio aberto. Isso é medido no
experimento 001 e tratado no 002 (`002-olhos-fechados.md`).

## 4. Decisões de pré-processamento

### Recorte

Quadrado com o mesmo centro da caixa anotada e lado igual a **1,25 vezes o maior lado**,
redimensionado para **256 × 256**. Faces com todos os 98 pontos dentro do recorte:

| Margem | 1,00 | 1,10 | **1,25** | 1,40 |
|---|---|---|---|---|
| Faces com todos os pontos dentro | 34,6 % | 89,6 % | **99,0 %** | 99,9 % |

A margem de 1,25 cobre 99 % das faces sem desperdiçar resolução. A mesma regra vale para a caixa do
detector YuNet, que segue a mesma convenção da caixa anotada (testa incluída).

### Resolução

No recorte de 256, a abertura mediana do olho (pontos 62 a 66) é de **10 px** (p10 = 5,5 px). Com
mapas de calor de 64 × 64 (passo 4), isso dá só 2,5 células: escolher a célula mais forte erraria
até 2 px, cerca de 20 % da abertura. Por isso a decodificação usa **soft-argmax**, com precisão
abaixo de um pixel. Uma entrada de 128 deixaria o olho com uns 5 px e foi descartada.

### Divisão entre treino, validação e teste

- Validação: **10 % das imagens do treino oficial**, semente 42, separadas **por imagem**, para que
  faces da mesma foto não fiquem dos dois lados: **6.737 faces de treino e 763 de validação**.
- O teste oficial (2.500 faces) só é usado na avaliação final de cada modelo escolhido.

### Data augmentation

| Transformação | Faixa | Observação |
|---|---|---|
| Espelhamento horizontal | 50 % | x' = W − 1 − x **e** reordenação dos índices com `Mirror98.txt` |
| Rotação | ±15° | |
| Escala do recorte | 0,85 a 1,2 | inclui aproximar a face (escala abaixo de 1) |
| Deslocamento | ±6 % por eixo | |
| Contraste e brilho | contraste ±30 %, brilho ±15 % de 255 | |
| Baixa resolução (só no modelo final) | 30 %, redução de 2 a 4 vezes | ver `002-olhos-fechados.md` |

As transformações geométricas são compostas numa única matriz afim aplicada à imagem e aos pontos.
Verificação automática: em 1.000 faces aumentadas, **nenhuma** ficou com os olhos trocados depois do
espelhamento.

### Cache de recortes

Para não abrir as fotos inteiras a cada época, cada face é salva uma vez como recorte de 384 × 384
com margem de 1,875 (1,25 × 1,5): os 256 px do centro são exatamente o recorte de entrada, e a
folga em volta comporta rotação, afastamento e deslocamento sem bordas pretas.

## 5. Conclusão

Pipeline de dados validado por testes e por inspeção visual (figuras nos notebooks 01 e 02).
Próximo passo: o baseline (`001-baseline.md`).
