# 002. Olhos fechados: 15 variantes, escolha do modelo final e aplicação

## 1. Problema

O baseline acerta onde ficam os pontos, mas **desenha um olho médio aberto**: na validação, o EAR
previsto dos olhos fechados fica em 0,19 (o anotado é cerca de 0,09) e só 18 % deles ficam abaixo
de 0,15. Para detectar sonolência, é exatamente essa medida que precisa funcionar.

Três causas prováveis, cada uma atacada por uma família de variantes:

| Causa | Ideia | Variantes |
|---|---|---|
| Poucos exemplos de olho fechado (9 % do treino) | sobreamostrar faces com EAR anotado < 0,15 (`WeightedRandomSampler`) | 002, 004, 010 |
| A perda quase não pune o erro entre as pálpebras | peso maior nos pontos 60 a 75; termo de **abertura** da pálpebra; Wing loss | 003, 004, 005, 007, 009 |
| Mapas de calor espalhados (o soft-argmax vira uma média) | regularização de Jensen-Shannon para uma gaussiana estreita, como no DSNT | 006, 006b, 008, 011, 014 |
| Faces pequenas de webcam | augmentation de baixa resolução (reduz 2 a 4 vezes e amplia de volta) | 012, 013, 014 |

## 2. Protocolo

- Todas as variantes usam a mesma divisão (semente 42) e são comparadas **só na validação** (763
  faces, 62 com olho fechado), sem TTA.
- **Medidas:** NME, NME dos olhos, correlação e inclinação do EAR, EAR previsto nos olhos fechados,
  fração de olhos fechados reconhecidos (EAR previsto < 0,15) e, para as principais, AUC da curva ROC
  de olho fechado contra não fechado e a fração de olhos não fechados marcados como fechados.
- **Regra de escolha:** entre as variantes com NME no máximo 0,2 ponto acima do baseline, a que
  reconhece mais olhos fechados; empate decidido pela correlação do EAR.
- O teste do WFLW é usado **uma vez**, no fim, só para o baseline e o modelo escolhido.

**Mudança de critério declarada:** o report 001 previa usar a correlação do EAR. A troca para a
fração de olhos fechados reconhecidos foi feita porque é ela que decide se o alerta acontece. Pela
correlação, a escolhida seria a 009 (0,746 contra 0,725), diferença dentro da variação entre
sementes (seção 5).

## 3. Resultados na validação

| Variante | O que muda | NME | NME olhos | Corr. EAR | Inclinação | EAR prev. fechados | Fechados reconhecidos |
|---|---|---|---|---|---|---|---|
| 001 | baseline | 4,13 | 2,87 | 0,650 | 0,48 | 0,190 | 17,7 % |
| 002 | sobreamostragem ×4 | 4,16 | 2,83 | 0,678 | 0,60 | 0,158 | 41,9 % |
| 003 | peso 4 nos olhos | 4,17 | 2,65 | 0,719 | 0,62 | 0,163 | 35,5 % |
| 004 | 002 + 003 | 4,22 | 2,65 | 0,714 | 0,68 | 0,138 | 59,7 % |
| 005 | abertura (peso 1) | 5,32 | 6,37 | 0,756 | 0,66 | 0,133 | 53,2 % |
| 006 | mapas (peso 1) | **4,01** | 2,76 | 0,679 | 0,54 | 0,177 | 17,7 % |
| 006b | mapas (peso 0,1) | 4,03 | 2,76 | 0,588 | 0,53 | 0,183 | 19,4 % |
| 007 | Wing loss | 4,10 | 2,83 | 0,605 | 0,48 | 0,197 | 11,3 % |
| 008 | 004 + mapas | 4,02 | 2,67 | 0,708 | 0,64 | 0,145 | 46,8 % |
| 009 | 004 + abertura 0,1 | 4,24 | 2,65 | 0,746 | 0,63 | 0,136 | 61,3 % |
| 010 | ×8 e peso 8 | 4,30 | 2,60 | 0,710 | 0,68 | **0,125** | **67,7 %** |
| 011 | 004 + mapas + abertura | 4,04 | 2,70 | 0,727 | 0,66 | 0,140 | 50,0 % |
| 012 | 004 + baixa resolução | 4,20 | 2,63 | 0,740 | 0,69 | 0,137 | 56,5 % |
| **013** | 010 + baixa resolução, 80 épocas | 4,17 | **2,55** | 0,725 | **0,71** | 0,127 | **67,7 %** |
| 014 | 013 + mapas | 4,07 | 2,71 | 0,715 | 0,66 | 0,133 | 62,9 % |

O que aprendemos:

- **Sobreamostragem e peso nos olhos** são o que mais ajuda, e os dois se somam (004, 010).
- **Abertura com peso 1** (005) melhora o EAR, mas estraga a posição dos pontos: o termo domina a
  perda. Com peso 0,1 (009) ele deixa de atrapalhar, com ganho pequeno.
- **Regularização dos mapas** dá o melhor NME geral (006), mas sozinha não ajuda os olhos fechados.
- **Wing loss** não ajudou nos olhos.

Pela regra, a escolhida é a **013**: empata com a 010 em olhos fechados reconhecidos (42 de 62) e
vence no desempate pela correlação do EAR.

## 4. Olho aberto contra fechado (curva ROC, validação)

| Variante | AUC (intervalo de 95 %) | Fechados reconhecidos | Não fechados marcados como fechados |
|---|---|---|---|
| 001 | 0,859 (0,796 a 0,916) | 17,7 % | 0,6 % |
| 004 | 0,909 (0,870 a 0,945) | 59,7 % | 5,1 % |
| 009 | 0,912 (0,868 a 0,951) | 61,3 % | 3,6 % |
| 010 | 0,909 (0,859 a 0,945) | 67,7 % | 6,3 % |
| 013 | 0,900 (0,848 a 0,943) | 67,7 % | 5,8 % |
| 014 | 0,909 (0,860 a 0,947) | 62,9 % | 4,9 % |

As variantes separam aberto de fechado melhor que o baseline, mas entre elas as AUCs são parecidas.
O que mais muda é **onde** o EAR previsto fica: as variantes mais fortes deslocam o EAR para baixo,
reconhecem mais olhos fechados no limiar fixo de 0,15 e marcam mais olhos abertos como fechados. Na
aplicação o limiar é relativo ao EAR típico da pessoa, o que compensa esse deslocamento.

## 5. Variação entre execuções e ablação

A 010 e a 013 foram treinadas de novo com as sementes 1 e 2 (mesma divisão):

| | NME | NME olhos | Corr. EAR | Fechados reconhecidos |
|---|---|---|---|---|
| 010 (40 épocas), 3 sementes | 4,32 ± 0,02 | 2,61 ± 0,02 | 0,72 ± 0,01 | 65,6 ± 2,5 % |
| 013 (80 épocas, baixa resolução), 3 sementes | **4,17 ± 0,01** | **2,57 ± 0,02** | **0,74 ± 0,01** | 67,7 ± 1,6 % |

Ablação na validação:

| Passo | NME | NME olhos | Corr. EAR | Fechados reconhecidos |
|---|---|---|---|---|
| baseline (001) | 4,13 | 2,87 | 0,650 | 17,7 % |
| + sobreamostragem ×4 e peso 4 (004) | 4,22 | 2,65 | 0,714 | 59,7 % |
| reforço para ×8 e peso 8 (010) | 4,30 | 2,60 | 0,710 | 67,7 % |
| + 80 épocas (015) | 4,16 | 2,56 | 0,724 | 64,5 % |
| + baixa resolução (013) | 4,17 | 2,55 | 0,725 | 67,7 % |

Conclusões que resistem à variação entre sementes:

- sair do baseline para as variantes com tratamento dos olhos é o grande ganho (18 % → 63 a 71 %);
- **80 épocas** em vez de 40 melhora o NME geral de forma consistente (4,32 → 4,17 %), e é isso que
  traz o NME do modelo final de volta para perto do baseline;
- a **baixa resolução** quase não muda as medidas na validação; o efeito esperado é em webcam;
- em olhos fechados, 010 e 013 **empatam**. A 013 foi escolhida pela regra; a vantagem comprovada
  dela sobre a 010 é o NME geral.

## 6. Média com a face espelhada (TTA)

| Validação | NME | NME olhos | Corr. EAR |
|---|---|---|---|
| 013 sem TTA | 4,17 | 2,55 | 0,725 |
| 013 com TTA | **4,05** | **2,48** | **0,754** |

O TTA melhora o NME, o NME dos olhos e a correlação do EAR em todos os modelos testados, ao custo de
rodar a rede duas vezes por face. Fica ligado no `predict.py` e no `app.py`.

## 7. Modelo final no teste do WFLW (2.500 faces, com TTA)

| | Baseline (001) | **Final (013)** |
|---|---|---|
| NME | **5,05 %** | 5,12 % |
| Taxa de falha | **6,08 %** | 6,68 % |
| NME dos olhos | 3,53 % | **3,27 %** |
| Correlação do EAR / erro médio | 0,644 | **0,725** / 0,048 |
| Inclinação do EAR | 0,45 | **0,65** |
| EAR previsto nos olhos fechados (anotado 0,08) | 0,188 | **0,127** |
| Olhos fechados reconhecidos (274 faces) | 23,4 % | **67,5 %** |
| Não fechados marcados como fechados (2.226 faces) | **1,0 %** | 5,2 % |

| Subconjunto | Pose | Expressão | Iluminação | Maquiagem | Oclusão | Desfoque |
|---|---|---|---|---|---|---|
| Baseline | 9,13 | 5,45 | 4,93 | 5,09 | 6,17 | 5,75 |
| Final | 9,22 | 5,53 | 5,05 | 5,23 | 6,28 | 5,89 |

**Caminho completo** (YuNet, recorte, modelo com TTA): 97,8 % das faces detectadas, NME de 5,08 %,
taxa de falha de 8,4 % (faces não detectadas contam como falha), NME dos olhos de 3,14 % e 150 ms
por imagem numa CPU de notebook.

**O compromisso:** o modelo final perde 0,07 ponto de NME geral e cerca de 0,1 ponto em cada
subconjunto, e marca mais olhos não fechados como fechados. Em troca, o NME dos olhos melhora 7 %
e o modelo reconhece cerca de três vezes mais olhos fechados.

## 8. Aplicação em vídeo

Vídeo de webcam de 33 s, cru, com um roteiro conhecido: dois fechamentos longos (de frente e com a
cabeça virada), cabeça virada com os olhos abertos, piscada de um olho só e piscadas.

| | Baseline | Final |
|---|---|---|
| EAR típico (mediana dos 3 primeiros segundos) | 0,260 | 0,268 |
| Fechamento de frente | 0,118 | **0,008** |
| Fechamento com a cabeça virada | 0,184 (limiar 0,182) | **0,039** (limiar 0,188) |

Com o modelo final, o `app.py` gera **2 alertas, exatamente nos dois fechamentos longos** (2,7 e
2,8 s), nenhum com a cabeça virada, 12 piscadas e 3 fechamentos curtos. Com o baseline, o segundo
fechamento só passa do limiar por 0,002.

Os parâmetros da aplicação e a sensibilidade de cada um estão em `08_aplicacao_video.ipynb`:

| Parâmetro | Valor | Resultado da variação |
|---|---|---|
| Fração de fechamento | 0,7 × mediana | 0,6 e 0,7 sem erros; 0,8 gera 1 falso alerta; 0,5 perde piscadas |
| Reabertura (histerese) | 0,8 × mediana | evita vários eventos quando o EAR oscila perto do limiar |
| Média móvel | 3 quadros | 9 quadros apaga piscadas (8 em vez de 12) |
| Piscada | até 0,4 s | limite superior de uma piscada espontânea |
| Alerta | a partir de 1,5 s | com 1,0 s, a piscada de um olho só (1,4 s) vira falso alerta |
| EAR | 6 pontos, média dos dois olhos | separação 19,2 (EAR8: 19,3; mínimo dos olhos: 17,8) |

## 9. Conclusão

- O problema do olho médio aberto foi resolvido sobretudo com **mais exemplos de olho fechado e mais
  peso nos pontos dos olhos**; o modelo final reconhece cerca de três vezes mais olhos fechados que
  o baseline, com NME geral praticamente igual.
- Na aplicação, o EAR do modelo final cai até perto de zero nos fechamentos, inclusive com a cabeça
  virada, e os alertas acontecem nos lugares certos.
- **Limitações:** uma semente por variante na busca (as repetições mostram que as diferenças entre
  as variantes fortes ficam dentro do ruído); só 62 olhos fechados na validação; aplicação testada
  com uma pessoa só; mais olhos não fechados marcados como fechados que no baseline.
- **Próximos passos possíveis:** um refinamento com recorte de alta resolução só dos olhos; mais
  dados de olhos fechados (por exemplo, sequências de vídeo anotadas); calibração do limiar com mais
  pessoas.
