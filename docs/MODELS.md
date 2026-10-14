# Modelos: o que existe, o que falta, e o que muda na T4

Levantado lendo `sdk_alpr` e `synth_plates`, e inspecionando os `.onnx`/`.ort` de produção com a
biblioteca `onnx`. Onde algo não existe, está escrito que não existe.

---

## O que está em produção hoje, verificado

Os três engines (Rust, C++, Go) carregam artefatos **byte-idênticos**.

| | detector | ocr |
|---|---|---|
| opset | 20 | 15 **+ `ai.onnx.ml:2`** |
| nodes | 846 | 500 |
| `DequantizeLinear` | 186 (= todos os 186 `Conv`) | 68 de 83 `MatMul` |
| **QOperator** | **0** | **0** |
| pesos INT8 / FP32 | 372 / 376 | 136 / 196 |

O contrato do `alpr-c/scripts/quantize_cuda_weightonly.py` está de pé: `qoperator_nodes == 0`,
então o grafo roda inteiro no CUDA EP sem cair para CPU. Os 15 `MatMul` não quantizados no OCR são
exatamente os de atenção (ativação×ativação), que o script deixa em float de propósito.

### Dois defeitos encontrados nos artefatos

**1. `ai.onnx.ml:2` é órfão no `ocr.onnx`** — declarado no `opset_import`, e **nenhum node usa**.
O próprio script de quantização tem limpeza para isso, e o docstring diz que domínio órfão
*"quebra o load em runtimes ORT mais antigos (ex.: 1.17.3 do container)"*. A conclusão é que **o
artefato em produção não saiu da versão atual do script**. Rodar de novo o resolve.

**2. Os `.ort` são para mobile, não para servidor.** Ambos carregam `CPUExecutionProvider`
embutido, e são **maiores** que os `.onnx` (3,3 contra 2,2 MB) porque guardam otimizações salvas.
O único conversor `.onnx → .ort` do workspace, `sdk_alpr/scripts/legacy/convert_models.py:604`,
roda com:

```
--optimization_style Runtime --target_platform arm --enable_type_reduction
```

Isso é configuração de **ARM/mobile**. `--enable_type_reduction` corta kernels por tipo, o que só
tira opções de um build CUDA. Num servidor com o runtime completo, o **`.onnx` carrega direto** e
o otimizador trabalha contra o provider que vai de fato executar.

> **Decidido: `.onnx` weight-only INT8 é o formato de produção no servidor.** O serviço já aceita
> — o ORT infere pela extensão — e agora avisa no boot quando recebe um `.ort` numa sessão CUDA.

### A troca, e por que não é retreino

O `ocr.ort` que o servidor carrega é **byte-idêntico** ao `assets/ocr/s.ort` do app Olho de Águia
(`0c6470334658`). Produção está rodando o modelo construído para o celular.

Os pesos estão certos — o `.onnx` correspondente tem `input [1,64,128,3]` NHWC com o rescale
dentro do grafo, `plate [1,10,37]`, `region [1,66]` presente, weight-only INT8 e zero QOperator.
**Nada no treino precisa ser refeito.** O que muda é qual arquivo o serviço abre:

```bash
DETECTOR_MODEL_PATH=/models/detector.onnx
OCR_MODEL_PATH=/models/ocr.onnx
```

### O órfão, corrigido sem tocar em peso

```bash
python3 tools/models/clean_opsets.py ocr.onnx ocr.clean.onnx
#   dropped : ai.onnx.ml:2
#   kept    : ai.onnx:15
```

Verificado depois: assinatura dos pesos e assinatura do grafo **idênticas** antes e depois. É
edição de metadado — nenhuma requantização, nenhuma mudança numérica, nenhum motivo para
revalidar acurácia. O `detector.onnx` não tem órfão e passa direto.

---

## A T4 muda o alvo da quantização

Weight-only INT8 **não acelera nada** no CUDA EP. O comentário do próprio script diz:

> *"Trade-off: é compressão de peso, não ganho de FLOPS (o CUDA EP dequantiza p/ float)."*

O ganho é tamanho, VRAM e tempo de carga. Com **16 GB por placa e sessões que somam poucos GB,
VRAM não é a restrição** — compute é.

Numa T4 (Turing, SM 7.5) a alavanca é **FP16**: os tensor cores existem e o FP32 não os usa. A doc
do ONNX Runtime liga isso explicitamente ao flag que já configurei:

> *"an important flag to use while using an fp16 model as this allows CuDNN to pick tensor core
> algorithms"* — sobre `cudnn_conv_use_max_workspace`, que este serviço agora liga por padrão.

Também **não** existe TF32 na T4: é Ampere para frente. `use_tf32` é no-op no alvo.

**O experimento que decide** é o §7 do briefing: exportar FP16 e medir FP32 × FP16 × weight-only
INT8 na T4, com `bench/fleet`. Nenhum dos três repos tem esse export hoje.

---

## O que NÃO existe no pipeline atual

`sdk_alpr` e `synth_plates` são pipelines **TFLite/mobile**. Isso não é crítica — foram feitos para
o app — mas significa que o caminho para produção em servidor está incompleto:

| Lacuna | Situação |
|---|---|
| `.keras` → `.onnx` | **não existe script.** O `.keras` só vira `.tflite` |
| `.pt` → `.onnx` de produção | os `export_lpr*` produzem TFLite; os `.onnx` em `yolo26/lpr/` são intermediários do `onnx2tf` |
| `.onnx` → `.ort` para x86/CUDA | só o conversor ARM acima |
| opset explícito | em lugar nenhum; `runs/.../args.yaml:71` → `opset: null` |
| validação de acurácia sobre `.onnx`/`.ort` | **não existe.** Todos os validadores carregam TFLite ou Keras |
| `quantize_cuda_weightonly.py` num pipeline | órfão: nenhum `.sh`, `.md`, `Dockerfile` ou CI o referencia |

O único portão que um `.ort` atravessa hoje é o
[`modelcontract.go`](../engine/onnx/modelcontract.go) deste serviço, que valida
shape, nome, contagem de classes e SHA no boot — mas **não mede acurácia**. Para isso existe
`internal/corpus/`, e ele precisa das imagens reais.

---

## O finetune BR perde o head de região — em dois lugares independentes

Este é o achado que mais afeta o produto.

**1. No treino.** `sdk_alpr/scripts/train_cct_v2_br.py:93-114` envolve o modelo global num
`keras.Model` com output só `{'plate': ...}`. O comentário na linha 96 é explícito:

> *"Wrapping desconecta a região do gradient graph — esses layers ficam frozen automaticamente."*

E `README_train_br.md:90` confirma: *"o wrap dropa região; `best.keras` sai com output só
`{'plate': (10, 37)}`"*.

**2. No export.** `sdk_alpr/scripts/export_cct_v2_int8.py:114-115`:

> *"TFLiteConverter falha com multi-output dict — extrai só o `plate` head (region é metadata, não
> usado no pipeline OCR)"*

O config `cct_xs_v2_br_plate_config.yaml` diz a mesma coisa no cabeçalho e **omite `plate_regions`
de propósito**.

**Por que importa aqui:** o produto é global. A região é o campo que diz de que país é a placa, e um
finetune BR feito por esse caminho produz um modelo sem ela. Este serviço trata isso como
degradação e não como falha — ele boota, e `RegionCount` fica 0 — mas toda leitura sai sem país.

Se o finetune BR for para produção, **um dos dois pontos precisa mudar**: manter a região no wrap
(e no `plate_regions` do config), ou aceitar que o modelo BR é só-placa e que a região vem do
modelo global.

---

## Onde estão os dados

**Reais**, em `sdk_alpr/backups/` (24 chunks de 95 MB, ~1,8 GB), com proveniência documentada:

| Dataset | Imagens |
|---|---|
| OCR V5 supermerged (28 datasets Roboflow) | 23.067 |
| LPR V1 | 16.740 |
| OCR V1 BR | 6.319 |

A limpeza do V5 está toda registrada em `backups/README.md:22-36` — dedup por SHA1 e pHash,
descarte de fora-de-faixa, e **4.805 ground-truths corrigidos por consenso de 3 OCRs**. Vale ler:
é o que diz a qualidade do rótulo.

Anotações BR já filtradas e **commitadas** em `sdk_alpr/configs/br_annotations/`: train 9.403,
val 2.693, test 1.345. Os CSVs existem; as imagens que eles referenciam **não** — precisa restaurar
o backup primeiro. E os paths são Windows (`..\..\datasets\...`).

**Sintéticos:** zero imagens no disco. `synth_plates` gera 128k sob demanda em 30-60 min
(`1_gen_br_dataset.sh`), split 70/15/15.

### O melhor resultado até agora

| | val_acc | nota |
|---|---|---|
| sintético (`br_synth_v1`, 106 epochs) | **0,9950** | val em dados sintéticos |
| finetune BR real (`23-31-52`, 76 epochs) | **0,8557** | alvo é ≥0,85 |

A distância entre os dois é o gap de domínio, e `synth_plates/training/validate_real.py:8-9` já o
nomeia: *"validacao em placas REAIS revela gap de domain (synth -> real). Esperado: queda 2-10pp"*.
Aqui a queda foi de 14pp.

---

## Armadilhas já pagas, que valem para o trabalho novo

Cada uma custou tempo a alguém.

**`mixed_float16` quebra o `TokenReducer`.** Dois runs de treino saíram com log vazio e sem
checkpoint por causa disso (`runs/cct_xs_v2_br/23-25-25` e `23-26-25`). A causa está em
`synth_plates/training/_keras_compat.py:4-17`: a layer custom soma tensores sem cast, e sob mixed
precision recebe fp16 e fp32 juntos. Workaround: `force_fp32()` depois do load. Commit `48f33d2`
desligou mixed precision por padrão.

**Double-softmax.** `sdk_alpr/ui/local_tflite.py:277`: o head do CCT já aplica softmax, e aplicar
de novo *"colapsa as confianças para ~1/N uniforme ~0.07"*. **É o mesmo invariante que este serviço
registra** e que `decode.go` protege com teste — encontrado independentemente dos dois lados.

**INT8 destrói ONNX end-to-end com NMS no grafo.** `export_lpr_onnx2tflite.py:12-19`: quantizar
esses ops produz *"bboxes ≈ 0 com confidence = 1.0 em toda detecção"*. Sintoma a reconhecer.

**Contagem de calibração é oposta entre os dois modelos.** O CCT **quebra** com mais de 100
imagens de calibração (ranges agressivos demais); o detector YOLO quer 100-400 e ganha nada acima
disso. Um número não serve para os dois.

**Calibrar em sintético enviesa.** `export_ocr_tflite.py:5-14`: *"Calibrating on synthetic
distribution biases INT8 ranges away from real-world production data → quantized model
hallucinates"*. Existe `build_ocr_calib_mixed.py` para 50/50.

**Converter que aceita ≠ runtime que roda.** `export_cct_v2_int8.py:79-95` sempre faz um invoke
dummy antes de escrever o arquivo, para pegar coisas como `Type 'INT16' is not supported by
tile.Node`. **Padrão que este serviço já copia**: o warmup roda três inferências antes de a sessão
entrar no pool, e checa que os scores saem em `[0,1]`.
