# Rodando o serviço ALPR

Três coisas rodam em máquinas diferentes, e a divisão não é arbitrária: o que não precisa de GPU
roda no laptop e no CI, e o que precisa roda no servidor. Misturar as duas é como se produz um
número que não significa nada.

---

## 1. No laptop — o pipeline inteiro, sem GPU

```bash
# um vídeo qualquer serve; sem `-re` o ffmpeg despeja o arquivo de uma vez
ffmpeg -f lavfi -i testsrc=size=640x480:rate=15:duration=6 -pix_fmt yuv420p clip.mp4

go run ./cmd/alpr-smoke -input clip.mp4 -seconds 6 -detect-every 250ms
```

Roda o subprocesso ffmpeg de verdade, o ring de buffers, o gate na ordem barato→caro, o dedup e o
encode JPEG. A inferência é `engine.Fake` — **é o desenho, não uma limitação**: tudo entre a câmera
e o modelo é exercitado de verdade, e o modelo é a única peça que precisa de CUDA.

Também aceita RTSP, e é assim que se testa contra câmera de verdade:

```bash
go run ./cmd/alpr-smoke -input "rtsp://usuario:senha@camera/stream" -seconds 30
```

> Erro nunca carrega credencial. Há teste que falha se carregar, porque essas mensagens vão para o
> painel de conectividade que o operador lê.

### Portões que precisam passar

```bash
go build ./... && go vet ./... && go vet -tags cuda ./...
go test ./... && go test -race ./...
```

`go vet -tags cuda` roda **nativo**, não com `GOOS=linux`: `onnxruntime_go` usa cgo, e
cross-compilar sem toolchain simplesmente exclui os arquivos — o vet passaria sem ter olhado nada.

### A superfície HTTP

O binário serve quatro grupos de rotas num listener só (`ALPR_HTTP_ADDR`, default `:8080` — a
mesma porta do worker Python).

| Rota | Auth | O que é |
|---|---|---|
| `GET /` · `GET /health` | **nenhuma** | paridade com o Python. O `/health` responde **200 sempre**, porque é o que o container faz `curl` |
| `GET /cameras` · `GET /cameras/{id}/status` · `POST /cameras/sync` · `GET /stats` | Bearer | a API de frota do Python, mesmos corpos |
| `POST /ocr/recognize` · `WS /ocr/ws` | Bearer | OCR avulso: manda a imagem, recebe a placa |
| `WS /ws/{camera_id}` | **Bearer** | overlay ao vivo do painel |
| `GET /_alpr/status` · `/metrics` · `/health` · `/diagnose` | nenhuma | as deste serviço. O `/_alpr/health` responde **503 com o motivo** |

Os dois `health` existem de propósito e respondem perguntas diferentes: um decide se o
orquestrador mata o processo, o outro conta a uma pessoa **por quê**.

> ⚠️ **O `WS /ws/{camera_id}` exige Bearer aqui, e o Python não exigia nada.** No Python, quem
> soubesse um `camera_id` recebia as bboxes e podia mandar `{"type":"config"}` alterando o overlay
> de outra pessoa. O engine Rust já exigia o token na mesma rota. **O painel do OdA precisa passar
> a mandá-lo** — no header ou em `?token=`, porque navegador não manda header em handshake de
> WebSocket. É a única mudança fora deste repositório.

Uma diferença de contrato que **destrava** um cliente: `POST /ocr/recognize` aceita
`detectionId: null`. O Python respondia 422, e o único cliente real manda `null` porque foi
escrito contra o Rust.

```bash
curl -s -H "Authorization: Bearer $ALPR_BOT_TOKEN" localhost:8080/stats | jq .
curl -s -H "Authorization: Bearer $ALPR_BOT_TOKEN" localhost:8080/ocr/recognize \
     -d '{"detectionId":null,"image":"'"$(base64 -i placa.jpg)"'"}' | jq .
```

---

## 2. No servidor CUDA — a medição

É isto que fecha o critério de aceitação. Sem número, a migração não está pronta.

```bash
export ONNXRUNTIME_LIB_PATH=/usr/local/lib/libonnxruntime.so

# os clipes viram RTSP de tempo real; o -re é estrutural, não cosmético
go run -tags cuda ./bench/fleet \
    -clips ./clips \
    -detector /models/detector.onnx \
    -ocr /models/ocr.onnx \
    -devices 0,1 \
    -sessions-per-gpu 2,4 \
    -start 8 -step 8 -max 128 \
    -warmup 60s -window 10m \
    -slo-p95 1.5 \
    -hwaccel cuvid \
    -out run.json
```

`-hwaccel cuvid` liga o NVDEC. Sem ele, decodificar uma câmera 1080p30 custa 5,6% de um core —
treze vezes o que o pipeline Go inteiro custa com os mesmos frames — e o número "câmeras por GPU"
que sair da varredura pode ser um fato sobre o host, não sobre a placa. Requer ffmpeg compilado com
`--enable-cuvid` e a capability `video` no container, como nos `Dockerfile` do Rust e do C++.

A varredura sobe o número de câmeras até o p95 de `alpr_pipeline_duration_seconds` cruzar o SLO.
**Esse cruzamento _é_ o número "câmeras por GPU"** — defensável de um jeito que "rodamos 40 e
pareceu bom" não é.

Cada linha de saída leva a configuração inteira, o SHA do git e o SHA256 de cada modelo. Sem isso,
duas execuções não são comparáveis, e todo mundo descobre isso do jeito difícil.

```bash
go run ./bench/compare baseline.json run.json   # imprime as regressões
```

### Conferir onde os modelos realmente rodaram

```bash
curl -s localhost:PORT/_alpr/status | jq '.providers'
```

`all_cuda: false` é **incidente**, não observação. Um modelo que caiu para CPU não dá erro: ele
fica lento, e lento é diagnosticado como problema de câmera ou de rede.

> O `all_cuda` do engine C++ irmão nunca mediu isto. Lá o provider é gravado *antes* da sessão
> existir, então reporta o que foi **pedido**. A versão Go lê o placement por nó do log do ONNX
> Runtime, e **relatório vazio não conta como sucesso** — nada foi medido.

---

## 3. O portão de acurácia — Python × Go

O risco real da migração mora aqui. Vazão falha visivelmente; acurácia falha em silêncio.

```bash
# lado Go (precisa dos modelos)
go run ./internal/corpus/rungo -detector /models/detector.onnx -ocr /models/ocr.onnx

# lado Python (dentro do ambiente do alpr-py, com OS MESMOS PESOS)
python3 internal/corpus/run_py.py

# modelo contra modelo
go run ./internal/corpus/compare -gates=off

# quanto os gates do Go descartam
go run ./internal/corpus/compare -gates=on
```

**Os dois modos existem por uma assimetria estrutural.** O `fast_alpr` detecta e lê numa chamada só
e filtra depois; o Go separa e descarta um candidato **antes** do OCR. O Go produz legitimamente
menos leituras — sem esse interruptor, **todo gate parece bug**.

O comparador é tolerante por campo, porque igualdade bit a bit é a régua errada:

| campo | tolerância | o que uma violação significa |
|---|---|---|
| placa | **exata** | pares de confusão repetidos (`8→B`, `0→O`) = alfabeto indexado com deslocamento |
| box | IoU ≥ 0,90 | IoU baixo = coordenada de letterbox vazando para fora do detector |
| confiança | 1e-3 | maior que isso é **pré-processamento**, não arredondamento |
| região | exata | divergência sistemática = tabela posicional reordenada, **indetectável em runtime** |

Quando a confiança diverge, a mensagem de falha lista os quatro suspeitos **nesta ordem**: cor do
padding (114), interpolação do resize, ordem BGR/RGB, e a escala da entrada — o OCR recebe
`[0,255]` e **não** pode ser dividido por 255.

### O corpus

`internal/corpus/frames/` tem seis frames sintéticos que provam o encanamento. **Os reais são
seus:** ~200 imagens PNG com a leitura humana de cada placa.

PNG, nunca JPEG — um codec com perda faz as duas implementações discordarem já no decode, e essa
diferença é indistinguível de um bug de pré-processamento no relatório.

Cada entrada do manifesto leva `notes` dizendo para que aquele frame serve. Um corpus sem isso
apodrece numa pasta de imagens que ninguém tem coragem de apagar.

---

## Onde o serviço se encaixa

`./` é Go puro — **não importa arandu**, e um teste falha se isso mudar. O
acoplamento com o framework vive só em `app/Modules/alpr/`.

A aplicação implementa duas interfaces e nada mais:

- `contract.Source` — quais câmeras rodar (snapshot inteiro, nunca delta)
- `contract.Sink` — o que fazer com uma leitura

O serviço não decide cobrança, organização, usuário, alerta, autorização, persistência, webhook,
plano nem assinatura.

> `Sink.Detect` é chamado **na goroutine da câmera** e a bloqueia. Quem decide "INSERT antes da
> imagem" é o Sink, com a linha e o id na mão. Do lado de dentro do serviço, a ordem errada não é
> expressável — em 19/08 ela derrubou a taxa de notificação de 100% para menos de 15% em meia hora.

### A inversão do `main.go`

Uma aplicação que usa o engine ONNX **precisa** ceder `main.main` ao construtor de sessões:

```go
func main() {
    builder := onnx.NewBuilder(cfg)
    if err := builder.Init(); err != nil { log.Fatal(err) }

    ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
    defer cancel()

    go bootstrap.Dispatch()                 // a aplicação, ao lado
    log.Fatal(builder.Serve(ctx))           // BLOQUEIA na thread principal
}
```

Criar sessão CUDA fora da thread principal termina em `SIGSEGV` (ONNX Runtime #26610), e
`runtime.LockOSThread` prende a goroutine a **alguma** thread, não à principal. Ver
`docs/arandu-gaps.md`, GAP-4.

O `Builder.Init` recusa bootar se a suposição não valer — uma frase de erro no lugar de um segfault
sem stack Go legível.
