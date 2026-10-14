# Onde o tempo vai

Medido, não estimado. Máquina: Apple M4 Max, 1080p30, detecção a cada 200 ms.
Reproduzir com `go test ./imaging/ -bench . -benchtime 2s -count 3`.

Metade do mapa está medida aqui e metade precisa do servidor. A divisão é honesta e está marcada.

---

## O que está medido

### Custo por câmera, por segundo

| | ms/s | % de 1 core |
|---|---:|---:|
| **ffmpeg** decode h264 + conversão bgr24 | 56,0 | **5,60 %** |
| Go: letterbox | 4,1 | 0,41 % |
| Go: foco + crop | 0,3 | 0,03 % |
| Go: encode JPEG | ~0 | ~0 % |
| Go: laço de câmera | ~0 | ~0 % |
| **Go, total** | **4,4** | **0,44 %** |

> **O código Go é 13× mais barato que o ffmpeg que o alimenta.** A migração de Python para Go
> comprou a parte pequena da conta de CPU. Isso não invalida a migração — ela comprou também
> previsibilidade de memória e a ausência do GIL — mas significa que otimizar mais o Go não move
> a capacidade do servidor.

### Escala

| | ffmpeg | Go | banda de memória |
|---|---:|---:|---:|
| 40 câmeras | 2,2 cores | 0,17 cores | 7,5 GB/s |
| 100 câmeras | 5,6 cores | 0,44 cores | 18,7 GB/s |
| 200 câmeras | 11,2 cores | 0,87 cores | 37,3 GB/s |

A banda é o número que ninguém olha e que morde primeiro. São os frames decodificados atravessando
o pipe: 1080p BGR24 são 6,2 MB por frame, e a 30 fps isso é 186 MB/s **por câmera**.

### Detalhes que mudam decisão

**O letterbox escala com a área de CONTEÚDO, não com a resolução de entrada.**

| entrada | conteúdo em 640×640 | medido |
|---|---|---:|
| 640×480 (4:3) | 640×480 = 307.200 px | 1066 µs |
| 1280×720 (16:9) | 640×360 = 230.400 px | 817 µs |
| 1920×1080 (16:9) | 640×360 = 230.400 px | 812 µs |
| 3840×2160 (16:9) | 640×360 = 230.400 px | 822 µs |

4K custa o mesmo que 1080p. **Uma câmera 4:3 custa 31 % a mais que uma 16:9** (previsto 1,333,
medido 1,313), porque preenche mais do quadrado e sobra menos padding. Resolução é quase de graça;
proporção de tela não é.

**O ajuste de brilho/contraste em 100/100 custa 1,5 ns** — o laço nem chama a função. Fosse
chamada, seriam 1,5 ms por frame, 45 ms/s por câmera: quase o custo do ffmpeg inteiro, para
multiplicar por um.

---

## O alvo: 2× Tesla T4 de 16 GB, câmeras 1080p

### Decodificação em hardware — feita

A T4 tem NVDEC, um bloco de silício **separado dos SMs**. Decodificar ali não custa CPU e não
disputa nada com a inferência: é capacidade que de outro modo fica parada.

Os engines Rust e C++ já carregam essa correção sob `STREAM_HWACCEL`, e o comentário do
`alpr-rs/crates/alpr-stream/src/rtsp.rs` cita a orientação da própria NVIDIA — *em pipelines
multi-câmera o decodificador de vídeo, não a inferência, costuma ser o que satura primeiro*.
Medi isso por conta própria antes de encontrar o comentário, o que é uma boa confirmação.

Portado para o Go em `streaming.Config.HWAccel`, com duas diferenças em relação à referência:

- **O decoder é escolhido pelo codec que o `ffprobe` já descobriu.** Rust e C++ obrigam o operador
  a escolher entre `cuvid` e `hevc_cuvid`, e escolher errado significa câmera que não abre. A
  pergunta já tinha resposta.
- **`-hwaccel_device` segue o placement da câmera.** As referências não têm isso — com uma placa
  não importa. Com duas importa: sem ele **toda** câmera decodifica no device 0, e o NVDEC da
  outra fica ocioso enquanto as duas reportam saudáveis.

`-c:v h264_cuvid` é passado explicitamente, como nas referências: `-hwaccel cuvid` sozinho volta
em silêncio para o decoder de software em alguns builds do ffmpeg — o pior resultado possível,
porque parece decodificação em hardware que não ajudou.

### O que isso muda no orçamento

| 1080p30, por câmera | CPU | NVDEC |
|---|---:|---:|
| decode em CPU | 5,60 % de core | — |
| decode em NVDEC | ~0 | 1 stream |
| Go, em qualquer caso | 0,44 % de core | — |

Com NVDEC ligado, **a CPU deixa de ser a conversa**: sobram 0,44 % de core por câmera, e mesmo
200 câmeras cabem em menos de um core.

### O lever que ainda não foi puxado: FP16

A quantização é **weight-only INT8** — os pesos são INT8, as ativações e a entrada são float. O
CUDA EP desempacota os pesos e roda convolução em FP32. **Não há ganho de compute**, só de VRAM e
banda de memória.

Numa T4 isso deixa performance na mesa: a placa tem Tensor Cores que fazem FP16 numa ordem de
grandeza acima do FP32. Exportar variantes FP16 e medir as três é exatamente o que o §7 pede, e é
mudança de modelo, não de código.

Com 16 GB por placa, **VRAM não é a restrição** — as sessões deste serviço somam poucos GB. A
restrição é compute, e é aí que o FP16 vive.

---

## A otimização que a medição encontrou

**39 % do custo de CPU do ffmpeg é a conversão yuv420p → bgr24**, e ela existe só porque o pipeline
pede BGR24.

```
decode h264            1,13 ms/frame
decode + bgr24         1,87 ms/frame
   a conversão          0,73 ms/frame   ← 39 %
```

Pedir `yuv420p` ao ffmpeg e converter dentro do laço do letterbox — que já toca cada pixel de saída
e já faz BGR→RGB — pagaria duas vezes:

- **−39 % da CPU do ffmpeg**, o componente dominante
- **metade dos bytes no pipe**: yuv420p são 1,5 bytes/px contra 3 do BGR24, o que corta a banda pela
  metade e o ring de 6,2 MB para 3,1 MB por buffer

Não foi feito. O motivo é o §30: a GPU ainda não foi medida, e otimizar o segundo gargalo antes de
conhecer o primeiro é otimizar no escuro. O número está aqui para quando a medição do servidor
disser que vale.

Risco conhecido: a subamostragem de croma do yuv420p significa que o resize precisa tratar luma e
croma em resoluções diferentes. É mais código no caminho quente, e o corpus é o que diria se mudou
alguma leitura.

---

## O que NÃO está medido

Tudo que precisa de CUDA, e é onde o gargalo provavelmente está:

- FPS real por GPU, e latência p50/p95/p99 com inferência de verdade
- VRAM em pico e em regime
- **câmeras por GPU** — o número que o produto vende
- comparação de vazão contra o serviço Python no mesmo hardware
- se algum nó cai para CPU em silêncio (`/_alpr/status` responde, e `all_cuda: false` é incidente)

`bench/fleet` produz todos eles. Ver [RUNNING.md](RUNNING.md).

**A previsão que a medição do servidor deve confirmar ou derrubar:** com NVDEC ligado, a GPU satura
antes da CPU por uma margem larga. O serviço Python rodava ~39 câmeras em produção; se as duas T4
não superarem isso com folga, o primeiro lugar a olhar é `/_alpr/status` — `all_cuda: false`
significa que parte do grafo caiu para CPU, e aí o número mede a coisa errada.

Ordem de investigação se a vazão decepcionar:

1. `all_cuda` — fallback silencioso para CPU é a explicação mais comum e não dá erro
2. **FP16** — a T4 tem Tensor Cores que o FP32 não usa; é a maior alavanca restante
3. `-hwaccel cuvid` — confirmar que o ffmpeg do container foi compilado com `--enable-cuvid`
4. a conversão yuv420p acima, se a CPU ainda aparecer no perfil

---

## Onde as alocações foram

| | antes | depois |
|---|---:|---:|
| frame decodificado | 6.226.019 B | **96 B** |
| letterbox | — | **0** |
| crop | — | **0** |
| variância de foco | — | **0** |
| NMS sobre 8400 âncoras | — | **0** (4,4 µs) |
| laço de câmera em regime | — | **0** (750 ns) |

O ring de três buffers por câmera substituiu o `make()` por frame. `sync.Pool` foi rejeitado: ele é
**esvaziado a cada ciclo de GC**, então um buffer de 6 MB seria realocado logo depois de toda
coleta — pior exatamente quando a heap já está sob pressão.

Sobram 96 B por frame: é o `Frame` que o `atomic.Pointer` publica, não a imagem.
