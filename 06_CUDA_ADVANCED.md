# Kurseinheit 6:  Fortgeschrittenes GPU Computing mit CUDA
## Shared Memory, Atomare Operationen, Streams und Pinned Memory

---

## Lernziele

Nach diesem Kapitel sollten Sie in der Lage sein:
* die vollständige GPU-Speicherhierarchie (Register, Shared Memory, Constant Memory, L2, Global Memory) zu beschreiben und situationsgerecht einzusetzen.
* statischen und dynamischen Shared Memory zu deklarieren, mit `__syncthreads()` korrekt zu synchronisieren und Bank Conflicts zu vermeiden.
* das Tiling-Prinzip auf die Faltung anzuwenden und Input-Tile-Größe aus Output-Tile-Größe und Maskenbreite herzuleiten.
* Constant Memory für read-only Broadcast-Daten einzusetzen (`__constant__`, `cudaMemcpyToSymbol`).
* atomare Operationen einzusetzen und deren Performance-Problem durch Privatisierung mit Shared Memory zu lösen.
* den Unterschied zwischen normaler und gepinnter Host-Memory-Allokation zu erklären und dessen Auswirkung auf `cudaMemcpy`-Geschwindigkeit zu benennen.
* CUDA-Streams zu erstellen und `cudaMemcpyAsync` mit Kernel-Starts zu überlappen (Pipelining).
* CUDA Events für präzise Zeitmessung und stream-übergreifende Synchronisation einzusetzen.

---

## 1. GPU-Speicherhierarchie im Detail

```
SM-0        SM-1        ...  SM-N
Register    Register         Register
L1/SMEM     L1/SMEM          L1/SMEM
         ↘      ↓      ↙
            L2 Cache        (geteilt von allen SMs)
               ↓
          Global Memory
```

| Speicherebene | Sichtbar für | Latenz | Besonderheit |
|---|---|---|---|
| **Register** | Einzelner Thread | ~1 Zyklus | Compiler entscheidet über Nutzung; Register Spilling → Local Memory |
| **Shared Memory (SMEM)** | Alle Threads im Block | ~5–20 Zyklen | Programmierbarer On-Chip-Cache; ~48–192 kB pro SM |
| **Constant Memory** | Alle Threads (read-only) | ~1 Zyklus (gecacht) | Ideal für Masken/Koeffizienten, die alle Threads gleich lesen |
| **Texture Memory / RO-Cache** | Alle Threads (read-only) | Gecacht | Optimiert für 2D-Lokalität |
| **L2 Cache** | Alle SMs | ~100–200 Zyklen | Automatisch, nicht direkt steuerbar |
| **Global Memory** | Alle Threads + Host | ~200–800 Zyklen | Größter Speicher; Flaschenhals bei speicherintensiven Kerneln |

> **Zentrale Idee:** Daten, die von mehreren Threads eines Blocks gemeinsam genutzt werden, sollten einmalig aus dem Global Memory in den **Shared Memory** geladen werden – statt dass jeder Thread denselben Wert einzeln aus dem langsamen Global Memory liest.

---

## 2. Shared Memory

### Motivation

Globale Speicherzugriffe sind der größte Flaschenhals bei speicherintensiven GPU-Anwendungen. Shared Memory ist ein **programmierbarer Cache** auf dem SM-Chip:
* Alle Threads eines Blocks teilen sich denselben Shared Memory.
* Zugriff ist ~10–100× schneller als Global Memory.
* Da Threads in einem Block zwar logisch parallel sind, aber in der Realität als **Warps** nicht alle gleichzeitig ausgeführt werden, ist **Synchronisation** zwingend notwendig.

### `__syncthreads()`

```c
__syncthreads();
```

Barrier für **alle Threads im Block**: Kein Thread darf die Zeile überschreiten, bis alle Threads des Blocks sie erreicht haben. Typisches Muster:

1. Threads laden gemeinsam Daten in Shared Memory.
2. `__syncthreads()` → sicherstellen, dass alle Ladevorgänge abgeschlossen sind.
3. Threads rechnen mit den Daten im Shared Memory.

> **Achtung:** `__syncthreads()` synchronisiert nur Threads **innerhalb eines Blocks**, nicht über Blöcke hinweg.

### Statischer Shared Memory

Größe wird zur **Compile-Zeit** festgelegt:

```c
__global__ void staticReverse(int *d, int n)
{
    __shared__ int s[64];       // 64 ints im Shared Memory, feste Größe
    int t  = threadIdx.x;
    int tr = n - t - 1;

    s[t] = d[t];                // Phase 1: alle Threads laden
    __syncthreads();            // Barriere: warten bis alle geladen haben
    d[t] = s[tr];               // Phase 2: alle Threads lesen gespiegelt
}
```

```c
// Host-Aufruf:
staticReverse<<<1, n>>>(d_d, n);
```

### Dynamischer Shared Memory

Größe wird beim **Kernel-Start** übergeben – nützlich, wenn die Größe erst zur Laufzeit bekannt ist:

```c
__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int s[];  // Größe wird beim Kernel-Start übergeben
    int t  = threadIdx.x;
    int tr = n - t - 1;

    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}
```

```c
// Host-Aufruf: dritter <<<...>>>-Parameter = Shared-Memory-Größe in Byte
dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
```

### Bank Conflicts

Shared Memory ist physisch in **32 Bänke** (Banks) aufgeteilt. Zugriffe verschiedener Threads auf **verschiedene Bänke** laufen vollständig parallel. Greifen jedoch mehrere Threads desselben Warps auf **dieselbe Bank** zu, werden die Zugriffe **serialisiert** – ein sogenannter **Bank Conflict**.

**Bank-Zuordnung:** Aufeinanderfolgende 4-Byte-Wörter liegen in aufeinanderfolgenden Bänken (Bank 0, 1, 2, …, 31, dann wieder Bank 0, …).

```
Bank:      0    1    2    3    ...   31    0    1    ...
Adresse:  [0]  [1]  [2]  [3]  ...  [31] [32] [33]  ...
```

**Konfliktfreier Zugriff (optimal):**
```c
// Jeder Thread greift auf eine andere Bank zu
s[threadIdx.x]           // Thread i → Bank i % 32  ✓
```

**2-facher Bank Conflict:**
```c
// Threads 0 und 16 greifen beide auf Bank 0 zu (Stride = 16)
s[threadIdx.x * 2]       // Thread 0 → Bank 0, Thread 1 → Bank 2, ..., Thread 16 → Bank 0  ✗
```

**Kein Conflict – Broadcast:**
```c
// Alle Threads lesen denselben Wert → Broadcast, keine Serialisierung
s[0]                     // alle Threads → Bank 0, aber identische Adresse → OK  ✓
```

**Praxisregel:** Shared-Memory-Arrays mit Stride 1 (jeder Thread liest das nächste Element) sind konfliktfrei. Stride 32 ist der schlimmste Fall (alle Threads auf dieselbe Bank) → 32-fache Serialisierung.

---

## 3. Faltung mit Shared Memory (Tiling)

### Das Problem der Basisimplementierung

Im einfachen Faltungs-Kernel (vorherige Einheit) liest **jeder Thread** alle seine benötigten Eingabewerte aus dem globalen Speicher. Benachbarte Ausgabeelemente brauchen jedoch **überlappende Eingabebereiche** – dieselben Werte werden mehrfach aus dem  gloabelen Speicher  gelesen.

Beispiel: Bei einer 1D-Faltung mit Maskenbreite 5 wird `N[2]` für die Berechnung von `P[0]`, `P[1]`, `P[2]`, `P[3]` und `P[4]` benötigt.

### Tiling-Prinzip

Man unterteilt das Ausgabe-Array in **Output-Tiles** der Breite `O_TILE_WIDTH`. Für jeden Output-Tile lädt der Block alle benötigten Eingabewerte **einmalig** in den Shared Memory:

```
Input-Tile-Breite  = O_TILE_WIDTH + (Mask_Width - 1)
                   = O_TILE_WIDTH + 2 * (Mask_Width / 2)
```

### Zwei Design-Varianten

| | **Design 1** | **Design 2** (implementiert) |
|---|---|---|
| Block-Größe | = Output-Tile-Größe | = Input-Tile-Größe |
| Threads beim Laden | Einige laden mehrere Elemente | Jeder Thread lädt genau ein Element |
| Threads bei Berechnung | Alle | Nur die Threads mit gültigem Output-Index |

### Index-Beziehung zwischen Input und Output

```
index_o = O_TILE_WIDTH * blockIdx.x + threadIdx.x    // Output-Index
index_i = index_o - Mask_Width / 2                   // Input-Index (kann negativ sein → Ghost-Element)
```

### 1D-Faltungs-Kernel mit Shared Memory

```c
#define O_TILE_WIDTH 1020
#define I_TILE_WIDTH (O_TILE_WIDTH + 4)   // Mask_Width = 5 → +4

__global__ void convolution_1D_kernel(float *N, float *M,
                                       float *P, int Mask_Width, int Width)
{
    __shared__ float Ns[I_TILE_WIDTH];

    int index_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int index_i = index_o - Mask_Width / 2;

    // Phase 1: Jeder Thread lädt ein Eingabeelement in den Shared Memory
    if (index_i >= 0 && index_i < Width)
        Ns[threadIdx.x] = N[index_i];
    else
        Ns[threadIdx.x] = 0.0f;           // Ghost-Element: Zero-Padding

    __syncthreads();                       // warten bis alle geladen haben

    // Phase 2: Nur Output-Threads berechnen
    if (threadIdx.x < O_TILE_WIDTH) {
        float Pvalue = 0.0f;
        for (int j = 0; j < Mask_Width; j++)
            Pvalue += M[j] * Ns[j + threadIdx.x];
        P[index_o] = Pvalue;
    }
}
```

**Host-Aufruf:**
```c
#define BLOCK_WIDTH I_TILE_WIDTH   // = O_TILE_WIDTH + (Mask_Width - 1)

dim3 dimBlock(BLOCK_WIDTH, 1, 1);
dim3 dimGrid((Width - 1) / O_TILE_WIDTH + 1, 1, 1);
// BLOCK_WIDTH sollte ein Vielfaches von 32 sein!
```

### 2D-Faltungs-Kernel mit Shared Memory

```c
#define O_TILE_WIDTH 28
#define I_TILE_WIDTH (O_TILE_WIDTH + 4)   // Mask_Width = 5

__global__ void convolution_2D_kernel(float *N, float *M, float *P,
                                       int Mask_Width, int Width, int Height)
{
    __shared__ float Ns[I_TILE_WIDTH][I_TILE_WIDTH];

    int x    = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int x_in = x - Mask_Width / 2;
    int y    = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
    int y_in = y - Mask_Width / 2;

    // Phase 1: Laden (mit Ghost-Element-Behandlung)
    if (x_in >= 0 && x_in < Width && y_in >= 0 && y_in < Height)
        Ns[threadIdx.y][threadIdx.x] = N[x_in + Width * y_in];
    else
        Ns[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Phase 2: Nur Output-Threads berechnen
    if (threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
        float Pvalue = 0.0f;
        for (int j = 0; j < Mask_Width; j++)
            for (int k = 0; k < Mask_Width; k++)
                Pvalue += M[j * Mask_Width + k] * Ns[j + threadIdx.y][k + threadIdx.x];
        P[y * Width + x] = Pvalue;
    }
}
```

---

## 4. Constant Memory

### Motivation

Die Faltungsmaske `M` wird von **allen Threads identisch** gelesen – jeder Thread braucht dieselben Gewichte. Sie in den Global Memory zu legen ist ineffizient: 32 Threads × gleiche Adresse = 32 separate Anfragen. Der **Constant Memory** löst das durch einen dedizierten Cache mit **Broadcast-Mechanismus**: Wenn alle Threads desselben Warps dieselbe Adresse lesen, wird der Wert einmal gelesen und an alle gleichzeitig gesendet.

### Eigenschaften

* Read-only für Kernel-Code.
* 64 kB pro GPU, gecacht pro SM.
* Latenz wie L1-Cache (~1–5 Zyklen) bei Cache-Hit und identischer Adresse aller Threads.
* Bei unterschiedlichen Adressen (verschiedene Threads lesen verschiedene Positionen) wird der Zugriff serialisiert → Constant Memory nur sinnvoll, wenn alle Threads dasselbe lesen.

### Deklaration und Befüllung

Constant Memory wird im **globalen Scope** (außerhalb von Funktionen) deklariert und vom Host mit `cudaMemcpyToSymbol` befüllt:

```c
#define MAX_MASK_WIDTH 10

__constant__ float M[MAX_MASK_WIDTH];   // im Constant Memory

// Host-Code: Maske in Constant Memory kopieren
cudaMemcpyToSymbol(M, h_M, Mask_Width * sizeof(float));
// (kein cudaMalloc nötig – Constant Memory ist statisch alloziert)
```

### Faltungs-Kernel mit Constant Memory

```c
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution_1D_const_kernel(float *N, float *P,
                                             int Mask_Width, int Width)
{
    __shared__ float Ns[I_TILE_WIDTH];

    int index_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int index_i = index_o - Mask_Width / 2;

    Ns[threadIdx.x] = (index_i >= 0 && index_i < Width) ? N[index_i] : 0.0f;
    __syncthreads();

    if (threadIdx.x < O_TILE_WIDTH) {
        float Pvalue = 0.0f;
        for (int j = 0; j < Mask_Width; j++)
            Pvalue += M[j] * Ns[j + threadIdx.x];   // M aus Constant Memory
        P[index_o] = Pvalue;
    }
}
```

Da `M` von allen Threads in derselben Reihenfolge gelesen wird und der Constant-Memory-Cache breit genug für eine typische Maske ist, entfallen alle Global-Memory-Zugriffe auf die Maske vollständig.

---

## 5. Atomare Operationen

### Motivation: Histogramm-Berechnung

Ein Histogramm zählt, wie oft ein Wert in einem bestimmten Bereich (Bin) vorkommt. Anwendungen:
* Merkmalsextraktion in der Bildverarbeitung (z. B. Objekterkennung).
* Betrugserkennung bei Transaktionsdaten.
* Astrophysik: Korrelation von Bewegungen.

**Problem:** Mehrere Threads wollen gleichzeitig denselben Bin-Zähler inkrementieren → **Race Condition**.

### `atomicAdd` und andere atomare Operationen

Eine atomare Operation führt Lesen, Modifizieren und Schreiben als **unteilbare Einheit** aus – kein anderer Thread kann dazwischenfunken:

```c
atomicAdd(int *address, int val)     // *address += val, atomar
atomicSub, atomicMin, atomicMax, atomicAnd, atomicOr, atomicXor, atomicExch, atomicCAS
```

### Einfacher Histogramm-Kernel

```c
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
    int i      = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        int pos = buffer[i] - 'a';
        if (pos >= 0 && pos < 26)
            atomicAdd(&histo[pos / 4], 1);   // Bin-Breite = 4 Buchstaben
        i += stride;
    }
}
```

### Performance-Problem: Globale Atomics

`atomicAdd` auf **Global Memory** ist teuer: Das Lesen und Schreiben auf DRAM hat je ~hunderte Zyklen Latenz. Während dieser Zeit kann kein anderer Thread auf denselben Speicherplatz zugreifen → **Serialisierung**. Bei einem heißen Bin (z. B. Buchstabe 'e') warten hunderte Threads hintereinander.

### Lösung: Privatisierung mit Shared Memory

Jeder Block arbeitet auf einer **privaten Kopie** des Histogramms im Shared Memory:

1. Initialisiere private Kopie mit 0.
2. Berechne Histogramm auf privater Kopie (Atomics auf Shared Memory: ~100× schneller als DRAM-Atomics).
3. Schreibe die private Kopie atomar in das globale Histogramm zurück.

```c
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
    __shared__ unsigned int s_histo[7];   // private Histogramm-Kopie

    // Phase 1: Shared Memory initialisieren
    if (threadIdx.x < 7)
        s_histo[threadIdx.x] = 0;
    __syncthreads();

    // Phase 2: Lokale Atomics auf Shared Memory
    int i      = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        int pos = buffer[i] - 'a';
        if (pos >= 0 && pos < 26)
            atomicAdd(&s_histo[pos / 4], 1);
        i += stride;
    }
    __syncthreads();

    // Phase 3: Merge in globales Histogramm
    if (threadIdx.x < 7)
        atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
}
```

**Speedup durch Privatisierung:** Typisch >10× gegenüber naiver globaler Atomic-Implementierung.

### Voraussetzungen für Privatisierung

* Die Operation muss **assoziativ und kommutativ** sein (Addition, Min, Max, And, Or – ja; Division – nein).
* Das private Histogramm muss **in den Shared Memory passen** (begrenzt auf ~48–192 kB).
* Bei zu großem Histogramm: partielle Privatisierung möglich (nur heiße Bins privatisieren).

---

## 5. Effizienter Datentransfer: Pinned Memory

### DMA und das Problem mit virtuellen Adressen

`cudaMemcpy` nutzt intern **DMA** (Direct Memory Access) – eine spezialisierte Hardware-Einheit, die Daten direkt zwischen physischen Speicheradressen überträgt, ohne die CPU zu belasten.

Problem: Der normale `malloc`-Speicher ist **virtuell** – das Betriebssystem kann Seiten jederzeit auslagern (Page Out). Der DMA benötigt aber garantiert im physischen Speicher vorhandene Adressen. Wenn der Host-Speicher nicht im RAM liegt, muss CUDA intern erst eine Kopie in einen sicheren Bereich anfertigen → **zusätzlicher Overhead**.

### Gepinnter Speicher (Pinned Memory / Page-Locked Memory)

Gepinnter Speicher ist **fest im physischen RAM verankert** – das Betriebssystem darf diese Seiten nicht auslagern. DMA-Transfers können direkt darauf zugreifen, ohne Zwischenkopie.

```c
// Allokation (statt malloc):
cudaHostAlloc((void **) &h_A, N * sizeof(float), cudaHostAllocDefault);

// Freigabe (statt free):
cudaFreeHost(h_A);
```

**Effekt:** `cudaMemcpy` ist mit Pinned Memory typisch **~2× schneller**, da keine interne Hilfskopie notwendig ist.

> **Achtung:** Gepinnter Speicher ist eine **begrenzte Ressource**. Übermäßige Nutzung kann das Betriebssystem destabilisieren, da weniger Speicher zum Auslagern zur Verfügung steht. Nur für Puffer verwenden, die tatsächlich für DMA-Transfers genutzt werden.

---

## 6. Asynchrones Arbeiten: CUDA-Streams

### Das serielle Ausgangsproblem

Ohne Optimierung läuft ein CUDA-Programm vollständig seriell:

```
Host→Device Transfer  →  Kernel  →  Device→Host Transfer
```

Während der Datentransfer läuft, ist die GPU-Recheneinheit idle. Während der Kernel läuft, ist die DMA-Engine idle. Das ist verschwendete Kapazität.

### `cudaMemcpyAsync`

```c
cudaMemcpyAsync(dst, src, size, kind, stream);
```

* Kehrt **sofort** zurück (nicht-blockierend für den Host).
* Der Transfer läuft im Hintergrund, asynchron.
* **Host-Speicher muss gepinned sein** – sonst muss CUDA intern synchron kopieren.
* Letzter Parameter: der Stream, in dem die Operation eingereiht wird.

### CUDA-Streams

Ein **Stream** ist eine geordnete FIFO-Warteschlange von GPU-Operationen (Kernel-Starts und `cudaMemcpyAsync`-Aufrufe). Operationen **innerhalb** eines Streams werden in Reihenfolge ausgeführt. Operationen in **verschiedenen Streams** können parallel ablaufen.

```c
// Streams anlegen
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// Kernel in Stream starten
vecAddKernel<<<Grid, Block, 0, stream0>>>(d_A0, d_B0, d_C0, n);

// Auf einzelnen Stream warten
cudaStreamSynchronize(stream0);

// Streams freigeben
cudaStreamDestroy(stream0);
```

### Pipelining: Rechnen und Kopieren überlappen

Die Idee: Teile die Eingabe in **Segmente** auf. Während Segment 0 berechnet wird, werden schon die Daten für Segment 1 übertragen.

**Naiver Ansatz (funktioniert schlecht auf älteren GPUs):**
```c
for (int i = 0; i < n; i += SegSize * 2) {
    // Stream 0: Transfer + Kernel + Rückkopie
    cudaMemcpyAsync(d_A0, h_A + i,         SegSize * sizeof(float), H2D, stream0);
    cudaMemcpyAsync(d_B0, h_B + i,         SegSize * sizeof(float), H2D, stream0);
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
    cudaMemcpyAsync(h_C + i, d_C0,         SegSize * sizeof(float), D2H, stream0);

    // Stream 1: Transfer + Kernel + Rückkopie (nächstes Segment)
    cudaMemcpyAsync(d_A1, h_A + i + SegSize, SegSize * sizeof(float), H2D, stream1);
    cudaMemcpyAsync(d_B1, h_B + i + SegSize, SegSize * sizeof(float), H2D, stream1);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
    cudaMemcpyAsync(h_C + i + SegSize, d_C1, SegSize * sizeof(float), D2H, stream1);
}
cudaStreamSynchronize(stream0);
cudaStreamSynchronize(stream1);
```

**Problem auf älteren GPUs:** Die PCIe-Verbindung hat getrennte Up/Down-Kanäle, aber ältere Geräte haben nur **eine Copy-Engine**. Alle Kopieroperationen werden dadurch sequentialisiert, auch wenn sie in verschiedenen Streams sind.

**Bessere Lösung – Operationen gleicher Art gebündelt:**
```c
for (int i = 0; i < n; i += SegSize * 2) {
    // Erst alle Transfers beider Streams (beide nutzen PCIe optimal)
    cudaMemcpyAsync(d_A0, h_A + i,           SegSize * sizeof(float), H2D, stream0);
    cudaMemcpyAsync(d_B0, h_B + i,           SegSize * sizeof(float), H2D, stream0);
    cudaMemcpyAsync(d_A1, h_A + i + SegSize, SegSize * sizeof(float), H2D, stream1);
    cudaMemcpyAsync(d_B1, h_B + i + SegSize, SegSize * sizeof(float), H2D, stream1);

    // Dann beide Kernel (überlappen mit noch laufenden Transfers des nächsten Segments)
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);

    // Dann Rückkopien
    cudaMemcpyAsync(h_C + i,           d_C0, SegSize * sizeof(float), D2H, stream0);
    cudaMemcpyAsync(h_C + i + SegSize, d_C1, SegSize * sizeof(float), D2H, stream1);
}
cudaStreamSynchronize(stream0);
cudaStreamSynchronize(stream1);
```

Durch das Bündeln gleichartiger Operationen können Transfers und Kernel-Ausführungen tatsächlich überlappen – auch auf Geräten mit nur einer Copy-Engine.

---

## 7. CUDA Events

### Präzise GPU-Zeitmessung

CPU-seitige Zeitmessung (`gettimeofday`) ist für GPU-Kernel ungeeignet, da Kernel asynchron laufen. **CUDA Events** sind Zeitstempel, die direkt in die GPU-Befehlswarteschlange eingereiht werden – die GPU schreibt den Zeitstempel, wenn sie den Event-Eintrag erreicht.

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);                          // Zeitstempel vor Kernel
vecAddKernel<<<Grid, Block>>>(d_A, d_B, d_C, n);
cudaEventRecord(stop);                           // Zeitstempel nach Kernel

cudaEventSynchronize(stop);                      // Host wartet, bis stop-Event erreicht ist

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel-Zeit: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

`cudaEventSynchronize` blockiert den Host, bis die GPU den Event abgearbeitet hat – ähnlich wie `cudaDeviceSynchronize`, aber nur bis zu diesem Event.

### Stream-übergreifende Synchronisation

Events können genutzt werden, um **einen Stream auf den Abschluss einer Operation in einem anderen Stream warten zu lassen** – ohne den Host zu blockieren:

```c
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 0: Kernel starten, danach Event setzen
kernelA<<<Grid, Block, 0, stream0>>>(...);
cudaEventRecord(event, stream0);         // Event in stream0 einreihen

// Stream 1: warten, bis stream0 den Event passiert hat – dann erst fortfahren
cudaStreamWaitEvent(stream1, event, 0); // blockiert stream1, nicht den Host!
kernelB<<<Grid, Block, 0, stream1>>>(...);

cudaEventDestroy(event);
```

Der Unterschied zu `cudaStreamSynchronize`: `cudaStreamWaitEvent` ist **asynchron** auf der Host-Seite – der Host-Thread läuft weiter. Nur die GPU-seitige Ausführung in `stream1` wird zurückgehalten.

**Typischer Einsatz:**
* Präzise Kernel-Laufzeitmessung (genauer als CPU-Timer).
* Abhängigkeiten zwischen Streams ausdrücken, ohne den Host einzubremsen.
* Profiling: mehrere Events setzen und Zeitabstände zwischen Phasen messen.

---

## 8. Zusammenfassung

| Konzept | Kernidee | CUDA-Schlüsselwort/-API |
|---|---|---|
| **Shared Memory** | Schneller On-Chip-Cache, geteilt im Block | `__shared__` |
| **`__syncthreads()`** | Barriere für alle Threads im Block | `__syncthreads()` |
| **Bank Conflicts** | Zugriffe auf dieselbe Bank → Serialisierung | Stride-1-Zugriffe bevorzugen |
| **Tiling** | Eingabedaten einmalig in Shared Memory laden | `I_TILE_WIDTH = O_TILE_WIDTH + Mask_Width - 1` |
| **Constant Memory** | Read-only, gecacht, Broadcast für gleiche Adresse | `__constant__`, `cudaMemcpyToSymbol` |
| **Atomare Operationen** | Read-Modify-Write ohne Race Condition | `atomicAdd`, `atomicMin`, … |
| **Privatisierung** | Atomics lokal auf Shared Memory, dann Merge | `__shared__` + `atomicAdd` |
| **Pinned Memory** | Nicht auslagerbarer Host-Speicher; ~2× schnellere Transfers | `cudaHostAlloc`, `cudaFreeHost` |
| **Streams** | Geordnete FIFO-Queues; verschiedene Streams parallel | `cudaStreamCreate`, `cudaStreamSynchronize` |
| **`cudaMemcpyAsync`** | Nicht-blockierender Transfer im Stream | `cudaMemcpyAsync(..., stream)` |
| **Pipelining** | Transfer und Kernel überlappen durch Segmentierung | Gleichartige Ops gebündelt pro Iteration |
| **CUDA Events** | GPU-seitige Zeitstempel; stream-übergreifende Sync | `cudaEventRecord`, `cudaEventElapsedTime`, `cudaStreamWaitEvent` |
