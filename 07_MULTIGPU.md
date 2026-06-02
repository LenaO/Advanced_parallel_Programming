# Kurseinheit 7: Multi-GPU Computing

## Lernziele

Nach dieser Einheit sollten Sie in der Lage sein:

- Den Unterschied zwischen klassischem MPI+CUDA und CUDA-Aware MPI zu erklären
- Den richtigen MPI-Threading-Level (`MPI_Init_thread`) für kombinierte MPI+CUDA-Anwendungen wählen
- Domain Decomposition für Multi-GPU-Probleme zu planen und umzusetzen
- GPUs korrekt auf MPI-Ranks zu verteilen (`MPI_Comm_split_type`, `cudaSetDevice`)
- Die Konzepte UVA, NVLink, GPUDirect P2P und GPUDirect RDMA zu beschreiben und deren Performance-Auswirkungen einzuschätzen
- P2P-Zugriff zwischen GPUs zu prüfen und zu aktivieren (`cudaDeviceCanAccessPeer`)
- Konvergenzprüfungen über alle Ranks mit `MPI_Allreduce` korrekt zu implementieren
- Rechnen und Kommunizieren mit CUDA-Streams zu überlappen
- NCCL für stream-bewusste GPU-Kommunikation einzusetzen
- Das PGAS-Modell von NVSHMEM und die Put/Get-API zu verstehen

---

## 1. Klassisches MPI mit CUDA

In klassischen CUDA+MPI-Programmen kennt MPI den GPU-Speicher nicht. Der Programmierer muss Daten manuell zwischen GPU und Host-Speicher kopieren, bevor MPI sie versenden kann.

**Ablauf beim Sender (Rank 0):**
1. Daten liegen auf der GPU (`s_buf_d`)
2. `cudaMemcpy` kopiert die Daten von GPU → Host (`s_buf_h`)
3. `MPI_Send` versendet die Host-Daten

**Ablauf beim Empfänger (Rank 1):**
1. `MPI_Recv` empfängt die Daten in den Host-Puffer (`r_buf_h`)
2. `cudaMemcpy` kopiert von Host → GPU (`r_buf_d`)

```c
/* Allokation von Host und GPU-Speicher, Initialisierung etc */
if (rank == 0) {
    cudaMemcpy(s_buf_h, s_buf_d, size, cudaMemcpyDeviceToHost);
    MPI_Send(s_buf_h, size, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
}
if (rank == 1) {
    MPI_Recv(r_buf_h, size, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &stat);
    cudaMemcpy(r_buf_d, r_buf_h, size, cudaMemcpyHostToDevice);
}
```

**Nachteil:** Jede Kommunikation erfordert zwei zusätzliche Speicherkopieroperationen (GPU→Host, Host→GPU). Das erzeugt Latenz und belastet den PCIe-Bus doppelt.

---

## 2. CUDA-Aware MPI

CUDA-Aware MPI bedeutet, dass MPI direkt mit Zeigern auf GPU-Speicher umgehen kann. Die expliziten `cudaMemcpy`-Aufrufe durch den Nutzer entfallen vollständig.

```c
/* Allokation von GPU-Speicher, Initialisierung etc */
if (rank == 0) {
    MPI_Send(s_buf_d, size, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
}
if (rank == 1) {
    MPI_Recv(r_buf_d, size, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &stat);
}
```

Der einzige Unterschied zur klassischen Variante: Die Zeiger `s_buf_d` und `r_buf_d` zeigen direkt auf GPU-Speicher — MPI erkennt dies automatisch über **Unified Virtual Addressing (UVA)** (siehe Abschnitt 6).

**Was passiert intern?** Ob und wie effizient die Übertragung durchgeführt wird, hängt davon ab, ob GPUDirect-Technologien verfügbar sind (siehe Abschnitt 7).

### 2.1 MPI Threading-Level

Sobald MPI und CUDA-Streams kombiniert werden — insbesondere wenn mehrere Streams gleichzeitig kommunizieren — muss der **MPI-Threading-Level** korrekt gesetzt werden. Statt `MPI_Init` wird `MPI_Init_thread` verwendet:

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
if (provided < MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "MPI_THREAD_MULTIPLE nicht unterstützt!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

Die vier Threading-Level in aufsteigender Stärke:

| Level | Bedeutung |
|---|---|
| `MPI_THREAD_SINGLE` | Nur ein Thread ruft MPI auf (Standard) |
| `MPI_THREAD_FUNNELED` | Mehrere Threads, aber nur der Haupt-Thread ruft MPI auf |
| `MPI_THREAD_SERIALIZED` | Mehrere Threads rufen MPI auf, aber nicht gleichzeitig |
| `MPI_THREAD_MULTIPLE` | Mehrere Threads rufen MPI gleichzeitig auf |

**Relevanz für Multi-GPU:**
- Bei Überlappung mit mehreren Streams und gleichzeitigen `MPI_Sendrecv`-Aufrufen aus verschiedenen Threads ist `MPI_THREAD_MULTIPLE` nötig
- Nicht alle MPI-Implementierungen unterstützen `MPI_THREAD_MULTIPLE` vollständig oder mit voller Performance — daher immer prüfen (`provided`)
- NCCL umgeht dieses Problem, da es streams-intern arbeitet und keine MPI-Aufrufe aus mehreren Threads benötigt

---

## 3. Laufendes Beispiel: Jacobi-Solver

Als durchgängiges Beispiel für Multi-GPU-Parallelisierung wird in dieser Einheit ein **Jacobi-Iterationsverfahren** auf einem 2D-Gitter verwendet. Der Jacobi-Solver berechnet iterativ neue Gitterwerte als Mittelwert der vier direkten Nachbarn:

**Seriell (CPU):**
```c
for (int iy = 1; iy < ny-1; iy++)
    for (int ix = 1; ix < nx-1; ix++)
        a_new[iy*nx+ix] = 0.25 * (
            a[iy*nx+(ix+1)] + a[iy*nx+ix-1] +
            a[(iy-1)*nx+ix] + a[(iy+1)*nx+ix]
        );
```

**Als CUDA-Kernel:**
```cuda
int iy = blockIdx.y * blockDim.y + threadIdx.y;
int ix = blockIdx.x * blockDim.x + threadIdx.x;
if (iy < ny-1 && ix < nx-1) {
    float new_val = 0.25f * (
        a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
        a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]
    );
    a_new[iy * nx + ix] = new_val;
}
```

**Iterationsschleife:**
```
Solange nicht konvergiert:
    1. Jacobi-Schritt (Kernel)
    2. Randbedingungen setzen
    3. a_new und a vertauschen
    4. Nächste Iteration
```

### 3.1 Konvergenzprüfung mit globalem Allreduce

Im Single-GPU-Fall wird die Konvergenz über die L2-Norm der Residuen geprüft. Im Multi-GPU-Fall berechnet jeder Rank nur seine lokale Teilnorm — diese muss über alle Ranks **aggregiert** werden:

```c
// Lokale L2-Norm auf der GPU berechnen (als Teil des Jacobi-Kernels)
// l2_norm_d enthält die lokale Summe der quadrierten Residuen

// GPU-Ergebnis auf Host kopieren
double l2_norm_h = 0.0;
cudaMemcpy(&l2_norm_h, l2_norm_d, sizeof(double), cudaMemcpyDeviceToHost);

// Globale L2-Norm über alle Ranks summieren
double global_l2_norm = 0.0;
MPI_Allreduce(&l2_norm_h, &global_l2_norm, 1, MPI_DOUBLE,
              MPI_SUM, MPI_COMM_WORLD);

global_l2_norm = sqrt(global_l2_norm);

// Konvergenzbedingung prüfen
if (global_l2_norm < tolerance) break;
```

Mit NCCL kann die Reduktion direkt auf der GPU ohne CPU-Zwischenspeicherung erfolgen:

```c
// L2-Norm direkt GPU-seitig reduzieren (bleibt auf GPU)
ncclAllReduce(l2_norm_d, l2_norm_global_d, 1,
              ncclDouble, ncclSum, nccl_comm, compute_stream);
```

> **Wichtig:** `MPI_Allreduce` ist eine **kollektive Operation** — alle Ranks müssen sie in jeder Iteration aufrufen. Dies ist ein Synchronisationspunkt: Kein Rank startet die nächste Iteration, bevor alle die aktuelle abgeschlossen haben. Das ist korrekt, da die Jacobi-Iteration sequentielle Abhängigkeiten zwischen Iterationen hat.

---

## 4. Domain Decomposition

Um den Jacobi-Solver auf mehrere GPUs zu verteilen, wird das Gitter aufgeteilt (**Domain Decomposition**). Jede GPU berechnet ihren Teilbereich, muss aber am Rand Daten von Nachbar-GPUs austauschen (Halo-Austausch).

### 4.1 Zeilenweise Aufteilung (Row Decomposition)

Das Gitter wird horizontal in Streifen aufgeteilt. Jede GPU erhält zusammenhängende Zeilen.

```
┌──────────────────┐
│   GPU 0 (Rank 0) │  ← oberer Rand → GPU 1
├──────────────────┤
│   GPU 1 (Rank 1) │  ← oberer Rand → GPU 0
│                  │  ← unterer Rand → GPU 2
├──────────────────┤
│   GPU 2 (Rank 2) │
└──────────────────┘
```

- **Vorteil:** Minimiert die Anzahl der Nachbarn (nur oben/unten); nur **zusammenhängende** Daten werden verschickt (bei row-major Layout)
- **Ideal für:** Latenz-dominierte Kommunikation

### 4.2 Spaltenweise Aufteilung (Column Decomposition)

Das Gitter wird vertikal aufgeteilt. Jede GPU erhält Spalten.

- **Vorteil:** Minimiert die Datenmenge, die pro Nachbar verschickt wird
- **Nachteil:** Bei row-major Layout **nicht zusammenhängend** — zusätzliches Pack/Unpack nötig
- **Ideal für:** Bandbreiten-begrenzte Probleme

### 4.3 Halo-Austausch: Obere und untere Nachbarn

Bei zeilenweiser Aufteilung sind die Randdaten zusammenhängend. Der Austausch erfolgt mit `MPI_Sendrecv`:

```c
// Sende letzte Zeile an unteren Nachbar, empfange obere Halo-Zeile
MPI_Sendrecv(a_new_d + offset_last_row,  m-2, MPI_DOUBLE, b_nb, 1,
             a_new_d + offset_top_boundary, m-2, MPI_DOUBLE, t_nb, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// Sende erste Zeile an oberen Nachbar, empfange untere Halo-Zeile
MPI_Sendrecv(a_new_d + offset_first_row, m-2, MPI_DOUBLE, t_nb, 0,
             a_new_d + offset_bottom_boundary, m-2, MPI_DOUBLE, b_nb, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

### 4.4 Halo-Austausch: Linke und rechte Nachbarn

Bei spaltenweiser Aufteilung oder bei 2D-Gittern mit vier Nachbarn sind die Spalten-Daten **nicht zusammenhängend**. Deshalb müssen sie zuerst mit einem eigenen Kernel in einen Buffer gepackt werden:

```c
// Pack-Kernel: kopiert Spalten-Daten in zusammenhängenden Puffer
pack<<<gs, bs, 0, s>>>(to_left_d, u_new_d, n, m);
cudaStreamSynchronize(s);

MPI_Sendrecv(to_left_d,   n-2, MPI_DOUBLE, l_nb, 0,
             from_right_d, n-2, MPI_DOUBLE, r_nb, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// Unpack-Kernel: schreibt empfangene Daten zurück ins Grid
unpack<<<gs, bs, 0, s>>>(u_new_d, from_right_d, n, m);
cudaStreamSynchronize(s);
```

> **Hinweis:** `cudaStreamSynchronize` stellt sicher, dass der Pack-Kernel vollständig abgeschlossen ist, bevor MPI auf die Daten zugreift.

---

## 5. GPU-Zuweisung zu MPI-Ranks

Bei Multi-GPU-Systemen mit mehreren Knoten müssen GPUs den MPI-Prozessen korrekt zugeordnet werden. Das Ziel: **Jeder Rank bekommt eine GPU**, und Ranks auf dem gleichen Knoten teilen sich die dort vorhandenen GPUs.

### Vorgehen: Local Rank bestimmen

```c
// Schritt 1: Kommunikator nur für Prozesse auf dem gleichen Knoten
MPI_Comm local_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &local_comm);

// Schritt 2: Lokalen Rank auf dem Knoten abfragen
int local_rank = -1;
MPI_Comm_rank(local_comm, &local_rank);
MPI_Comm_free(&local_comm);

// Schritt 3: GPU entsprechend dem lokalen Rank auswählen
int num_devs = 0;
cudaGetDeviceCount(&num_devs);
cudaSetDevice(local_rank % num_devs);
```

- `MPI_COMM_TYPE_SHARED` gruppiert alle Ranks, die sich einen gemeinsamen Speicher teilen (= gleicher Knoten)
- `local_rank % num_devs` verteilt die Ranks zyklisch auf die verfügbaren GPUs
- Beispiel: 4 Ranks, 2 GPUs → Rank 0 & 2 → GPU 0; Rank 1 & 3 → GPU 1

---

## 6. CUDA Unified Virtual Addressing (UVA)

Unified Virtual Addressing (UVA) ist die technologische Grundlage für CUDA-Aware MPI. UVA schafft einen **einzigen virtuellen Adressraum** für Host-Speicher und GPU-Speicher.

```
Adressraum: 0x0000 ──────────────────────────── 0xFFFF
                   │ System Memory  │ GPU Memory │
```

**Eigenschaften:**
- Ein Pointer kann auf Host-Speicher **oder** GPU-Speicher zeigen
- Die Laufzeitumgebung (z.B. MPI) kann anhand der Adresse bestimmen, wo die Daten liegen
- Keine manuelle Kennzeichnung des Speichertyps durch den Nutzer nötig

**Voraussetzungen:**
- Compute Capability 2.0 oder höher
- 64-Bit-Anwendung unter Linux oder Windows (+ TCC-Modus)

UVA ermöglicht erst, dass CUDA-Aware MPI funktioniert: MPI prüft den übergebenen Zeiger, erkennt ob er auf GPU-Speicher zeigt, und leitet die Übertragung entsprechend weiter.

---

## 7. GPUDirect-Technologien

GPUDirect ist eine Familie von NVIDIA-Technologien, die direkte Datenübertragungen zwischen GPUs und anderen PCIe-Geräten (z.B. Netzwerkkarten) ermöglichen — ohne Umweg über den Host-Speicher.

### 7.0 NVLink

**NVLink** ist NVIDIAs proprietärer Hochgeschwindigkeits-GPU-Interconnect, der in Server-GPUs (A100, H100) als Alternative zu PCIe eingesetzt wird.

| Eigenschaft | PCIe 4.0 (x16) | NVLink 3.0 (A100) |
|---|---|---|
| Bandbreite (bidirektional) | ~32 GB/s | **~600 GB/s** |
| Latenz | höher | deutlich niedriger |
| Reichweite | Knoten-intern | Knoten-intern (bis 6 Links) |

**Bedeutung für Multi-GPU:**
- NCCL nutzt NVLink automatisch als bevorzugten intra-node-Pfad, wenn verfügbar
- Dies erklärt die sehr hohen Bandbreiten in den P2P-Benchmarks (~87 GB/s auf JUWELS Booster mit A100-GPUs und NVLink)
- NVLink ist der Grund, warum CUDA-Aware MPI mit P2P auf dem gleichen Knoten deutlich schneller ist als über das Netzwerk

### 7.1 GPUDirect P2P (Peer-to-Peer)

Ermöglicht direkte Speicherzugriffe zwischen GPUs **im gleichen Knoten** über PCIe-Switch oder NVLink, ohne den CPU-Hauptspeicher zu durchlaufen.

```
GPU A ──NVLink / PCIe Switch── GPU B
              (direkt)
```

**P2P-Zugriff aktivieren:**

Bevor P2P-Transfers genutzt werden können, muss geprüft und aktiviert werden, ob zwei GPUs P2P-Zugriff unterstützen:

```c
int can_access = 0;
cudaDeviceCanAccessPeer(&can_access, src_device, dst_device);

if (can_access) {
    cudaSetDevice(src_device);
    cudaDeviceEnablePeerAccess(dst_device, 0);  // 0 = flags (reserviert)

    cudaSetDevice(dst_device);
    cudaDeviceEnablePeerAccess(src_device, 0);
} else {
    // Fallback: über Host-Speicher kopieren
}
```

Nach der Aktivierung können GPU-Speicher direkt kopiert werden:

```c
// Direkte GPU-zu-GPU-Kopie (kein Host-Puffer nötig)
cudaMemcpyPeerAsync(dst_ptr, dst_device,
                    src_ptr, src_device,
                    size, stream);
```

> **Hinweis:** CUDA-Aware MPI und NCCL erledigen das Aktivieren von P2P intern, wenn UVA und die Hardware es erlauben. Manuelles `cudaDeviceEnablePeerAccess` ist nur nötig, wenn P2P-Transfers direkt im eigenen Code verwendet werden.

### 7.2 GPUDirect RDMA (Remote Direct Memory Access)

Ermöglicht direkte Datenübertragungen zwischen GPU-Speicher und **Netzwerkkarte (HCA)** über den PCIe-Switch — knotenübergreifend, ohne Beteiligung des Host-Speichers.

```
GPU ──PCIe Switch── HCA ──Netzwerk── HCA ──PCIe Switch── GPU
                  (direkt, kein Host-Durchlauf)
```

### 7.3 CUDA-Aware MPI mit vs. ohne GPUDirect RDMA

Obwohl der API-Aufruf **identisch** ist, unterscheidet sich der interne Datenpfad erheblich:

| Variante | Datenpfad |
|---|---|
| **CUDA-Aware MPI + GPUDirect RDMA** | GPU → HCA → Netz → HCA → GPU (kein Host) |
| **CUDA-Aware MPI ohne GPUDirect RDMA** | GPU → Host-Speicher → HCA → Netz → HCA → Host-Speicher → GPU |
| **Klassisches MPI (Staging)** | GPU → Host (manuell) → HCA → Netz → HCA → Host (manuell) → GPU |

```c
// API-Aufruf ist in allen drei Fällen gleich:
MPI_Send(s_buf_d, size, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
MPI_Recv(r_buf_d, size, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &stat);
```

### 7.4 Benchmarkergebnisse (JUWELS Booster)

Messungen mit OpenMPI 4.1.0RC1 + UCX 1.9.0 auf dem JUWELS-Booster-Supercomputer:

**GPUDirect RDMA (knotenübergreifend):**

| Variante | Latenz (1 Byte) | Spitzenbandbreite |
|---|---|---|
| CUDA-Aware MPI (mit GDR) | **4,27 µs** | ~24.000 MiB/s |
| CUDA-Aware MPI (ohne GDR) | 24,56 µs | ~16.000 MiB/s |
| Klassisches MPI (Staging) | 25,64 µs | ~10.000 MiB/s |

**GPUDirect P2P (innerhalb eines Knotens):**

| Variante | Latenz (1 Byte) | Spitzenbandbreite |
|---|---|---|
| CUDA-Aware MPI (mit P2P) | **2,45 µs** | ~87.000 MiB/s |
| CUDA-Aware MPI (ohne P2P) | 22,01 µs | ~20.000 MiB/s |
| Klassisches MPI (Staging) | 23,50 µs | ~10.000 MiB/s |

**Erkenntnisse:**
- GPUDirect RDMA reduziert die Latenz um Faktor **~6×** gegenüber Staging
- GPUDirect P2P erreicht durch NVLink/PCIe-Direktverbindung nochmals deutlich höhere Bandbreiten (~87 GB/s)
- Ohne GPUDirect (aber CUDA-Aware) ist der Performancevorteil gegenüber manuellem Staging minimal — der Host-Umweg bleibt erhalten

---

## 8. Überlappung von Rechnen und Kommunizieren

### 8.1 Motivation

Das naive Muster — erst vollständig berechnen, dann kommunizieren — ist ineffizient:

```
[──── GPU Jacobi ────][─MPI─][──── GPU Jacobi ────][─MPI─]
```

Da für die Kommunikation nur die **Randzeilen** benötigt werden, können Rand- und Innenbereich in separaten Kerneln auf verschiedenen Streams berechnet werden:

```
[Rand][──── Jacobi Innen ────][Rand][──── Jacobi Innen ────]
      [─── MPI ───]                 [─── MPI ───]
```

Während die Randzeilen gesendet/empfangen werden, rechnet die GPU bereits den Innenbereich.

### 8.2 Implementierung mit MPI

```c
// 1. Rand-Kernel auf push_top_stream (oberste Zeile)
launch_jacobi_kernel(a_new, a, l2_norm_d,
    iy_start, (iy_start + 1), nx, push_top_stream);

// 2. Rand-Kernel auf push_bottom_stream (unterste Zeile)
launch_jacobi_kernel(a_new, a, l2_norm_d,
    (iy_end - 1), iy_end, nx, push_bottom_stream);

// 3. Innen-Kernel auf compute_stream (parallel zu MPI)
launch_jacobi_kernel(a_new, a, l2_norm_d,
    (iy_start + 1), (iy_end - 1), nx, compute_stream);

// 4. Warte auf oberen Rand, dann Austausch mit oberem Nachbar
const int top    = rank > 0 ? rank - 1 : (size - 1);
const int bottom = (rank + 1) % size;

cudaStreamSynchronize(push_top_stream);
MPI_Sendrecv(a_new + iy_start * nx,  nx, MPI_REAL_TYPE, top, 0,
             a_new + (iy_end * nx),  nx, MPI_REAL_TYPE, bottom, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 5. Warte auf unteren Rand, dann Austausch mit unterem Nachbar
cudaStreamSynchronize(push_bottom_stream);
MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0,
             a_new,                      nx, MPI_REAL_TYPE, top, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

**Wichtig:** `cudaStreamSynchronize` vor dem MPI-Aufruf stellt sicher, dass der Rand-Kernel fertig ist, bevor MPI auf die Daten zugreift.

---

## 9. NCCL — NVIDIA Collective Communication Library

### 9.1 Motivation: Problem mit MPI und Streams

CUDA-Aware MPI kann zwar GPU-Zeiger entgegennehmen, kennt aber **keine CUDA-Streams**. Deshalb muss vor jedem MPI-Aufruf mit GPU-Daten explizit synchronisiert werden:

```c
cudaStreamSynchronize(push_top_stream);  // blockiert CPU
MPI_Sendrecv(...);                        // blockiert CPU weiter
```

Diese häufige CPU-GPU-Synchronisation (`cudaDeviceSynchronize`, `cudaStreamSynchronize`) kostet Zeit und verhindert echte Überlappung.

**Alternative:** Stream-bewusste Kommunikationsbibliotheken — **NCCL** und **NVSHMEM**.

### 9.2 Was ist NCCL?

**NCCL** (NVIDIA Collective Communication Library) ist eine Bibliothek für effiziente Kommunikation zwischen GPUs über verschiedene Verbindungen:

| Verbindungstyp | Beispiel |
|---|---|
| Intra-Knoten | NVLink, PCIe, Shared Memory |
| Inter-Knoten | InfiniBand, Sockets, andere Netzwerke |

**Eigenschaften:**
- Bibliothek, die **auf der GPU läuft**: Kommunikationsaufrufe werden in CUDA-Kernel umgewandelt, die auf einem Stream laufen
- Seit Version **2.8**: Unterstützung für `ncclSend`/`ncclRecv` zwischen beliebigen Ranks (nicht nur kollektive Operationen)
- Ursprünglich für kollektive Operationen in Deep-Learning-Anwendungen entwickelt

### 9.3 Initialisierung (innerhalb von MPI)

NCCL benötigt einen eigenen Kommunikator (`ncclComm_t`) und eine eindeutige Kennung (`ncclUniqueId`):

```c
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Unique ID auf Rank 0 erzeugen und an alle verteilen
ncclUniqueId nccl_uid;
if (rank == 0) ncclGetUniqueId(&nccl_uid);
MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

// NCCL-Kommunikator initialisieren
ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, size, nccl_uid, rank);

/* ... Programmcode ... */

ncclCommDestroy(nccl_comm);
MPI_Finalize();
```

### 9.4 Kommunikationsbefehle

**Point-to-Point (ab NCCL 2.8):**
```c
ncclSend(void* sbuff, size_t count, ncclDataType_t type,
         int peer, ncclComm_t comm, cudaStream_t stream);

ncclRecv(void* rbuff, size_t count, ncclDataType_t type,
         int peer, ncclComm_t comm, cudaStream_t stream);
```

**Kollektive Operationen:**
```c
ncclAllReduce(sbuff, rbuff, count, type, op, comm, stream);
ncclBroadcast(sbuff, rbuff, count, type, root, comm, stream);
ncclReduce(sbuff, rbuff, count, type, op, root, comm, stream);
ncclReduceScatter(sbuff, rbuff, count, type, op, comm, stream);
ncclAllGather(sbuff, rbuff, count, type, comm, stream);
```

Entscheidend: Jeder Aufruf nimmt einen `cudaStream_t` entgegen — die Kommunikation läuft **auf dem Stream** und ist dadurch mit GPU-Kerneln überlappbar.

### 9.5 Gruppierte Kommunikationsaufrufe (`ncclGroupStart/End`)

Mehrere aufeinanderfolgende `ncclSend`/`ncclRecv`-Aufrufe sollten immer mit `ncclGroupStart()` und `ncclGroupEnd()` zusammengefasst werden:

**Gründe:**
1. **Deadlock-Vermeidung:** Ohne Gruppierung könnte ein `ncclSend` blockieren, wenn der Empfänger noch nicht in `ncclRecv` ist
2. **Performance:** Gruppierte Aufrufe können intern effizienter als einzelne Aufrufe ausgeführt werden

**Beispiele:**

```c
// SendRecv zwischen zwei Ranks
ncclGroupStart();
ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
ncclGroupEnd();

// Broadcast (manuell mit Send/Recv implementiert)
ncclGroupStart();
if (rank == root) {
    for (int r = 0; r < nranks; r++)
        ncclSend(sendbuff[r], size, type, r, comm, stream);
}
ncclRecv(recvbuff, size, type, root, comm, stream);
ncclGroupEnd();

// Nachbar-Austausch in mehrdimensionalem Gitter
ncclGroupStart();
for (int d = 0; d < ndims; d++) {
    ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
    ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
}
ncclGroupEnd();
```

### 9.6 Jacobi mit NCCL

```c
// Gesamten Jacobi-Schritt auf compute_stream berechnen
launch_jacobi_kernel(a_new, a, l2_norm_d,
    iy_start, iy_end, nx, compute_stream);

// Halo-Austausch auf demselben Stream (kein Synchronisationspunkt nötig!)
ncclGroupStart();
ncclRecv(a_new,                    nx, NCCL_REAL_TYPE, top, nccl_comm, compute_stream);
ncclSend(a_new + (iy_end-1)*nx,   nx, NCCL_REAL_TYPE, btm, nccl_comm, compute_stream);
ncclRecv(a_new + (iy_end*nx),     nx, NCCL_REAL_TYPE, btm, nccl_comm, compute_stream);
ncclSend(a_new + iy_start * nx,   nx, NCCL_REAL_TYPE, top, nccl_comm, compute_stream);
ncclGroupEnd();
```

**Vorteil gegenüber MPI:** Da NCCL den Stream kennt, wird der Halo-Austausch automatisch **nach** dem Jacobi-Kernel auf demselben Stream ausgeführt — ohne explizites `cudaStreamSynchronize`.

### 9.7 Überlappung mit Priority-Streams

Um Rechnen und Kommunizieren wirklich zu überlappen, werden **Priority-Streams** benötigt. Diese geben der GPU einen Hinweis, welche Arbeit bevorzugt ausgeführt werden soll:

```c
// Prioritätsbereich abfragen
int leastPriority   = 0;
int greatestPriority = leastPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// Streams mit unterschiedlichen Prioritäten erstellen
cudaStream_t compute_stream;
cudaStream_t push_stream;
cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, leastPriority);
cudaStreamCreateWithPriority(&push_stream,    cudaStreamDefault, greatestPriority);
```

### 9.8 Jacobi mit NCCL und Überlappung

```c
// Rand-Kernel auf push_stream (höchste Priorität)
launch_jacobi_kernel(a_new, a, l2_norm_d,
    iy_start, iy_start + 1,         nx, push_stream);
launch_jacobi_kernel(a_new, a, l2_norm_d,
    iy_end - 1,  iy_end,            nx, push_stream);

// Innen-Kernel auf compute_stream (läuft parallel zur Kommunikation)
launch_jacobi_kernel(a_new, a, l2_norm_d,
    iy_start + 1, iy_end - 1,       nx, compute_stream);

// Kommunikation auf push_stream (nach den Rand-Kerneln, parallel zum Innen-Kernel)
ncclGroupStart();
ncclRecv(a_new,                    nx, NCCL_REAL_TYPE, top, nccl_comm, push_stream);
ncclSend(a_new + (iy_end-1)*nx,   nx, NCCL_REAL_TYPE, btm, nccl_comm, push_stream);
ncclRecv(a_new + (iy_end*nx),     nx, NCCL_REAL_TYPE, btm, nccl_comm, push_stream);
ncclSend(a_new + iy_start * nx,   nx, NCCL_REAL_TYPE, top, nccl_comm, push_stream);
ncclGroupEnd();
```

**Ablauf:**
1. `push_stream` (hohe Priorität): Rand-Kernel → NCCL-Kommunikation
2. `compute_stream` (niedrige Priorität): Innen-Kernel läuft parallel zur Kommunikation

### 9.9 Kompilierung einer MPI+NCCL-Anwendung

```c
#include <nccl.h>
```

```makefile
MPICXX_FLAGS = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LD_FLAGS     = -L$(CUDA_HOME)/lib64 -lcudart -lnccl

# CUDA-Kernels kompilieren
$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c -o jacobi_kernels.o

# MPI-Code kompilieren und linken
$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi
```

---

## 10. NVSHMEM

### 10.1 Konzept und Programmiermodell

NVSHMEM implementiert die **OpenSHMEM**-API für Cluster von NVIDIA-GPUs. Es basiert auf dem **Partitioned Global Address Space (PGAS)**-Modell, das aus der MPI-RMA-Einheit bekannt ist — hier aber komplett GPU-zentrisch.

**Kernprinzipien:**
- **Einseitige Kommunikation** mit Put/Get (wie MPI-RMA, aber auf GPUs)
- **Symmetric Heap:** Ein verteilter Speicherbereich, der auf allen GPUs (Processing Elements, PEs) gleichzeitig allokiert wird
- **GPU-zentrische Bibliothek:** Kommunikationsaufrufe können direkt aus CUDA-Kerneln heraus erfolgen

### 10.2 Symmetrisches Speichermodell

```
┌────────────────────────────────────────────────────────────────┐
│               Partitioned Global Address Space                  │
│  ┌─────────┐       ┌─────────┐             ┌─────────┐        │
│  │GPU 0    │       │GPU 1    │             │GPU N    │        │
│  │(PE 0)   │       │(PE 1)   │             │(PE N)   │        │
│  │┌───────┐│       │┌───────┐│             │┌───────┐│        │
│  ││Symm.  ││       ││Symm.  ││  ←────────→ ││Symm.  ││        │
│  ││Heap   ││       ││Heap   ││             ││Heap   ││        │
│  │└───────┘│       │└───────┘│             │└───────┘│        │
│  │┌───────┐│       │┌───────┐│             │┌───────┐│        │
│  ││Private││       ││Private││             ││Private││        │
│  │└───────┘│       │└───────┘│             │└───────┘│        │
│  └─────────┘       └─────────┘             └─────────┘        │
└────────────────────────────────────────────────────────────────┘
```

**Speicherallokation:**
```c
// Symmetrische Objekte: kollektiv auf allen PEs allokiert
// (muss auf ALLEN PEs mit gleicher Größe aufgerufen werden!)
void* shared_data = nvshmem_malloc(shared_size);

// Privater Speicher: wie gewohnt
void* private_data;
cudaMalloc(&private_data, private_size);
```

### 10.3 GPU-zentrische Kommunikation

NVSHMEM ist in drei Modi verwendbar:

| Modus | Initiiert von | Beispiel |
|---|---|---|
| **GPU-initiiert** | Thread / Warp / Block im Kernel | `nvshmem_put(...)` im Kernel |
| **Stream/Graph-basiert** | CPU, läuft auf Stream | Kommunikations-Kernel auf Stream |
| **CPU-initiiert** | CPU, klassisch | `nvshmemx_putmem_on_stream(...)` |

**Interoperabilität:** NVSHMEM ist mit OpenSHMEM und MPI interoperabel (mit einigen API-Erweiterungen).

### 10.4 NVSHMEM Put/Get API

Die grundlegenden einseitigen Kommunikationsprimitive in NVSHMEM sind `nvshmem_put` und `nvshmem_get`. Sie können direkt aus einem CUDA-Kernel heraus aufgerufen werden:

```cuda
// Daten von diesem PE zu einem anderen PE schicken (Put)
// nvshmem_TYPE_put(dest_ptr, src_ptr, nelems, pe)
__global__ void kernel_put(double* data, int my_pe, int neighbor_pe, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Schreibe data[idx] direkt in den symmetrischen Heap von neighbor_pe
        nvshmem_double_p(&data[idx], data[idx], neighbor_pe);
    }
    // Sicherstellen, dass alle Puts abgeschlossen sind
    nvshmem_quiet();
}

// Daten von einem anderen PE lesen (Get)
__global__ void kernel_get(double* local, double* remote_sym, int src_pe, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Lese Wert aus dem symmetrischen Heap von src_pe
        local[idx] = nvshmem_double_g(&remote_sym[idx], src_pe);
    }
}
```

**Jacobi-Halo-Austausch mit NVSHMEM (konzeptuell):**

```c
// Allokation im symmetrischen Heap (kollektiv auf allen PEs)
double* a = (double*) nvshmem_malloc(nx * ny * sizeof(double));

// Im Kernel: Rand direkt in den Heap des Nachbarn schreiben
__global__ void jacobi_with_halo(double* a_new, double* a,
                                  int iy_start, int iy_end, int nx,
                                  int top_pe, int btm_pe) {
    // ... Jacobi-Berechnung ...

    // Halo-Austausch GPU-seitig (nur ein Thread pro Reihe nötig)
    if (threadIdx.x == 0 && iy == iy_start) {
        // Sende oberste Zeile an top_pe
        nvshmem_double_put(a_new,                       // Ziel: Anfang von a_new auf top_pe
                           a_new + iy_start * nx,       // Quelle: lokale erste Zeile
                           nx, top_pe);
    }
    nvshmem_quiet();  // warte bis alle Puts fertig
}
```

**Wichtige NVSHMEM-Synchronisationsprimitive:**

| Funktion | Bedeutung |
|---|---|
| `nvshmem_quiet()` | Warte bis alle ausstehenden Puts/Gets des lokalen PE abgeschlossen sind |
| `nvshmem_fence()` | Garantiert Reihenfolge von Put-Operationen |
| `nvshmem_barrier_all()` | Globale Barriere über alle PEs |
| `nvshmemx_barrier_all_on_stream(stream)` | Wie `barrier_all`, aber stream-basiert |

---

## 11. Vergleich der Ansätze

| Eigenschaft | Klass. MPI | CUDA-Aware MPI | CUDA-Aware MPI + GDR | NCCL | NVSHMEM |
|---|---|---|---|---|---|
| **GPU-Zeiger in Komm.-Aufruf** | Nein | Ja | Ja | Ja | Ja |
| **Host-Staging nötig** | Ja (manuell) | Intern (ohne GDR) | Nein | Nein | Nein |
| **Stream-Aware** | Nein | Nein | Nein | **Ja** | **Ja** |
| **Latenz (1 Byte, P2P)** | ~23 µs | ~22 µs | **~2,5 µs** | < 2,5 µs | < 2,5 µs |
| **Kollektive Ops** | Ja | Ja | Ja | Ja (optimiert) | Ja |
| **Aufruf aus Kernel** | Nein | Nein | Nein | Nein | **Ja** |
| **Typischer Einsatz** | Legacy-Code | Einfache Portierung | HPC-Anwendungen | Deep Learning, HPC | GPU-zentrisches HPC |

---

## Zusammenfassung

| Konzept | Kernaussage |
|---|---|
| **Klassisches MPI + CUDA** | Manuelles Staging GPU↔Host vor/nach MPI-Aufrufen; einfach aber ineffizient |
| **CUDA-Aware MPI** | GPU-Zeiger direkt in MPI-Aufrufen; UVA erkennt Speichertyp automatisch |
| **MPI Threading-Level** | `MPI_Init_thread(MPI_THREAD_MULTIPLE)` nötig bei gleichzeitigen MPI-Aufrufen aus mehreren Streams/Threads |
| **UVA** | Ein virtueller Adressraum für CPU und GPU; Grundlage für CUDA-Aware MPI |
| **NVLink** | Dedizierter GPU-Interconnect (~600 GB/s); von NCCL bevorzugt gegenüber PCIe (~32 GB/s) |
| **GPUDirect P2P** | Direkte GPU↔GPU-Übertragung innerhalb eines Knotens; `cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess` |
| **GPUDirect RDMA** | Direkte GPU↔Netzwerkkarte-Übertragung; umgeht Host-Speicher knotenübergreifend; Latenz ~4 µs statt ~25 µs |
| **Domain Decomposition** | Gitteraufteilung auf GPUs; Zeilenaufteilung für latenzdomin. Comm., Spalten für bandbreitendominiert |
| **GPU-Zuweisung** | `MPI_Comm_split_type` + `cudaSetDevice(local_rank % num_devs)` |
| **Konvergenzprüfung** | Lokale L2-Norm per `MPI_Allreduce` / `ncclAllReduce` über alle Ranks aggregieren |
| **Compute/Comm-Überlappung** | Rand- und Innenbereich auf separaten Streams; MPI nach `cudaStreamSynchronize` |
| **NCCL** | Stream-aware Kommunikationsbibliothek; Komm.-Aufrufe werden zu GPU-Kerneln; kein explizites Sync nötig |
| **ncclGroupStart/End** | Fasst Send/Recv zusammen; vermeidet Deadlocks, verbessert Performance |
| **Priority-Streams** | `cudaStreamCreateWithPriority`; Rand/Komm. auf high-priority Stream für echte Überlappung |
| **NVSHMEM** | PGAS-Modell für GPU-Cluster; GPU-initiierte Put/Get aus Kerneln heraus; symmetrischer Heap via `nvshmem_malloc` |
