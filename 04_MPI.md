
# Kurseinheit 4: Advanced MPI: Einseitige Kommunikation und mehr

---

## Lernziele

Nach diesem Kapitel sollten Sie in der Lage sein:
* nicht-blockierende kollektive Operationen anzuwenden und Rechnen mit Kommunikation zu überlappen.
* persistente Kommunikation zu erklären, aufzusetzen und korrekt mit Double-Buffering zu kombinieren.
* den Ablauf einer RMA-Operation (Window anlegen → Put/Get/Accumulate → Synchronisieren) zu beschreiben.
* die drei RMA-Synchronisationsmechanismen (Fence, PSCW, Lock/Unlock) zu unterscheiden und situationsgerecht auszuwählen.
* das MPI-Speichermodell (*separate* vs. *unified*) zu erklären und dessen Konsequenzen für die Korrektheit von RMA-Code zu benennen.
* typische Undefined-Behavior-Fallen bei RMA zu erkennen und zu vermeiden.
* MPI Shared Memory Windows anzulegen und über direkte Pointer-Zugriffe zu nutzen.

---

### 1. Rückblick: Klassisches MPI

MPI basiert ursprünglich auf **zweiseitiger Kommunikation**: Jeder Datentransfer erfordert die aktive Beteiligung beider Seiten – Sender und Empfänger müssen passende `Send`/`Recv`-Aufrufe ausführen.

**Blockierend:**
```c
// Prozess 0 sendet, Prozess 1 empfängt
MPI_Send(data, 1024, MPI_INT, 1, 0xA, MPI_COMM_WORLD);
MPI_Recv(data, 1024, MPI_INT, 0, 0xA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

**Nicht-blockierend** (mit `MPI_Request`):
```c
MPI_Request requests[2];
MPI_Isend(data, 1024, MPI_INT, n1, 0xA, MPI_COMM_WORLD, &requests[0]);
MPI_Irecv(in,   1024, MPI_INT, n2, 0xA, MPI_COMM_WORLD, &requests[1]);
MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
```

---

### 2. Nicht-blockierende Kollektive Operationen

Kollektive Operationen (z. B. `MPI_Reduce`, `MPI_Alltoall`) blockieren standardmäßig, bis alle Prozesse fertig sind. Seit MPI 3.0 gibt es nicht-blockierende Varianten:

* Das Namensschema ist analog zu `Isend`/`Irecv`: Vorangestelltes **`I`** + `MPI_Request`-Parameter.
* Beispiel: `MPI_Ireduce(..., MPI_COMM_WORLD, &request)`
* Abschluss wie gewohnt mit `MPI_Wait` / `MPI_Waitall`.

**Wichtige Regeln:**
* Alle Prozesse eines Kommunikators müssen die **nicht-blockierende** Variante verwenden – blockierende und nicht-blockierende Kollektive dürfen **nicht gemischt** werden.
* Die **Reihenfolge** der kollektiven Aufrufe muss auf allen Prozessen identisch sein.

**Warum?**
* Vermeidung von Deadlocks.
* **Überlappung von Rechnen und Kommunizieren** – wenn die Hardware es unterstützt, kann die Netzwerkkarte Daten senden, während die CPU weiterrechnet.

#### Praxisbeispiel: 2-D FFT mit überlappender Kommunikation

Eine 2-D FFT benötigt nach den lokalen 1-D FFTs einen globalen Datentausch (`MPI_Alltoall`). Mit `MPI_Ialltoall` kann man den Kommunikationsaufwand mit dem nächsten Berechnungsblock überlappen:

```c
MPI_Request req[nb];
for (b = 0; b < nb; ++b) {
    // 1. Lokale 1-D FFTs für Block b berechnen
    for (x = b*n/p/nb; x < (b+1)*n/p/nb; ++x)
        fft_1d(/* x-th stencil */);

    // 2. Frühere Kommunikationen testen (nicht blockierend)
    for (i = max(0, b - nt); i < b; ++i)
        MPI_Test(&req[i], &flag, MPI_STATUS_IGNORE);

    // 3. Kommunikation für Block b starten (nicht blockierend)
    MPI_Ialltoall(&in, n/p*n/p/bs, cplx_t,
                  &out, n/p*n/p,   cplx_t, comm, &req[b]);
}
MPI_Waitall(nb, req, MPI_STATUSES_IGNORE);

// 4. Abschließende 1-D FFTs in y-Richtung
for (y = 0; y < n/p; ++y)
    fft_1d(/* y-th stencil */);
```

---

### 3. Persistente Kommunikation

In vielen Programmen werden **dieselben Kommunikationsaufrufe** in einer Schleife wiederholt (z. B. Halo-Austausch in Gittersimulationen). Jeder `MPI_Send`/`MPI_Recv`-Aufruf hat einen internen **Setup-Overhead** (Metadaten, Puffer-Registrierung). Persistente Kommunikation vermeidet diesen Overhead, indem Setup und Ausführung getrennt werden.

#### Drei Phasen

**Phase 1 – Request anlegen** (einmalig):
```c
int MPI_Send_init(void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request *request)

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Request *request)
```

**Phase 2 – Kommunikation starten** (pro Iteration):
```c
int MPI_Start(MPI_Request *request)
int MPI_Startall(int count, MPI_Request *array_of_requests)
```

**Phase 3 – Warten** (pro Iteration):
```c
MPI_Wait / MPI_Waitall
```

#### Beispiel: Halo-Austausch
```c
MPI_Request req[2];
MPI_Send_init(&data,          16, MPI_INT, right, 0xB, MPI_COMM_WORLD, &req[0]);
MPI_Recv_init(&data[offset],  16, MPI_INT, left,  0xB, MPI_COMM_WORLD, &req[1]);

for (int i = 0; i < iter; i++) {
    compute(data, i);
    MPI_Startall(2, req);
    // ... anderes Rechnen möglich ...
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
}
```

#### Achtung: Werte sind beim Setup eingefroren
Der Request speichert den **Zeiger** auf den Buffer, nicht die Werte. Wenn der Buffer zwischen Iterationen wechselt (z. B. Double-Buffering), braucht man **zwei separate Requests** und tauscht sie zusammen mit den Buffer-Pointern:

```c
MPI_Request req, req_b;
MPI_Send_init(data_a, 16, MPI_INT, right, 0xB, MPI_COMM_WORLD, &req);
MPI_Send_init(data_b, 16, MPI_INT, right, 0xB, MPI_COMM_WORLD, &req_b);

for (int i = 0; i < iter; i++) {
    compute(data_a, data_b, n);
    MPI_Start(&req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    // Buffer und Request gemeinsam tauschen
    int *tmp   = data_a; data_a = data_b; data_b = tmp;
    MPI_Request tmp_r = req; req = req_b; req_b = tmp_r;
}
```

---

### 4. Einseitige Kommunikation (RMA – Remote Memory Access)

#### 4.1 Motivation

Bei klassischer zweiseitiger Kommunikation muss der Ziel-Prozess **aktiv** ein `Recv` aufrufen. Bei **einseitiger Kommunikation** kann ein Prozess direkt in den Speicher eines anderen lesen oder schreiben, ohne dass dieser aktiv mitmachen muss:

| Modell | Beteiligte | Beispiel |
|---|---|---|
| Zweiseitig (Send/Recv) | Sender + Empfänger | `MPI_Send` / `MPI_Recv` |
| Einseitig (RMA) | Nur der Initiator | `MPI_Put`, `MPI_Get` |

MPI RMA wurde in MPI 2.0 eingeführt und in MPI 3.0/4.0 erweitert. Der wichtigste eigenständige Vertreter einseitiger Kommunikation außerhalb von MPI ist **OpenSHMEM**.

#### 4.2 Ablauf einer RMA-Operation

Eine RMA-Kommunikation besteht aus drei Schritten:

1. **Memory Window anlegen:** Definiere, welcher Speicherbereich für RMA-Zugriffe freigegeben wird. Dies ist eine kollektive Operation, die von allen Prozessen eines Kommunikators ausgeführt wird!
2. **Daten verschieben:** `MPI_Put`, `MPI_Get` oder `MPI_Accumulate` ausführen (asynchron, das bedeutet, die Funktionen kehren direkt zurück).
3. **Synchronisieren:** Sicherstellen, dass die Operationen abgeschlossen sind und beide Seiten konsistente Daten sehen.

---

### 5. RMA-Fenster (Memory Windows)

#### Anlegen mit `MPI_Win_create`

```c
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
                   MPI_Info info, MPI_Comm comm, MPI_Win *win)
```

| Parameter | Bedeutung |
|---|---|
| `base` | Zeiger auf den lokalen Speicherbereich |
| `size` | Größe des Fensters in Byte |
| `disp_unit` | Abstand zwischen Elementen (z. B. `sizeof(double)`) |
| `info` | Performance-Hinweise; `MPI_INFO_NULL` ist immer korrekt |
| `comm` | Kommunikator – **alle** Prozesse müssen diesen Aufruf ausführen (kollektiv!) |
| `win` | Handle auf das erzeugte Fenster |

```c
int MPI_Win_free(MPI_Win *win)   // Fenster freigeben
```

**Beispiel:** Nur Prozess 1 stellt Speicher bereit; Prozess 0 meldet ein leeres Fenster an:
```c
if (rank == 0)
    MPI_Win_create(MPI_BOTTOM, 0, sizeof(int), MPI_INFO_NULL, comm, &win);
else if (rank == 1) {
    int *inbuf = malloc(sizeof(int) * n);
    MPI_Win_create(inbuf, n * sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &win);
}
```

#### Alternative: `MPI_Win_allocate` (empfohlen)

```c
int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                     MPI_Comm comm, void *baseptr, MPI_Win *win)
```

Alloziert Speicher und erstellt gleichzeitig das Fenster. MPI kann intern optimieren (z. B. günstigeres Memory-Alignment wählen) und ist daher **in der Regel besser als `MPI_Win_create`**, wenn kein vorhandener Buffer verwendet werden muss.

---

### 6. RMA-Operationen: Put, Get, Accumulate

#### `MPI_Put` – Schreiben in entfernten Speicher

```c
int MPI_Put(const void *origin_addr, int origin_count,
            MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype,
            MPI_Win win)
```

Der **lokale Prozess** schreibt Daten aus `origin_addr` in das Fenster von `target_rank` ab Offset `target_disp`.

#### `MPI_Get` – Lesen aus entferntem Speicher

```c
int MPI_Get(void *origin_addr, int origin_count,
            MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype,
            MPI_Win win)
```

Liest Daten aus dem Fenster von `target_rank` in den lokalen Buffer `origin_addr`.

#### `MPI_Accumulate` – Atomare Remote-Operation

```c
int MPI_Accumulate(const void *origin_addr, int origin_count,
                   MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp,
                   int target_count, MPI_Datatype target_datatype,
                   MPI_Op op, MPI_Win win)
```

Wie `MPI_Put`, aber mit einer **Reduktionsoperation** (`MPI_SUM`, `MPI_MIN`, `MPI_AND`, …). Mehrere `Accumulate`-Operationen auf denselben Speicher sind erlaubt – im Gegensatz zu überlappenden `Put`-Operationen. `MPI_REPLACE` entspricht einem direkten `Put`, erlaubt aber Überlappungen.



### 7. Synchronisation: Active Target (Fence)

RMA-Operationen sind **asynchron** – sie kehren zurück, bevor die Daten tatsächlich übertragen wurden. Ohne Synchronisation ist der Zustand des Ziel-Speichers undefiniert.

#### `MPI_Win_fence` – Kollektive Barriere

```c
int MPI_Win_fence(int assert, MPI_Win win)
```

* **Kollektiv:** Alle Prozesse des Kommunikators, der das Fenster erzeugt hat, müssen `MPI_Win_fence` aufrufen.
* Beendet alle RMA-Operationen seit dem letzten Fence.
* Stellt sicher, dass auch **lokale Schreiboperationen** auf Fenster-Speicher für RMA sichtbar sind (z. B. `A[4] = 4`, wenn `A` im Fenster liegt).
* `assert = 0` ist immer korrekt. Optimierungshinweise (können die Performance verbessern):

| Flag | Bedeutung |
|---|---|
| `MPI_MODE_NOSTORE` | Lokaler Fensterspeicher wurde seit dem letzten Fence nicht lokal beschrieben |
| `MPI_MODE_NOPUT` | Nach diesem Fence kommen keine `Put`/`Accumulate`-Zugriffe auf dieses Fenster |
| `MPI_MODE_NOPRECEDE` | Dieser Fence schließt keine vorherigen RMA-Aufrufe ab |
| `MPI_MODE_NOSUCCEED` | Dieser Fence startet keine neuen RMA-Aufrufe |

**Weil Fence kollektiv ist, muss das Ziel immer mitsynchronisieren.** Diese Art nennt sich daher **Active Target Synchronisation**.

#### Typisches Fence-Muster:
```c
MPI_Win_create(A, ..., &win);
MPI_Win_fence(0, win);              // Epoch öffnen

if (rank == 0) {
    MPI_Put(..., win);              // Daten schreiben
    MPI_Put(..., win);
}
MPI_Win_fence(0, win);              // Epoch schließen – alle Puts sind fertig

MPI_Get(..., win);                  // Daten lesen
MPI_Win_fence(0, win);              // Epoch schließen

A[rank] = 4;                        // Lokaler Schreibzugriff auf Fenster-Speicher
MPI_Win_fence(0, win);              // Muss getrennt werden!

MPI_Put(..., win);
MPI_Win_fence(0, win);
```

---
#### Praxisbeispiel: Matrix-Vektor-Multiplikation

Jeder Prozess `i` besitzt einen Streifen der Matrix und berechnet lokal seinen Teilbeitrag `t[i]` zum Ergebnisvektor `w`. Anschließend akkumuliert jeder Prozess seinen Anteil direkt per RMA in den Fensterspeicher der anderen Prozesse (Beitrag zu `w[j]` auf Prozess `j`):

```
w[0] = A[0][0]*v[0] + A[0][1]*v[1] + ... + A[0][n]*v[n]
     = t[0][0]       + t[1][0]       + ... + t[p][0]
```
Jeder Prozess liefert einen lokalen Teilergebnis-Block `t`, der per `MPI_Accumulate(..., MPI_SUM, win)` zu `w` auf dem Ziel-Prozess addiert wird:

```c
double t[m], w[m/p];
MPI_Win win;
MPI_Win_create(w, m/p * sizeof(double), sizeof(double),
               MPI_INFO_NULL, MPI_COMM_WORLD, &win);

// Lokale Matrix-Vektor-Multiplikation → füllt t[]
compute_local(t);

MPI_Win_fence(0, win);

for (int i = 0; i < p; i++) {
    if (i != my_rank)
        MPI_Accumulate(&t[i * (m/p)], m/p, MPI_DOUBLE,
                       i, 0, m/p, MPI_DOUBLE, MPI_SUM, win);
}

MPI_Win_fence(0, win);
// w[] enthält nun das vollständige Teilergebnis dieses Prozesses
```

Das Elegante an dieser Lösung: Es braucht kein explizites `MPI_Reduce` – jeder Prozess schreibt seinen Beitrag direkt in den Speicher desjenigen Prozesses, dem das Ergebnis gehört.

---

### 8. Fallen bei RMA (Undefined Behavior)

RMA hat strikte Regeln. Verstöße führen zu **Undefined Behavior** – kein Compile-Fehler, aber undefinierte Ergebnisse:

```c
double b[10];
for (i = 0; i < 10; i++) b[i] = rank * 10.0 + i;
MPI_Win_create(b, 10*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

MPI_Win_fence(0, win);
if (rank == 0) {
    b[2] = 1./3.;                               // Lokaler Schreibzugriff
}
else if (rank == 1) {
    MPI_Put(b, 10, MPI_DOUBLE, 0, 0, 10, MPI_DOUBLE, win);  // RMA auf denselben Speicher
}
MPI_Win_fence(0, win);
// → UNDEFINED BEHAVIOR! Lokales Schreiben und MPI_Put überlappen sich.
```

Das ist kein theoretisches Problem: `MPI_Put` überträgt Daten **nicht zwingend atomar**. Ein `double` (8 Byte) kann wortweise übertragen werden. Wenn Prozess 0 lokal `b[2] = 1./3.` schreibt (Bitmuster `0x3FD5555555555555`) und gleichzeitig Prozess 1 via `MPI_Put` `b[2] = 12.0` schreibt (Bitmuster `0x4028000000000000`), kann der Speicher am Ende **eine Mischung beider Bitmuster** enthalten – weder `1./3.` noch `12.0`, sondern ein korrumpierter Gleitkommazahl-Wert. Das ist der eigentliche Grund, warum lokale Schreibzugriffe und `MPI_Put` auf denselben Fensterspeicher durch `MPI_Win_fence` getrennt werden müssen.

#### Einfache Regeln zur Vermeidung:

| Situation | Regel |
|---|---|
| Zwei `Put`-Operationen im selben Fence-Abschnitt | Ziel-Bereiche dürfen **nicht überlappen** |
| Mehrere `Accumulate`-Operationen auf denselben Speicher | Erlaubt, aber Datentyp **und** Operation müssen identisch sein |
| Lokales Schreiben + `MPI_Put`/`MPI_Accumulate` | Müssen durch `MPI_Win_fence` getrennt werden |
| Lokales Lesen + `MPI_Get` | Erlaubt, solange nicht gleichzeitig geschrieben wird |

---

### 9. Synchronisation: PSCW (Post/Start/Complete/Wait)

PSCW ist der Mittelweg zwischen Fence und Lock/Unlock: nicht so kollektiv wie Fence, aber strukturierter als Lock. Die Idee: **nur die tatsächlich beteiligten Prozesse** synchronisieren miteinander.

PSCW besteht aus zwei Seiten:

**Ziel-Seite (Target) – stellt Fenster bereit:**
```c
MPI_Win_post(MPI_Group origin_group, int assert, MPI_Win win)
// ... Ziel kann weiterrechnen ...
MPI_Win_wait(MPI_Win win)
```

**Ursprungs-Seite (Origin) – greift auf Fenster zu:**
```c
MPI_Win_start(MPI_Group target_group, int assert, MPI_Win win)
MPI_Put(..., win)   // oder Get / Accumulate
MPI_Win_complete(MPI_Win win)
```

**Ablauf:**
1. Das Ziel ruft `MPI_Win_post` auf → meldet, dass sein Fenster für eine Gruppe von Ursprungs-Prozessen zugänglich ist.
2. Der Ursprung ruft `MPI_Win_start` auf → wartet, bis das Ziel `Post` ausgeführt hat, und startet dann die RMA-Zugriffe.
3. Der Ursprung ruft `MPI_Win_complete` auf → beendet alle seine RMA-Operationen.
4. Das Ziel ruft `MPI_Win_wait` auf → wartet, bis alle `Complete`-Aufrufe der Ursprungs-Gruppe eingetroffen sind.

**Vorteil gegenüber Fence:** Nur die tatsächlich kommunizierenden Prozesse synchronisieren – kein kollektiver Overhead über den gesamten Kommunikator.

Leider ist es am Ende doch nicht so einfach, wie es zunächst erscheint, denn hier braucht man MPI-Gruppen:
#### Einschub: MPI Gruppen in der PSCW-Synchronisation
In der Welt von MPI (Message Passing Interface) ist ein Kommunikator (wie `MPI_COMM_WORLD`) das soziale Netzwerk, während eine Gruppe (`MPI_Group`) lediglich die Mitgliederliste dieses Netzwerks darstellt. Bei der PSCW-Synchronisation sind diese Gruppen das entscheidende Werkzeug für die Performance.

1. Was ist eine MPI-Gruppe?
Eine Gruppe ist eine geordnete Menge von Prozess-IDs (Ranks). Im Gegensatz zu einem Kommunikator kann man mit einer Gruppe allein keine Nachrichten versenden. Sie dient lediglich dazu, Teilmengen von Prozessen zu definieren.

2. Warum sind Gruppen für PSCW zwingend?
Das PSCW-Modell ist für die aktive Ziel-Synchronisation gedacht. Während andere Methoden (wie `MPI_Win_fence`) den **Holzhammer**  nutzen und das gesamte System anhalten, funktioniert PSCW wie ein Skalpell. Die Gruppen sind hierbei notwendig, um die Kopplung zu definieren:

**Eingrenzung der Beteiligten:** Durch Gruppen weiß das System genau, welche Prozesse an einer Transaktion beteiligt sind. Wenn Prozess A nur Daten in Prozess B schreibt, müssen Prozesse C bis Z nichts davon wissen und können ungehindert weiterrechnen.

**Vermeidung von Deadlocks:** Die Gruppen legen fest, wer auf wen wartet. Das Ziel (Target) gibt mit `MPI_Win_post` eine Gruppe von erlaubten Sendern frei. Der Ursprung (Origin) gibt mit `MPI_Win_start` eine Gruppe von Zielen an, auf die er zugreifen möchte.

**Ressourceneffizienz:**  Die Hardware kann die Netzwerkverbindung spezifisch zwischen den Gruppenmitgliedern optimieren, anstatt globale Barrieren aufzubauen.

3. Das Prinzip der "Matching Groups"
Damit PSCW funktioniert, müssen die Gruppen korrespondieren:

Die Gruppe, die beim Target im `MPI_Win_post`  angegeben wird, muss den Rank des Origins enthalten.

Die Gruppe, die beim Origin im `MPI_Win_start` angegeben wird, muss den Rank des Targets enthalten.

#### Wie werden die Gruppen erzeugt?
Hier ist der Standard-Workflow, um eine Gruppe für PSCW zu erstellen:

1. Die Basis-Gruppe holen: Man kann keine Gruppe aus dem Nichts erschaffen. Man extrahiert zuerst die Gruppe aus einem bestehenden Kommunikator.
2. Die Teil-Gruppe definieren: Man gibt an, welche Ranks (Prozess-Nummern) in die neue Gruppe sollen.
3. Die neue Gruppe erstellen: Mit Funktionen wie `MPI_Group_incl` (einschließen) oder `MPI_Group_excl` (ausschließen) erzeugt man das neue Gruppen-Objekt

```c
MPI_Group world_group, target_group;
int target_rank = 1; // Der Prozess, mit dem ich reden will

// 1. Die Gruppe aller Prozesse holen
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

// 2. & 3. Eine neue Gruppe erstellen, die nur den 'target_rank' enthält
// Parameter: (Basisgruppe, Anzahl der Ranks, Array der Ranks, Neue Gruppe)
MPI_Group_incl(world_group, 1, &target_rank, &target_group);

// Jetzt kann man  target_group in MPI_Win_start verwenden:
// MPI_Win_start(target_group, 0, win);
```


#### Zusammenfassung
Ohne Gruppen wäre PSCW nicht "punktgenau". Erst durch die Definition dieser Teilmengen wird die einseitige Kommunikation (RMA) wirklich effizient, da sie die Synchronisation auf das absolut notwendige Minimum reduziert – nämlich genau auf die Prozesse, die Daten austauschen.

**Merksatz:** In PSCW definieren Gruppen das „Wer mit Wem“, damit das System nicht fragen muss „Alle auf Einmal?“.
#### Beispiel für PSCW Synchronisation

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int window_buffer = 0;
    int put_data = 100;
    MPI_Win win;

    // 1. Fenster erstellen
    MPI_Win_create(&window_buffer, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Gruppen definieren (Wer spricht mit wem?)
    MPI_Group world_group, origin_group, target_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    int origin_rank = 0;
    int target_rank = 1;
    MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
    MPI_Group_incl(world_group, 1, &target_rank, &target_group);

    if (rank == 1) { // DAS ZIEL (Target)
        // POST: Signalisiert, dass der Ursprung (0) jetzt starten darf
        MPI_Win_post(origin_group, 0, win);
        
        // WAIT: Wartet, bis der Ursprung mit 'COMPLETE' signalisiert hat
        MPI_Win_wait(win);
        printf("Prozess 1: Daten erhalten! Wert im Fenster ist nun: %d\n", window_buffer);
    } 
    else if (rank == 0) { // DER URSPRUNG (Origin)
        // START: Beginnt die Zugriffsphase auf das Ziel (1)
        MPI_Win_start(target_group, 0, win);
        
        // Eigentlicher Datentransfer
        MPI_Put(&put_data, 1, MPI_INT, 1, 0, 1, MPI_INT, win);
        
        // COMPLETE: Beendet den Zugriff und signalisiert dem Ziel 'Fertig'
        MPI_Win_complete(win);
        printf("Prozess 0: Daten übertragen und Phase abgeschlossen.\n");
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
```

### 10. Passive Target Synchronisation

`MPI_Win_fence` ist kollektiv, alle Prozesse eines Kommunikators müssen es aufrufen. Bei der PSCW Synchronisation muss auch das **Ziel** immer einen Aufruf ausführen. Das ist für viele Szenarien unnötig aufwändig. **Passive Target Synchronisation** ermöglicht RMA-Zugriffe, ohne dass der Ziel-Prozess aktiv synchronisiert.

#### `MPI_Win_lock` / `MPI_Win_unlock`

```c
int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)
int MPI_Win_unlock(int rank, MPI_Win win)
```

* `MPI_Win_lock` **startet** eine RMA-Zugriffsperiode auf das Fenster von Prozess `rank`.
* `MPI_Win_unlock` **beendet** sie und stellt sicher, dass alle RMA-Operationen abgeschlossen sind.
* **`MPI_Win_lock` blockiert nicht** – der Aufruf kehrt sofort zurück (asynchron).

| `lock_type` | Bedeutung |
|---|---|
| `MPI_LOCK_EXCLUSIVE` | Nur ein Prozess hat Zugriff|
| `MPI_LOCK_SHARED` | Mehrere Prozesse können gleichzeitig lesend zugreifen; überlappende Schreibzugriffe bleiben verboten |

**Einfaches Beispiel – blockierendes Put:**
```c
int Blocking_put(const void *buf, int count, MPI_Datatype dtype,
                 int target_rank, MPI_Aint target_offset,
                 int target_count, MPI_Datatype target_dtype, MPI_Win win)
{
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
    int err = MPI_Put(buf, count, dtype, target_rank, target_offset,
                      target_count, target_dtype, win);
    MPI_Win_unlock(target_rank, win);
    return err;
}
```

#### Vergleich der drei RMA-Synchronisationsmechanismen

| Mechanismus | Kollektiv? | Ziel muss aktiv mitmachen? | Typischer Einsatz |
|---|---|---|---|
| **Fence** | Ja (alle im Kommunikator) | Ja | Einfache, reguläre Muster; alle Prozesse gleich aktiv |
| **PSCW** | Nein (nur definierte Gruppen) | Ja (`Post`/`Wait`) | Wenn nur Teilmengen von Prozessen kommunizieren |
| **Lock/Unlock** | Nein | Nein (passiv) | Asynchroner Zugriff, Mutexe, globale Zähler |

---

### 11. Atomare Operationen: `MPI_Fetch_and_op` und `MPI_Win_flush`

#### Das Problem: Reihenfolge von RMA-Operationen ist unbestimmt

Mehrere RMA-Operationen innerhalb einer Lock/Unlock-Periode werden **nicht garantiert in Reihenfolge** ausgeführt. Folgender Code für einen globalen Zähler ist **falsch**:

```c
// FALSCH (Versuch 1): Get und Accumulate können in beliebiger Reihenfolge ankommen
MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
MPI_Get(&value, 1, MPI_INT, rank, idx, 1, MPI_INT, win);
MPI_Accumulate(&one, 1, MPI_INT, rank, idx, 1, MPI_INT, MPI_SUM, win);
MPI_Win_unlock(rank, win);
```

Ein naiver zweiter Versuch – `Get` und `Accumulate` in **getrennten** Lock/Unlock-Paaren – ist ebenfalls falsch:

```c
// FALSCH (Versuch 2): MPI_Win_lock blockiert nicht!
MPI_Win_lock(MPI_LOCK_EXCLUSIVE, size-1, 0, counterWin);  // globaler Lock

MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, counterWin);
MPI_Get(&value, 1, MPI_INT, rank, idx, 1, MPI_INT, counterWin);
MPI_Win_unlock(rank, counterWin);

MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, counterWin);
MPI_Accumulate(&one, 1, MPI_INT, rank, idx, 1, MPI_INT, MPI_SUM, counterWin);
MPI_Win_unlock(rank, counterWin);

MPI_Win_unlock(size-1, counterWin);
```

**Warum falsch?** `MPI_Win_lock` **blockiert nicht** – der Aufruf kehrt sofort zurück, ohne zu warten, bis der Lock tatsächlich gewährt wurde. Der Code gibt daher keine echte Serialisierung zwischen `Get` und `Accumulate`. (Einzige Ausnahme: wenn `rank == size-1`, weil dann der exklusive Lock und der Shared Lock denselben Prozess betreffen.)

**`MPI_MODE_NOCHECK`** als `assert`-Argument bei `MPI_Win_lock` ist ein Performance-Hint: Der Aufrufer garantiert, dass kein konkurrierender Lock auf diesem Fenster existiert – die Laufzeit muss das nicht überprüfen. Nur verwenden, wenn man sicher ist, dass kein Konflikt auftreten kann.

#### Lösung 1: `MPI_Fetch_and_op` – atomares Fetch-and-Modify

```c
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                     MPI_Datatype datatype,
                     int target_rank, MPI_Aint target_disp,
                     MPI_Op op, MPI_Win win)
```

Liest den aktuellen Wert aus dem Fenster (`result_addr`) und wendet `op` atomar an – in **einer** garantiert unteilbaren Operation:

```c
int one = 1;
MPI_Win_lock(MPI_LOCK_SHARED, lrank, 0, counterWin);
MPI_Fetch_and_op(&one, &value, MPI_INT, lrank, lidx, MPI_SUM, counterWin);
MPI_Win_unlock(lrank, counterWin);
// value enthält den alten Zählerstand, Zähler wurde um 1 erhöht
```

#### Lösung 2: `MPI_Win_flush` – erzwinge Abschluss einzelner Operationen

```c
int MPI_Win_flush(int rank, MPI_Win win)
```

Beendet alle bisher ausgegebenen RMA-Operationen auf das Fenster von `rank`, **ohne** die Zugriffsperiode (Lock/Unlock) zu beenden. Damit kann man innerhalb einer Periode die Reihenfolge kontrollieren.

#### Beispiel: RMA-Mutex mit Spin-Wait
```c
MPI_Win_lock(MPI_LOCK_SHARED, lrank, 0, mutex_win);
do {
    MPI_Fetch_and_op(&one, &oldval, MPI_INT, lrank, lidx, MPI_SUM, mutex_win);
    MPI_Win_flush(lrank, mutex_win);    // warte, bis Fetch abgeschlossen
    if (oldval == 0) break;             // Mutex war frei → wir haben ihn
    // Mutex war belegt → rückgängig machen und erneut versuchen
    MPI_Accumulate(&mone, 1, MPI_INT, lrank, lidx, 1, MPI_INT, MPI_SUM, mutex_win);
    MPI_Win_flush(lrank, mutex_win);
} while (1);
MPI_Win_unlock(lrank, mutex_win);

// kritischer Abschnitt ...

MPI_Win_lock(MPI_LOCK_SHARED, lrank, 0, mutex_win);
MPI_Accumulate(&mone, 1, MPI_INT, lrank, lidx, 1, MPI_INT, MPI_SUM, mutex_win);
MPI_Win_unlock(lrank, mutex_win);       // Mutex freigeben
```

---

### 12. MPI Memory Model: *Separate* vs. *Unified*

Bevor RMA-Daten korrekt ausgetauscht werden können, muss klar sein, wann lokale Schreibzugriffe auf den Fensterspeicher für andere Prozesse sichtbar werden. MPI definiert dafür zwei Speichermodelle:

#### Separate Memory Model (Standard bei verteiltem Speicher)

Der Fensterspeicher existiert in zwei „Sichten": der **öffentlichen** (was andere Prozesse per RMA sehen) und der **privaten** (was der lokale Prozess per Load/Store sieht). Beide Sichten sind **nicht automatisch synchron**.

* Lokale Schreibzugriffe (`b[2] = 4.0`) sind zunächst nur in der privaten Sicht sichtbar.
* Erst nach einem Synchronisationspunkt (Fence, Complete, Unlock, Flush) werden lokale Änderungen in die öffentliche Sicht übernommen – und damit für RMA-Zugriffe anderer Prozesse sichtbar.
* Das ist der Grund für die Regel aus Abschnitt 8: **Lokales Schreiben und `MPI_Put` auf denselben Speicher müssen durch `MPI_Win_fence` getrennt werden.**

#### Unified Memory Model (typisch bei Shared Memory / `MPI_Win_allocate_shared`)

Private und öffentliche Sicht sind **identisch** – lokale Stores sind sofort durch das Fenster sichtbar, ohne explizite Synchronisation der Sichten. Es sind aber weiterhin **Synchronisationspunkte** nötig, um sicherzustellen, dass der andere Prozess die Daten tatsächlich liest (Speicherbarrieren, `MPI_Barrier`, `__sync()`).

Das Modell eines Fensters kann zur Laufzeit abgefragt werden:
```c
int *model;
MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
// *model == MPI_WIN_SEPARATE oder MPI_WIN_UNIFIED
```

**Faustregel:** Bei `MPI_Win_allocate_shared` (Shared Memory) immer *unified* annehmen – aber dennoch Barrieren setzen. Bei allen anderen Fenstertypen auf verteiltem Speicher immer *separate* annehmen und alle lokalen Schreibzugriffe explizit durch Synchronisation publizieren.

---

### 13. MPI Shared Memory

Für MPI-Prozesse, die auf **demselben Knoten** laufen, unterstützt MPI eine spezielle Variante: RMA-Fenster im gemeinsamen physischen Speicher (Shared Memory). Der Datenaustausch erfolgt dann direkt per **Load/Store** – ohne Netzwerk-Overhead.

#### Kommunikator auf einen Knoten einschränken

```c
MPI_Comm shmcomm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
```

`MPI_Comm_split_type` mit `MPI_COMM_TYPE_SHARED` erzeugt für jeden physischen Knoten einen eigenen Sub-Kommunikator, der nur die lokalen Prozesse enthält.

#### Shared-Memory-Fenster anlegen

```c
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info,
                             MPI_Comm comm, void *baseptr, MPI_Win *win)
```

* `size`: Lokale Fenstergröße dieses Prozesses (darf `0` sein).
* Alle lokalen Größen zusammen bilden das gemeinsame Fenster.

#### Zeiger auf andere Prozess-Teile abfragen

```c
int MPI_Win_shared_query(MPI_Win win, int rank,
                          MPI_Aint *size, int *disp_unit, void *baseptr)
```

* `rank`: Prozess, dessen Fensterteil abgefragt wird.
* `MPI_PROC_NULL` als `rank` liefert den **Basis-Pointer** des gesamten gemeinsamen Fensters.

#### Vollständiges Beispiel
```c
MPI_Comm shmcomm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

MPI_Win win;
double *mem;
MPI_Win_allocate_shared(local_size * sizeof(double), sizeof(double),
                         MPI_INFO_NULL, shmcomm, &mem, &win);

MPI_Aint sz; int dispunit;
double *ptr;
MPI_Win_shared_query(win, myrank, &sz, &dispunit, &ptr);       // eigener Teil

double *base_ptr;
MPI_Win_shared_query(win, MPI_PROC_NULL, &sz, &dispunit, &base_ptr); // Basis

// Daten initialisieren
for (int i = 0; i < local_size; ++i) ptr[i] = ...;

__sync();                  // Hardware-Speicherbarriere: alle Schreibvorgänge sichtbar
MPI_Barrier(shmcomm);      // Alle Prozesse warten

// Auf gemeinsamen Daten arbeiten
... = base_ptr[...];

MPI_Win_free(&win);
```

#### Shared Memory in einer Stencil-Berechnung
```c
double *topptr, *bottomptr;
MPI_Win_shared_query(win, top,    &sz, &dispunit, &topptr);
MPI_Win_shared_query(win, bottom, &sz, &dispunit, &bottomptr);

for (int iter = 0; iter < niters; ++iter) {
    MPI_Win_fence(0, win);   // Synchronisations-Epoch

    if (top != MPI_PROC_NULL)
        for (int i = 0; i < bx; ++i)
            a2[ind(i+1, 0)] = topptr[ind(i+1, by)];      // Halo von oben

    if (bottom != MPI_PROC_NULL)
        for (int i = 0; i < bx; ++i)
            a2[ind(i+1, by+1)] = bottomptr[ind(i+1, 1)]; // Halo von unten

    update(&a2);
}
```

#### Hinweis: Speicheradressen bei Shared Memory

Die physische Adresse desselben Fensters ist auf verschiedenen Prozessen **nicht dieselbe** – jeder Prozess sieht das Shared-Memory-Window unter seiner eigenen virtuellen Adresse. `MPI_Win_shared_query` liefert jeweils den richtigen lokalen Pointer. Der `base_ptr` (via `MPI_PROC_NULL`) zeigt auf den Anfang des zusammenhängenden Shared-Blocks.

Mit der Option `alloc_shared_noncontig=True` kann der Speicher **nicht-zusammenhängend** alloziert werden – sinnvoll auf **NUMA-Systemen**, damit jeder Prozess primär auf seinen lokalen NUMA-Knoten zugreift und teure Fernzugriffe vermieden werden.

---

### 14. Ausblick: OpenSHMEM — Reines einseitiges Kommunikationsmodell

MPI RMA ist eine Erweiterung eines primär zweiseitigen Modells. **OpenSHMEM** ist dagegen von Grund auf als **reines einseitiges Kommunikationsmodell** konzipiert — einfacher in der API, konsequenter im Programmiermodell.

#### Grundprinzip: PGAS

OpenSHMEM basiert auf dem **Partitioned Global Address Space (PGAS)**-Modell: Alle Prozesse (Processing Elements, PEs) teilen sich einen gemeinsamen logischen Adressraum, der physisch auf die PEs verteilt ist.

```
PE 0          PE 1          PE 2
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Symmetric│  │ Symmetric│  │ Symmetric│  ← gemeinsam adressierbar
│ Heap     │  │ Heap     │  │ Heap     │
├──────────┤  ├──────────┤  ├──────────┤
│ Private  │  │ Private  │  │ Private  │  ← nur lokal
└──────────┘  └──────────┘  └──────────┘
```

Der **Symmetric Heap** ist der Schlüssel: Objekte, die kollektiv auf allen PEs mit `shmem_malloc` alloziert werden, sind unter derselben logischen Adresse auf jedem PE erreichbar. Kein explizites Window-Management wie bei MPI RMA nötig.

#### Grundlegende API

```c
#include <shmem.h>

// Initialisierung
shmem_init();
int me    = shmem_my_pe();   // wie MPI_Comm_rank
int npes  = shmem_n_pes();   // wie MPI_Comm_size

// Symmetrischer Speicher allokieren (kollektiv auf allen PEs!)
double *data = (double*) shmem_malloc(n * sizeof(double));

// Einseitiges Schreiben: lokale Daten → PE target
shmem_double_put(data,        // Ziel (symmetrische Adresse auf target)
                 local_buf,   // Quelle (lokaler Buffer)
                 n,           // Anzahl Elemente
                 target);     // Ziel-PE

// Einseitiges Lesen: PE source → lokale Daten
shmem_double_get(local_buf,   // Ziel (lokaler Buffer)
                 data,        // Quelle (symmetrische Adresse auf source)
                 n,           // Anzahl Elemente
                 source);     // Quell-PE

// Synchronisation
shmem_quiet();          // warte auf alle ausstehenden Puts/Gets dieses PE
shmem_barrier_all();    // globale Barriere über alle PEs

// Aufräumen
shmem_free(data);
shmem_finalize();
```

#### Atomare Operationen

OpenSHMEM bietet direkte atomare Fetch-and-Modify-Operationen, ohne separates Lock/Unlock:

```c
// Atomares Fetch-and-Add (entspricht MPI_Fetch_and_op mit MPI_SUM)
long old_val = shmem_long_atomic_fetch_add(counter,  // symm. Zeiger
                                            1,        // Inkrement
                                            target);  // Ziel-PE
```

#### Vergleich: MPI RMA vs. OpenSHMEM

| Eigenschaft | MPI RMA | OpenSHMEM |
|---|---|---|
| **Speicherverwaltung** | Explizites Window (`MPI_Win_create`) | Symmetrischer Heap (`shmem_malloc`) |
| **Synchronisation** | Fence / PSCW / Lock+Unlock | `shmem_quiet`, `shmem_barrier_all` |
| **Atomare Ops** | `MPI_Fetch_and_op` | `shmem_TYPE_atomic_fetch_*` |
| **API-Komplexität** | Höher (Window-Konzept) | Niedriger (symmetrischer Heap) |
| **Interoperabilität** | Teil von MPI | Eigenständig; OpenSHMEM+MPI möglich |
| **GPU-Variante** | — | **NVSHMEM** (siehe Multi-GPU-Einheit) |

> **Ausblick — letzte Kurseinheit (Multi-GPU):** Die Bibliothek **NVSHMEM** ist die GPU-zentrische Weiterentwicklung von OpenSHMEM. Sie implementiert dieselbe PGAS-Idee — symmetrischer Heap, Put/Get, Barrieren — aber für NVIDIA-GPU-Cluster, mit der Erweiterung, dass `nvshmem_put/get` direkt aus CUDA-Kerneln aufgerufen werden kann. In der letzten Kurseinheit (*Multi-GPU Computing*) wird NVSHMEM ausführlich behandelt, inklusive symmetrischem Speichermodell, API-Beispielen und dem Vergleich mit CUDA-Aware MPI und NCCL.

---

### 15. Zusammenfassung

| Konzept | Kernidee | Schlüssel-API |
|---|---|---|
| **Non-blocking Kollektive** | Rechnen und Kommunizieren überlappen | `MPI_I*` + `MPI_Wait` |
| **Persistente Kommunikation** | Setup-Overhead einmalig, Ausführung wiederholt | `MPI_Send_init`, `MPI_Start` |
| **RMA / Einseitige Kommunikation** | Schreiben/Lesen ohne aktives Ziel | `MPI_Put`, `MPI_Get`, `MPI_Accumulate` |
| **Active Target Sync (Fence)** | Kollektive Barriere, einfach und sicher | `MPI_Win_fence` |
| **Passive Target Sync (Lock)** | Ziel muss nicht mitsynchronisieren | `MPI_Win_lock/unlock` |
| **Atomare Operationen** | Garantierte Reihenfolge bei concurrent access | `MPI_Fetch_and_op`, `MPI_Win_flush` |
| **Shared Memory** | Direkte Load/Store auf gemeinsamem Speicher | `MPI_Win_allocate_shared`, `MPI_Win_shared_query` |
