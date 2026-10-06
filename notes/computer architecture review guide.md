
그리고 한 가지 원칙을 먼저 잡자. 아래에서 **필수**라고 적은 PDF도 처음부터 끝까지 전 페이지를 읽겠다는 뜻은 아니야. 각 Module을 시작할 때 내가 필요한 페이지/주제를 골라서 진행하고, 나머지는 과감히 건너뛰는 방식으로 하자.

## 프롤로그 — 전체 Computer Systems 지도만 복원

본격적인 Module 1 전에 **`lecture1-abstraction.pdf`**를 아주 짧게 보는 건 추천해. 이 강의가 컴퓨터구조 전체의 abstraction, CPU, memory, performance 등의 큰 지도를 깔아주는 역할이라서 이후 Module들이 어디에 위치하는지 잡기 좋거든. lecture1-abstraction

여기는 **한 세션을 쓰지 말고 15~30분 정도**면 충분해.

----------

# Module 1. Machine Representation

### “프로그램의 데이터는 실제 memory에서 어떻게 표현되는가?”

### 핵심 PDF

**System Programming**

-   `03.Memory Representation-1.pdf`  03.Memory Representation-1
-   `03.Memory Representation-2.pdf`  03.Memory Representation-2

이 둘이 Module 1의 중심이야. 실제 수업에서도 struct layout, padding/alignment, hexadecimal, byte ordering, sign extension, complement, floating point 등을 하나로 묶어서 다뤘어. 03.Memory Representation-2

### 필요한 부분만 참조

**Computer Architecture**

-   `lecture5-arithmetic(1).pdf` — Arithmetic for Computers (1) lecture5-arithmetic(1)
-   `lecture6-arithmetic(2).pdf` — Arithmetic for Computers (2) lecture6-arithmetic(2)

컴구 Arithmetic 두 개는 **전체 재수강 X**. integer representation, signed/unsigned, overflow, floating point 등 Module 1에서 필요한 부분만 가져오자.

### 이 Module에서 굳이 안 할 것

곱셈/나눗셈 hardware algorithm을 시험 문제 수준으로 반복하거나 IEEE-754 계산을 수십 번 푸는 건 생략.

**예상: 1~2 sessions**

----------

# Module 2. ISA → Function → Stack / ABI

### “C/C++ 함수 호출 하나가 machine level에서는 어떻게 실행되는가?”

여기는 컴구와 시프가 가장 예쁘게 연결되는 부분 중 하나야.

### 핵심 PDF — Computer Architecture

-   `lecture2-instrctions(1).pdf` — Instruction Set Architecture (1) lecture2-instrctions(1)
-   `lecture3-instrctions(2).pdf` — Instruction Set Architecture (2) lecture3-instrctions(2)
-   `lecture4-instrctions(3).pdf` — Instruction Set Architecture (3) lecture4-instrctions(3)

여기서 MIPS 문법을 외우는 게 아니라 register, memory operand, load/store, branch/jump, function call에 필요한 ISA 개념을 복원한다. 실제 Lecture 4에는 MIPS assembly programming까지 수업 범위였어. lecture4-instrctions(3)

### 핵심 PDF — System Programming

-   `09. functions - 1.pdf`  09. functions - 1
-   `09. functions - 2.pdf`  09. functions - 2

첫 파일에서 function call과 automatic variable이 stack을 사용하고 이것이 ABI의 일부라는 데서 시작하고, 09. functions - 1 두 번째에서는 stack frame, `%rsp`, `%rbp`, call/ret까지 내려가. 09. functions - 2

이 Module의 최종 연결은:

```
C function call
→ arguments
→ registers / stack
→ call
→ return address
→ stack frame
→ local variables
→ ret
```

**예상: 2~3 sessions**

----------

# Module 3. Processor & Pipeline

### “Instruction이 실제 CPU 안에서 어떻게 실행되고, 여러 instruction은 어떻게 겹쳐 실행되는가?”

**이번 전체 복습의 핵심 Module #1.**

### 워밍업 / 빠르게

-   `lecture7-performance.pdf` — Performance lecture7-performance
-   `lecture8-logic.pdf` — Logic Design Basics lecture8-logic

Performance는 clock/CPI/execution time 관계를 복원하기 위해 필요하고, Logic은 datapath/control을 이해할 만큼만 본다. Logic gate 설계를 다시 한 학기 수준으로 할 필요는 없어.

### 핵심 PDF

-   `lecture9-processor(1).pdf`  lecture9-processor(1)
-   `lecture10-processor(2).pdf`  lecture10-processor(2)
-   `lecture11-processor(3).pdf` — Multicycle Implementation lecture11-processor(3)
-   `lecture12-processor(4).pdf` — Pipelining #1 lecture12-processor(4)
-   `lecture13-processor(5).pdf` — Pipelining #2 lecture13-processor(5)
-   `lecture14-processor(6).pdf` — Pipelining #3 lecture14-processor(6)

Lecture 9에서 “이제 MIPS의 implementation을 본다”로 넘어가는 구조이고, lecture9-processor(1) 뒤에서는 single-cycle/multicycle을 거쳐 pipelining으로 이어진다. Pipelining의 목적도 여러 instruction을 동시에 실행해 performance를 높이는 것으로 명시되어 있어. lecture12-processor(4)

여기서는 PDF가 많아 보이지만 **6개를 통독하는 게 아니다.** 우리가 핵심 datapath 그림, IF-ID-EX-MEM-WB, single vs multi, pipeline hazards 부분을 뽑아서 하나의 흐름으로 만들 거야.

**예상: 3~4 sessions**

----------

# Module 4. Memory Hierarchy & Virtual Memory

### “CPU가 원하는 data는 어디에 있고, 어떻게 빠르게 가져오는가?”

**전체 복습의 핵심 Module #2.** 연구 관심을 고려하면 여기에는 시간을 아끼지 않는 게 좋아.

### 핵심 PDF — Computer Architecture

-   `lecture15-memory (1).pdf` — Memory Hierarchy: Introduction lecture15-memory (1)
-   `lecture16-memory(2).pdf` — Caches #1 lecture16-memory(2)
-   `lecture17-memory(3).pdf` — Caches #2 lecture17-memory(3)
-   `lecture18-memory(4).pdf` — Virtual Memory #1 lecture18-memory(4)
-   `lecture19-memory(5).pdf` — Virtual Memory #2 lecture19-memory(5)

컴구 쪽에서는 memory access 자체를 주요 performance bottleneck으로 출발하고, lecture15-memory (1) cache → VM → page table/TLB까지 간다. Lecture 19에서는 page table이 physical memory에 있기 때문에 address translation과 data access가 추가 memory access를 유발한다는 문제까지 다뤄. lecture19-memory(5)

### 핵심 PDF — System Programming

-   `08.virtual_memory.pdf`  08.virtual_memory

이건 같은 VM을 **programmer/OS 관점에서 다시 연결하는 용도**로 쓰자. 자료에서도 physical addressing과 현대 시스템의 virtual addressing을 CPU/MMU 구조로 대비하고 있어. 08.virtual_memory

최종 목표:

```
virtual address
→ TLB
→ page table
→ physical address
→ cache
→ DRAM
```

그리고 cache 쪽은 실제 주소를 가지고 tag/index/offset, set, hit/miss 정도는 직접 계산한다.

**예상: 3~4 sessions**

----------

# Module 5. Program → Process → OS

### “내가 만든 source code가 실행 중인 process가 되기까지 무슨 일이 일어나는가?”

이 Module은 **System Programming 중심**이다.

### 가장 핵심

-   `04. process-1.pdf`  04. process-1
-   `04. Process-2.pdf`  04. Process-2

Process-1에서 executable structure, memory layout, process environment를 다루고, 04. process-1 process가 virtual memory를 통해 독립된 memory를 갖고 dedicated CPU에서 실행되는 것처럼 보이는 구조까지 이어져. 04. process-1

### Linking

-   `05.linking-1.pdf`  05.linking-1
-   `05.linking-2.pdf`  05.linking-2

`linking-1`은 **필수**. Source → object → executable 흐름을 복구한다.  
`linking-2`의 library interpositioning 같은 후반 세부 기법은 **빠르게/선택적으로** 보자. 실제 자료도 compile/link/load-run time interposition을 다루고 있어. 05.linking-2

### I/O / IPC

-   `06.input_output.pdf`  06.input_output
-   `7. pipes and redirection.pdf`  7. pipes and redirection

I/O는 C library 밑에서 결국 kernel system call이 동작한다는 연결이 핵심이고, 06.input_output pipe는 IPC와 file descriptor 관점을 잡는 데 사용한다. 자료에서도 pipes/socket/shared memory/signals 등을 IPC로 묶고 있어. 7. pipes and redirection

### 압축해서 볼 것

-   `10. Shell(1).pdf`  10. Shell(1)
-   `11. signals - 1.pdf`  11. signals - 1
-   `11. signals - 2.pdf`  11. signals - 2

Shell은 command 암기가 아니라 **shell이 process를 실행시키는 프로그램이라는 관점**에서만 보고, signals도 signal 번호 암기보다 asynchronous event/IPC/process control에 초점을 둔다. POSIX signal 자체도 IPC이자 concurrency를 만드는 방식으로 소개되어 있어. 11. signals - 1

그래서 이 Module은 파일 수는 많지만 실제 중요도는:

**Process > Linking > I/O/Pipe > Shell/Signals**

순으로 보면 돼.

**예상: 2~3 sessions**

----------

# Module 6. Concurrency & Parallelism

### “여러 execution flow가 같이 움직이면 어떤 문제가 생기는가?”

### 핵심 PDF — System Programming

-   `12.Concurrency.pdf`  12.Concurrency
-   `13.Races and Synchronization-1.pdf`  13.Races and Synchronization-1
-   `13. Races and Synchronization - 2.pdf`  13. Races and Synchronization -…

Concurrency에서는 여러 logical control flow의 execution이 시간적으로 overlap하는 개념부터 시작하고, **single processor에서도 multitasking을 통해 concurrency가 가능하다**는 중요한 구분이 나온다. 12.Concurrency

그 다음 Race/Synchronization에서 race condition → synchronization → pthread → mutex/semaphore/condition variable 등의 흐름으로 간다. 13.Races and Synchronization-1  13. Races and Synchronization -…

### 마지막 연결 — Computer Architecture

-   `lecture20-parallel.pdf` — Parallel Architectures lecture20-parallel

컴구 마지막에서는 single-core performance가 power와 long memory latency에 제약되면서 parallel architecture로 넘어가는 동기로 시작한다. lecture20-parallel

여기서 마지막으로

```
Concurrency
≠ 반드시 simultaneous execution

Parallelism
= 실제 여러 computation의 동시 실행
```

을 연결하고 전체 Systems 복습을 끝내면 돼.

**예상: 2~3 sessions**
