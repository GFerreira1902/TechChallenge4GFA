# Monitoramento Multimodal de Pacientes

[![Testes](https://github.com/GFerreira1902/TechChallenge4GFA/actions/workflows/tests.yml/badge.svg)](https://github.com/GFerreira1902/TechChallenge4GFA/actions/workflows/tests.yml)

**Tech Challenge — Fase 4 · Pós-graduação IA para Devs (FIAP)**
Guilherme Ferreira de Arruda · RM 373210 · Grupo 47

Protótipo de um sistema hospitalar que acompanha pacientes internados combinando **vídeo** (câmera do quarto ou sessão de fisioterapia), **voz** (gravações de consulta) e **dados clínicos** (sinais vitais e prescrições). Cada fonte é analisada por um modelo específico; as evidências são fundidas em um **score de risco** e, quando necessário, um **alerta é emitido para a equipe médica** — sempre com aviso de validação humana e registro de auditoria.

| Entregável                                                 | Onde                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------ |
| Relatório técnico (fluxo, modelos, resultados, limitações) | [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md) |
| Diagramas do fluxo multimodal                              | [docs/diagrama_fluxo.md](docs/diagrama_fluxo.md)       |
| Vídeo de demonstração (YouTube)                            | https://youtu.be/gcieGLWFtVA                           |

---

## 1. O que o sistema faz

```
 vídeo ─────► YOLOv8n (objetos) + YOLOv8n-pose (postura) ──► queda · postura · imobilidade · área crítica ──┐
 voz ───────► Azure Speech to Text ──► Azure AI Language (sentimento, termos críticos)                     │
         └──► librosa (pausas, tosse) + modelo de dispneia treinado no SPIRA                                ├─► LangGraph ─► score 0-100 ─► alerta + relatório
 vitais ────► limites clínicos + z-score móvel + IsolationForest                                           │                            └─► auditoria JSON
 prescrições► regras de salto/suspensão de dose ───────────────────────────────────────────────────────────┘
```

| Requisito do enunciado                                                                                                        | Como foi atendido                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Analisar vídeos clínicos com OpenPose / YOLOv8 e gerar relatórios de desvios                                                  | YOLOv8n + YOLOv8n-pose (equivalente funcional ao OpenPose, 17 pontos do corpo). Detecta queda, postura anômala, imobilidade e entrada em área crítica. Gera **PDF** com linha do tempo e quadros anotados.                                              |
| Transcrever áudio com Azure Speech to Text; detectar alterações vocais; termos críticos e sentimento com Azure Text Analytics | Transcrição pt-BR real; sentimento, frases-chave e entidades via Azure AI Language; lista própria de termos críticos em 3 níveis; medidas acústicas (librosa); **modelo de dificuldade respiratória** treinado em 432 áudios reais do SPIRA (AUC 0,82). |
| Anomalias em sinais vitais, prescrições e movimentação; alertas automáticos                                                   | Três detectores combinados para FC/PA/SpO2, regras de dose, trajetória do paciente extraída do vídeo. Alerta auditado com disclaimer obrigatório.                                                                                                       |
| Fusão multimodal e integração com nuvem                                                                                       | Grafo LangGraph com estado compartilhado e score consolidado; clientes Azure com **mock mode** (funciona sem chave).                                                                                                                                    |

Resultados nos dados reais: 2/2 quedas detectadas e 0 falsos positivos em atividades normais (UR Fall); transcrição idêntica à referência (Common Voice); AUC 0,82 em dispneia (SPIRA, split de teste); recall 94% com 0 falsos positivos em vitais sintéticos e 0 alertas indevidos em pacientes reais estáveis (PhysioNet BIDMC). Detalhes no [relatório](docs/relatorio_tecnico.md).

---

## 2. Rodando localmente

### 2.1 Pré-requisitos

| Item            | Versão   | Observação                                                                                         |
| --------------- | -------- | -------------------------------------------------------------------------------------------------- |
| Python          | **3.11** | testado em 3.11.1 (Windows 11)                                                                     |
| Git             | qualquer |                                                                                                    |
| GPU NVIDIA      | opcional | acelera o YOLO (~40 ms/quadro na GTX 1650). Sem GPU roda em CPU (~0,5 s/quadro)                    |
| Conta Azure     | opcional | recursos **Speech** e **Language** no tier gratuito F0. Sem chaves o sistema roda em **mock mode** |
| Espaço em disco | ~4 GB    | venv com torch (2,5 GB CUDA ou 200 MB CPU) + amostras (~400 MB)                                    |

### 2.2 Instalação (Windows / PowerShell)

```powershell
git clone https://github.com/GFerreira1902/TechChallenge4GFA.git
cd TechChallenge4GFA
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch: escolha UMA linha
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cpu     # sem GPU

pip install -r requirements.txt
```

Linux/macOS: troque `py -3.11 -m venv .venv` por `python3.11 -m venv .venv` e a ativação por `source .venv/bin/activate`.

### 2.3 Chaves Azure (opcional, mas recomendado para ver a integração real)

```powershell
copy .env.example .env
```

Edite o `.env`:

```
AZURE_SPEECH_KEY=<chave do recurso Speech>
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=<chave do recurso Language>
AZURE_LANGUAGE_ENDPOINT=https://<nome-do-recurso>.cognitiveservices.azure.com/
```

Como obter: portal Azure → _Criar recurso_ → **Speech** (tier F0) e **Language** (tier F0) → em cada um, _Chaves e Ponto de Extremidade_. O endpoint do Language deve ser o personalizado do recurso (`https://<nome>.cognitiveservices.azure.com/`), não o regional genérico.

Sem o `.env` tudo funciona: transcrição e sentimento voltam simulados e a auditoria marca `simulated: true`. O dashboard mostra na barra lateral se cada serviço está em modo REAL ou SIMULADO.

### 2.4 Dados de exemplo

```powershell
python scripts/download_samples.py
```

Baixa automaticamente (~150 MB): sinais vitais reais do **PhysioNet BIDMC**, 44 clipes de tosse/respiração do **ESC-50** e 4 vídeos do **UR Fall Detection** (2 quedas, 2 atividades normais), convertendo-os para MP4.

Dois conjuntos exigem download manual (login) e são **opcionais** — sem eles, os pacientes PAC-001/002/003 rodam sem a etapa de áudio e o PAC-005 sem voz:

| Dataset             | Para quê                                     | Como obter                                                                                                       | Onde colocar                                                                         |
| ------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **SPIRA** (USP)     | voz real de pacientes com falta de ar, pt-BR | link do Google Drive em https://github.com/Edresson/SPIRA-ACL2021 → extrair                                      | `data/samples/audio/spira/` (pastas `pacientes/`, `controle/` e os `metadata_*.csv`) |
| **Common Voice PT** | voz saudável de referência                   | https://commonvoice.mozilla.org/pt/datasets → 22 clipes listados em `data/samples/audio/commonvoice/selecao.csv` | `data/samples/audio/commonvoice/clips/`                                              |

Detalhes, licenças e citações em [data/samples/README.md](data/samples/README.md).

### 2.5 Verificar a instalação

```powershell
python -m pytest tests/ -q
```

Esperado: `59 passed`. Os testes não precisam de GPU, chaves Azure nem datasets.

---

## 3. Executando

> **Console Windows:** antes de rodar, execute `chcp 65001` e `$env:PYTHONIOENCODING="utf-8"` para exibir acentos corretamente.

### 3.1 Demonstração completa no terminal (~40 s)

```powershell
python -m src.demo --paciente PAC-003
```

Executa, em sequência, os quatro blocos do enunciado para o paciente escolhido: **(1)** análise de áudio com Azure, **(2)** análise de vídeo com YOLO, **(3)** fusão das modalidades e score de risco, **(4)** alerta final e trilha de auditoria. Use `--rapido` para pular 1 e 2 e ver só a fusão.

### 3.2 Dashboard web

```powershell
streamlit run src/dashboard/app.py
```

Abre em `http://localhost:8501`. Escolha o paciente na barra lateral e clique **Rodar fluxo completo** (10–15 s). Abas: _Alertas_, _Sinais vitais_ (com botão **Simular stream** que mostra o alerta surgindo conforme os dados chegam), _Áudio_ (player, transcrição, sentimento, probabilidade de dispneia), _Vídeo_ (player, quadros anotados, eventos), _Auditoria_.

### 3.3 Apenas o fluxo de fusão

```powershell
python -m src.fusion.monitoring_flow --paciente-id PAC-004 [--sem-audio] [--sem-video]
```

### 3.4 Pacientes disponíveis

| ID          | Cenário                                                   | O que esperar de resultado                                     |
| ----------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| **PAC-003** | Insuficiência cardíaca descompensada — **caso principal** | score 100, crítico: vitais + prescrição + áudio apontam juntos |
| **PAC-004** | Queda no quarto                                           | score 60, crítico: queda detectada no vídeo aos 2,6 s          |
| PAC-001     | Dor torácica                                              | alto: dispneia na fala, pico de FC, medicamento novo           |
| PAC-002     | Sepse                                                     | alto: PA cai, antibiótico trocado                              |
| **PAC-005** | Idoso estável (controle)                                  | **score 0, nenhum alerta**                                     |

Pacientes são fictícios; as mídias associadas vêm dos datasets públicos acima.

### 3.5 Onde ficam as saídas

| Arquivo                                         | Conteúdo                                                                                  |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `outputs/alerts/monitoramento_<ID>_<data>.json` | relatório consolidado: score, nível, alertas por modalidade, resumo de áudio/vídeo/vitais |
| `outputs/reports/video_<ID>_<data>.pdf`         | relatório automático de vídeo com linha do tempo e quadros anotados                       |
| `outputs/figures/*.jpg`                         | quadros anotados (caixas YOLO + esqueleto)                                                |
| `outputs/audit_log.json`                        | toda chamada Azure/YOLO e todo alerta, com data, metadados e flag real/simulado           |

### 3.6 Docker (alternativa, CPU)

```bash
docker build -t tc4-monitoramento .
docker run -p 8501:8501 --env-file .env -v ./data/samples:/app/data/samples tc4-monitoramento
```

Sobe o dashboard em `http://localhost:8501` com torch CPU. Monte `data/samples` como volume para que os dados baixados fiquem disponíveis dentro do container.

---

## 4. Scripts auxiliares

| Script                               | Para quê                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `scripts/download_samples.py`        | baixa BIDMC, ESC-50 e UR Fall                                              |
| `scripts/train_dyspnea_model.py`     | retreina o modelo de dispneia no SPIRA e gera as curvas ROC (requer SPIRA) |
| `scripts/gerar_figuras_relatorio.py` | regenera todas as figuras usadas no relatório técnico                      |

## 5. Testes e integração contínua

59 testes `pytest` cobrem cada módulo e o grafo de fusão, usando stubs para YOLO e Azure. O GitHub Actions ([tests.yml](.github/workflows/tests.yml)) executa a suíte a cada push em `main` usando `requirements-ci.txt` (torch CPU).

## 6. Estrutura do projeto

```
src/
├── anomaly_detection/   vital_signs · prescriptions · movement · alerts (estrutura de alerta e score)
├── audio_analysis/      acoustic_features · azure_speech_client · azure_language_client · dyspnea_model · audio_analyzer
├── video_analysis/      frame_extractor · object_detector · pose_analyzer · video_report · video_analyzer
├── fusion/              monitoring_flow — grafo LangGraph
├── dashboard/           app (Streamlit) · helpers
├── guardrails/          audit_logger · safety_rules (disclaimer obrigatório, detecção de prescrição direta)
├── langchain_pipeline/  patient_records (prontuário sintético) · llm_client (Groq, opcional, não usado na demo)
└── demo.py
data/raw/synthetic_patients.json   pacientes fictícios com configuração de monitoramento
data/samples/                      amostras públicas (não versionadas)
scripts/                           download · treino · figuras
tests/                             59 testes
docs/                              relatório técnico · diagramas · figuras · enunciado
```

## 7. Solução de problemas

| Sintoma                                              | Causa provável                                                           | Solução                                                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `torch.cuda.is_available()` é `False` com GPU NVIDIA | torch CPU instalado pelo `requirements.txt`                              | `pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124`           |
| Acentos quebrados no terminal                        | console em cp1252                                                        | `chcp 65001` e `$env:PYTHONIOENCODING="utf-8"`                                                                           |
| `[Language] 401 Access denied`                       | endpoint regional genérico no `.env`                                     | use o endpoint personalizado `https://<recurso>.cognitiveservices.azure.com/`                                            |
| Azure Speech `SPXERR_INVALID_HEADER`                 | arquivo não é WAV PCM                                                    | o cliente converte automaticamente; se persistir, verifique se `librosa`/`soundfile` estão instalados                    |
| Vídeo não toca no dashboard                          | MP4 em codec `mp4v`                                                      | `scripts/download_samples.py` já gera H.264; para vídeos próprios, reencode com ffmpeg (`-c:v libx264 -pix_fmt yuv420p`) |
| Paciente sem etapa de áudio                          | SPIRA/Common Voice não baixados                                          | ver seção 2.4; o fluxo continua e registra o aviso em `erros`                                                            |
| Primeira execução lenta (~6 s no primeiro quadro)    | download dos pesos `yolov8n.pt` e `yolov8n-pose.pt` e aquecimento da GPU | normal; pesos ficam em `models/`                                                                                         |
