# Monitoramento Multimodal de Pacientes — Tech Challenge Fase 4

[![Testes](https://github.com/GFerreira1902/TechChallenge4GFA/actions/workflows/tests.yml/badge.svg)](https://github.com/GFerreira1902/TechChallenge4GFA/actions/workflows/tests.yml)

Pós-graduação **IA para Devs** (FIAP) — Guilherme Ferreira de Arruda, RM 373210, Grupo 88.

Protótipo de um sistema hospitalar que acompanha pacientes juntando **vídeo**, **voz** e **dados clínicos** (sinais vitais e prescrições), detecta anomalias em cada modalidade, funde as evidências em um score de risco e **alerta a equipe médica** automaticamente — sempre com validação humana obrigatória e auditoria completa.

> Projeto independente que herda padrões da Fase 3 (guardrails, auditoria, mock mode, prontuário sintético, fluxo LangGraph) — mesmo modelo adotado da Fase 2 para a 3.

## Entregáveis

| Item | Onde |
|---|---|
| Código-fonte completo | este repositório |
| Relatório técnico | [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md) |
| Diagrama do fluxo multimodal | [docs/diagrama_fluxo.md](docs/diagrama_fluxo.md) |
| Vídeo de demonstração (≤ 15 min) | _link será adicionado na entrega_ |

## O que o sistema faz

| Requisito do enunciado | Como foi atendido | Módulo |
|---|---|---|
| Análise de vídeo (OpenPose / YOLOv8), relatórios automáticos de desvios | YOLOv8n (objetos) + YOLOv8n-pose (postura, equivalente funcional ao OpenPose); detecta queda, postura anômala, imobilidade e área crítica; gera PDF com timeline e frames anotados | `src/video_analysis/` |
| Análise de áudio: alterações vocais, Azure Speech to Text, Azure Text Analytics | Transcrição pt-BR real, sentimento + frases-chave + termos críticos (Azure AI Language), medidas acústicas (librosa) e modelo de dispneia treinado no dataset SPIRA | `src/audio_analysis/` |
| Anomalias em vitais, prescrições e movimentação; alertas automáticos | Limites clínicos + z-score + IsolationForest; regras de dose; trajetória do paciente; alerta auditado com disclaimer | `src/anomaly_detection/` |
| Fusão multimodal e integração com nuvem | Grafo LangGraph com score de risco consolidado; clientes Azure com mock mode | `src/fusion/` |
| Interface | Dashboard Streamlit + demo CLI | `src/dashboard/`, `src/demo.py` |

## Arquitetura

```
vídeo ──► YOLOv8 + pose ──► eventos ─┐
voz   ──► Azure Speech ──► Azure Language ──┤
      └─► librosa + modelo SPIRA ───────────┼──► LangGraph ──► score ──► alerta + relatório
vitais ─► limites + z-score + IsolationForest┤                          └──► AuditLogger
prescrições ─► regras de dose ───────────────┘
```

Detalhes e diagramas Mermaid em [docs/diagrama_fluxo.md](docs/diagrama_fluxo.md).

## Dados

Pacientes fictícios (PAC-001..005) em `data/raw/synthetic_patients.json`, cada um ligado a amostras **reais** de datasets públicos:

| Modalidade | Fonte | Licença |
|---|---|---|
| Voz de pacientes com insuficiência respiratória (PT-BR) | [SPIRA – USP/HC-FMUSP](https://github.com/Edresson/SPIRA-ACL2021) | CC BY-SA 4.0 (uso local, não redistribuído) |
| Voz saudável (PT) | [Mozilla Common Voice 27.0](https://commonvoice.mozilla.org/pt) | CC0 |
| Tosse / respiração | [ESC-50](https://github.com/karolpiczak/ESC-50) | CC BY-NC 3.0 |
| Vídeo (quedas e atividades normais) | [UR Fall Detection](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) | acadêmico |
| Sinais vitais reais | [PhysioNet BIDMC](https://physionet.org/content/bidmc/1.0.0/) | ODC-By 1.0 |

As amostras não são versionadas; `python scripts/download_samples.py` baixa as de acesso direto (ver [data/samples/README.md](data/samples/README.md)).

## Instalação

Requisitos: Python 3.11, Windows/Linux. GPU NVIDIA opcional (acelera o YOLO).

```powershell
git clone https://github.com/GFerreira1902/TechChallenge4GFA.git
cd TechChallenge4GFA
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU; use /cpu para CPU
pip install -r requirements.txt
copy .env.example .env     # preencher chaves Azure (opcional)
python scripts/download_samples.py
```

### Chaves Azure (opcionais)

Crie recursos **Speech** e **Language** no tier gratuito F0 e preencha o `.env`:

```
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=...
AZURE_LANGUAGE_ENDPOINT=https://<recurso>.cognitiveservices.azure.com/
```

Sem chaves o sistema entra em **mock mode**: continua funcionando com respostas simuladas e marca isso na auditoria.

## Como usar

```powershell
# demonstração completa no terminal (4 blocos: áudio, vídeo, fusão, alerta)
chcp 65001 ; $env:PYTHONIOENCODING="utf-8"
python -m src.demo --paciente PAC-003

# fluxo LangGraph direto
python -m src.fusion.monitoring_flow --paciente-id PAC-004

# dashboard
streamlit run src/dashboard/app.py
```

Saídas em `outputs/`: `alerts/*.json` (relatórios de monitoramento), `reports/*.pdf` (relatórios de vídeo), `figures/` (frames anotados), `audit_log.json`.

### Docker

```bash
docker build -t tc4-monitoramento .
docker run -p 8501:8501 --env-file .env tc4-monitoramento
```

## Testes e CI

```powershell
python -m pytest tests/ -v
```

59 testes cobrindo cada módulo e o grafo de fusão, usando stubs para YOLO e Azure — rodam sem GPU, sem chaves e sem datasets. GitHub Actions executa a suíte a cada push em `main` ([tests.yml](.github/workflows/tests.yml)) com `requirements-ci.txt` (torch CPU).

## Resultados resumidos

| Modalidade | Validação | Resultado |
|---|---|---|
| Vídeo | 4 vídeos UR Fall | 2/2 quedas detectadas, 0 falsos positivos nas atividades normais |
| Áudio – dispneia | SPIRA, split de teste oficial (108 áudios) | AUC 0,82 (CV 0,94) |
| Áudio – transcrição | Common Voice PT | transcrição idêntica à referência |
| Áudio – tosse | ESC-50 | 18/22 detectadas |
| Vitais | sintético com ground truth | recall 94%, 0 falsos positivos |
| Vitais | PhysioNet BIDMC (real) | 0 alertas em pacientes estáveis; 1 queda real de SpO2 detectada |
| Fusão | 5 casos de uso | alerta nos 4 pacientes com problema, silêncio no controle |

Detalhes, figuras e limitações no [relatório técnico](docs/relatorio_tecnico.md).

## Estrutura

```
src/
├── anomaly_detection/   vital_signs · prescriptions · movement · alerts
├── audio_analysis/      acoustic_features · azure_speech_client · azure_language_client · dyspnea_model · audio_analyzer
├── video_analysis/      frame_extractor · object_detector · pose_analyzer · video_report · video_analyzer
├── fusion/              monitoring_flow (LangGraph)
├── dashboard/           app (Streamlit) · helpers
├── guardrails/          audit_logger · safety_rules
├── langchain_pipeline/  patient_records · llm_client (opcional)
└── demo.py
scripts/   download_samples · train_dyspnea_model · gerar_figuras_relatorio
tests/     59 testes pytest
docs/      relatório técnico · diagrama · figuras
```

## Aviso

Sistema acadêmico com dados fictícios e amostras públicas. Não é dispositivo médico. Nenhuma saída deve ser usada para decisão clínica sem validação de um profissional de saúde.
