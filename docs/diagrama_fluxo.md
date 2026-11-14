# Diagrama do fluxo multimodal

## Visão geral — das fontes ao alerta

```mermaid
flowchart LR
    subgraph Fontes["Fontes de dados (por paciente)"]
        V[("Vídeo\nUR Fall / mp4")]
        A[("Áudio\nSPIRA / Common Voice / ESC-50")]
        S[("Sinais vitais\nsintético / PhysioNet BIDMC")]
        P[("Prescrições\nprontuário sintético")]
    end

    subgraph Video["video_analysis"]
        V --> FE[frame_extractor\nOpenCV 10 fps]
        FE --> OD[object_detector\nYOLOv8n]
        FE --> PA[pose_analyzer\nYOLOv8n-pose · 17 pontos]
        OD & PA --> EV[eventos\nqueda · postura · imobilidade · ROI]
        EV --> PDF[video_report\nPDF com frames anotados]
    end

    subgraph Audio["audio_analysis"]
        A --> AF[acoustic_features\nlibrosa · pausas · tosse]
        A --> ST[azure_speech_client\nSpeech to Text pt-BR]
        ST --> LG[azure_language_client\nsentimento · frases-chave · termos críticos]
        A --> DM[dyspnea_model\nregressão logística · SPIRA]
    end

    subgraph Anom["anomaly_detection"]
        S --> VS[vital_signs\nlimites · z-score · IsolationForest]
        P --> PR[prescriptions\nsalto de dose · suspensão]
        EV --> MV[movement\ntrajetória do centroide]
    end

    EV & AF & LG & DM & VS & PR & MV --> F{{"fusion/monitoring_flow\nLangGraph"}}
    F --> SC[score de risco 0-100]
    SC -->|"≥ 30 ou crítico"| AL[["emitir_alerta\n+ enforce_human_validation"]]
    SC -->|"< 30"| RL
    AL --> RL[gerar_relatorio\nJSON em outputs/alerts]
    AL & RL & ST & LG & OD --> AU[(AuditLogger\noutputs/audit_log.json)]
    RL --> DB[/Dashboard Streamlit\nou demo.py/]
```

## O grafo LangGraph em detalhe

```mermaid
stateDiagram-v2
    [*] --> receber_paciente
    receber_paciente --> coletar_vitais : prontuário + config de monitoramento
    coletar_vitais --> analisar_prescricoes : alertas de vitais
    analisar_prescricoes --> analisar_audio : + alertas de prescrição
    analisar_audio --> analisar_video : + alertas de áudio (Azure / librosa / SPIRA)
    analisar_video --> consolidar_anomalias : + alertas de vídeo e movimento
    consolidar_anomalias --> emitir_alerta : score ≥ 30 ou algum alerta crítico
    consolidar_anomalias --> gerar_relatorio : score < 30
    emitir_alerta --> gerar_relatorio : mensagem auditada + disclaimer
    gerar_relatorio --> [*]
```

**Estado compartilhado** (`MonitoringState`, um `TypedDict`): `paciente_id`, `paciente`, `monitoramento`, `vitais_df`, `laudo_audio`, `laudo_video`, `alertas: list[Alerta]`, `score_risco`, `nivel_risco`, `deve_alertar`, `mensagem_alerta`, `relatorio`, `erros`.

**`Alerta`**: `paciente`, `modalidade` (vitais | prescricao | audio | video | movimento), `severidade` (baixa | media | alta | critica), `evidencia`, `timestamp`.

## Score de risco

```
peso(severidade)  = baixa 1 · media 2 · alta 4 · critica 12
bonus             = 1 + 0,25 × (nº de modalidades distintas − 1)
score             = min(100, soma dos pesos × bonus × 5)

nível: 0 normal · 1–29 moderado · 30–59 alto · 60–100 crítico
alerta é emitido se score ≥ 30 OU existe alerta crítico
```

## Guardrails em todo caminho de saída

```mermaid
flowchart LR
    X[texto de alerta / relatório] --> G{enforce_human_validation}
    G -->|detecta "tome", "administre", "prescrevo"...| W[aviso de segurança registrado]
    G --> D[anexa disclaimer de validação humana]
    D --> L[(AuditLogger)]
```
