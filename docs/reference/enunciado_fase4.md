# Enunciado — Tech Challenge Fase 4 (resumo de referência)

Pós-graduação IA para Devs (FIAP). Vale 90% da nota da fase. Entrega em grupo (aqui, individual).

## Contexto

Um hospital que já usa IA para laudos e exames quer **monitorar pacientes de forma contínua** com dados multimodais (áudio, vídeo, texto) para detectar sinais precoces de risco, integrando serviços gerenciados em nuvem (Azure Cognitive Services).

Além de processar laudos e exames, o sistema passará a:
- Analisar vídeos de cirurgias ou sessões de fisioterapia para identificar padrões anômalos;
- Processar gravações de voz de pacientes em consultas, detectando sintomas relacionados à fala (fadiga, disartria);
- Detectar anomalias em sinais vitais, prescrições e evolução clínica, alertando a equipe médica em tempo real;
- Integrar tudo isso com serviços gerenciados em nuvem.

## Objetivos

- Realizar a análise e fusão de diferentes tipos de dados médicos (texto, áudio, vídeo);
- Utilizar serviços em nuvem para ampliar a capacidade de processamento e inteligência;
- Aplicar técnicas de detecção de anomalias em tempo real para monitoramento preventivo.

## Requisitos técnicos

1. **Análise de vídeo** — processar vídeos clínicos (fisioterapia/cirurgias); detectar movimentos ou eventos fora do padrão com modelos como OpenPose (postura) e YOLOv8 (objetos/áreas críticas); gerar relatórios automáticos de desvios ou falhas.
2. **Análise de áudio** — processar áudios de consultas; detectar alterações vocais (cansaço, dificuldade respiratória, fadiga, disartria); usar Azure Speech to Text para transcrição; identificar termos críticos e sentimentos com Azure Text Analytics.
3. **Detecção de anomalias** — em séries temporais de sinais vitais (batimentos, pressão, oxigenação), evolução de prescrições (mudanças inesperadas) e padrões de movimentação do paciente na internação; alertas automáticos à equipe médica.

Datasets sugeridos: https://physionet.org/ e https://research.google.com/audioset/

## Entregáveis

- Repositório Git com código-fonte completo.
- Relatório técnico com: descrição do fluxo multimodal, modelos aplicados por tipo de dado, resultados e exemplos de anomalias detectadas.
- Vídeo de até 15 min (YouTube/Vimeo, público ou não listado) demonstrando: análise de áudio e vídeo, detecção e resposta a anomalias, integração dos serviços Azure, fluxo final do alerta à equipe médica.

## Como este projeto mapeia os requisitos

| Requisito | Implementação |
|---|---|
| Vídeo: OpenPose / YOLOv8 | `src/video_analysis/` — YOLOv8n + YOLOv8n-pose (equivalente funcional ao OpenPose) |
| Áudio: Speech to Text + Text Analytics | `src/audio_analysis/` — Azure Speech, Azure AI Language, librosa, modelo SPIRA |
| Anomalias: vitais, prescrições, movimentação | `src/anomaly_detection/` |
| Alertas em tempo real | `src/fusion/monitoring_flow.py` (LangGraph) + dashboard Streamlit |
| Relatório técnico | `docs/relatorio_tecnico.md` |
