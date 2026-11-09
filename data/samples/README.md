# Amostras de dados publicos (nao versionadas)

Amostras **reduzidas** por limitacao de hardware/tempo; a escolha esta documentada no
relatorio tecnico. Para recriar, execute `python scripts/download_samples.py` ou siga as fontes.

| Pasta | Fonte | Conteudo | Licenca |
| --- | --- | --- | --- |
| `vitals/` | [PhysioNet BIDMC 1.0.0](https://physionet.org/content/bidmc/1.0.0/) | `bidmc_{01,05,09}_Numerics.csv` - HR, SpO2, RESP a 1 Hz (8 min) | ODC-By 1.0 |
| `audio/esc50/` | [ESC-50](https://github.com/karolpiczak/ESC-50) | 22 clipes de tosse (classe 24) e 22 de respiracao (classe 23), 5 s | CC BY-NC 3.0 |
| `audio/commonvoice/` | [Mozilla Common Voice 27.0 PT](https://commonvoice.mozilla.org/pt/datasets) (via Mozilla Data Collective) | `clips/` 22 clipes mp3 de fala real (falantes 30-70+ anos) + `selecao.csv` com transcricao de referencia, idade e sexo | CC0 |
| `audio/spira/` | [SPIRA - USP/HC-FMUSP](https://github.com/Edresson/SPIRA-ACL2021) (Casanova et al., ACL 2021) | 219 audios de pacientes com insuficiencia respiratoria + 213 controles, PT-BR, frase padronizada; metadados com `nivel_falta_de_ar`. **Uso local apenas, sem redistribuicao** (ver ressalva no relatorio) | CC BY-SA 4.0 (README do repo) |
| `video/` | [UR Fall Detection](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) | `fall-01`, `fall-05` (quedas) e `adl-01`, `adl-05` (atividades normais), cam0 RGB 640x480 @30fps convertidos para mp4 | uso academico |
| `video/` | [Pexels](https://www.pexels.com/search/videos/physiotherapy/) | 1-2 videos CC0 de exercicio/fisioterapia (download manual) | CC0 |

Os sinais vitais dos pacientes ficticios PAC-00X sao **sinteticos** (`data/raw/synthetic_patients.json`);
o BIDMC e usado apenas como validacao com dados reais.
