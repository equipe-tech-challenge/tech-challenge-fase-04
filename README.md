# Women Health AI

Plataforma para registro e acompanhamento de atendimentos com analise automatizada de video e audio clinico. O sistema recebe arquivos, processa com modelos de IA e gera um relatorio em PDF para apoio a decisao profissional.

## Principais funcionalidades
- API FastAPI para submissao de jobs e consulta de status
- Worker Celery para processamento assincrono
- Pipeline de IA para video (cirurgia, consulta, fisioterapia) e audio
- Geracao automatica de relatorios em PDF
- Persistencia de jobs e resultados em banco SQL (SQLite por padrao)

## Arquitetura em alto nivel
- API: `api/main.py` e `api/routes/jobs.py`
- Worker: `workers/celery_app.py` e `workers/tasks/process_job.py`
- IA (video e audio): `ai/video` e `ai/audio`
- Banco: `database/db.py` e `database/models.py`
- Relatorios: `reports/pdf_report.py`
- Storage: `storage/` com `uploads/`, `results/`, `reports/`

## Fluxo de processamento
1. Cliente envia arquivo para a API.
2. A API cria o job no banco e enfileira a tarefa no Celery.
3. O worker executa o pipeline de IA.
4. O resultado e eventos sao salvos no banco.
5. Um PDF e gerado com resumo do job.

## Modulos de IA
### Video
- Cirurgia: deteccao de sangramento com YOLOv8 (`ai/video/yolo_bleeding`)
- Consulta: analise nao-verbal com MediaPipe (`ai/video/consultation/nonverbal_analyzer.py`)
- Fisioterapia: analise de movimento com MediaPipe (`ai/video/physiotherapy/physio_analyzer.py`)

### Audio
- Analise de hesitacao, ansiedade e trauma (proxies) com Librosa
- Pipeline: `ai/audio/extract_audio.py` + `ai/audio/consultation/audio_analyzer.py`

## Endpoints da API
Base URL: `http://localhost:8000`

### POST `/jobs/video`
Cria um job de video. O parametro `job_request` seleciona o tipo de analise.

Valores aceitos:
- `Surgery: bleeding detection`
- `Consultation: non-verbal`
- `Physiotherapy: movement`
- `Video audio analysis: hesitation, anxiety and trauma`

Exemplo (multipart):
```bash
curl -X POST "http://localhost:8000/jobs/video?job_request=Surgery:%20bleeding%20detection" \
  -F "file=@/caminho/video.mp4"
```

### POST `/jobs/audio`
Cria um job de audio. Espera arquivo `.wav`.
```bash
curl -X POST "http://localhost:8000/jobs/audio" \
  -F "file=@/caminho/audio.wav"
```

### GET `/jobs/{job_id}`
Retorna status, resultados e disponibilidade do relatorio.

### GET `/jobs/{job_id}/report`
Baixa o PDF do relatorio do job.

## Storage e artefatos
- Uploads: `storage/uploads/`
- Resultados: `storage/results/<job_id>/`
- Relatorios: `storage/reports/Report_WomenHealth_<job_id>.pdf`

## Variaveis de ambiente
- `REDIS_URL` (default: `redis://localhost:6379/0`)
- `DATABASE_URL` (default: `sqlite:////app/storage/app.db`)
- `STORAGE_ROOT` (default: `/app/storage`)

## Executando com Docker
```bash
docker compose -f docker/docker-compose.yml up --build
```

Servicos:
- API: `http://localhost:8000`
- Redis: `localhost:6379`

## Executando localmente
Requisitos: Python 3.11, ffmpeg, Redis

```bash
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Worker:
```bash
celery -A workers.celery_app worker --loglevel=INFO
```

## Treinamento YOLOv8 (sangramento)
Scripts em `ai/video/yolo_bleeding/`:
- `train.py`: treina modelo e salva em `ai/video/yolo_bleeding/model.pt`
- `validate.py`: valida o modelo treinado
- `export_onnx.py`: exporta para ONNX
- Dataset e `data.yaml`: `ai/video/yolo_bleeding/data/`

## Observacoes tecnicas
- O pipeline de audio usa proxies e nao gera diagnostico clinico.
- O relatorio PDF e um apoio ao profissional de saude.
- Videos anotados sao gerados em `storage/results/<job_id>/`.
