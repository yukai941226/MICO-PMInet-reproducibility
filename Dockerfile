FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps .

COPY configs ./configs
COPY data ./data
COPY reference ./reference
COPY tests ./tests
COPY run.py LICENSE CITATION.cff DATA_DICTIONARY.md QUICKSTART_ZH.md VALIDATION.md MODEL_ARCHITECTURE.md ARTICLE_ALIGNMENT.md DATA_AND_CODE_AVAILABILITY.md RELEASE_NOTES.md ./

CMD ["python", "run.py", "verify"]
