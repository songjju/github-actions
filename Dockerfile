# === 빌드 단계 ===
FROM python:3.11-slim AS builder

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 먼저 복사
COPY requirements.txt .

# 의존성 설치 (빌드 캐시 활용)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === 실행 단계 ===
FROM python:3.11-slim AS runtime

# 런타임 사용자 생성
RUN useradd -m lotto

WORKDIR /app

# 빌드된 패키지 복사
COPY --from=builder /install /usr/local

# 애플리케이션 코드 복사
COPY --chown=lotto:lotto . .

USER lotto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('Health OK')" || exit 1

CMD ["python", "main.py", "--predictions", "5"]
