# === 빌드 단계 ===
FROM python:3.11-slim AS builder

# 🔥 핵심: setuptools 먼저 업그레이드 (취약점 해결)
RUN pip install --upgrade pip && \
    pip install --force-reinstall --no-cache-dir "setuptools>=70.0.0" "wheel>=0.42.0"

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 먼저 복사
COPY requirements.txt .

# 🚀 개선: setuptools 버전 확인 후 의존성 설치
RUN echo "🔍 Builder setuptools version:" && pip show setuptools | grep Version && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# 설치 후 검증
RUN echo "✅ Build completed. Installed packages:" && \
    ls -la /install/lib/python*/site-packages/ | head -10

# === 실행 단계 ===
FROM python:3.11-slim AS runtime

# 🔥 핵심: Runtime에서도 setuptools 업그레이드
RUN pip install --upgrade pip && \
    pip install --force-reinstall --no-cache-dir "setuptools>=78.1.1" "wheel>=0.42.0"

# 런타임 의존성 설치 (빌드 도구는 제외)
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libffi8 \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 런타임 사용자 생성
RUN useradd -m -s /bin/bash lotto

WORKDIR /app

# 빌드된 패키지 복사
COPY --from=builder /install /usr/local

# 🚀 개선: 필요한 디렉토리 사전 생성
RUN mkdir -p \
    data/raw \
    data/processed \
    outputs/predictions \
    outputs/reports \
    outputs/visualizations \
    logs \
    && chown -R lotto:lotto /app

# 애플리케이션 코드 복사
COPY --chown=lotto:lotto . .

# 🔍 런타임 setuptools 버전 확인
RUN echo "🎯 Runtime setuptools version:" && pip show setuptools | grep Version

USER lotto

# 환경변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 🚀 개선: setuptools 버전 검증 포함
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import setuptools; print(f'Setuptools: {setuptools.__version__}'); assert setuptools.__version__ >= '70.0.0', 'Setuptools version too old'" || exit 1

CMD ["python", "main.py", "--predictions", "5"]