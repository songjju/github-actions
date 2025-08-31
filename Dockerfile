# 로또 예측 시스템 Dockerfile
FROM python:3.11-slim

# 메타데이터
LABEL maintainer="Lotto Prediction System"
LABEL version="2.0"
LABEL description="천재적 사고 기반 로또 예측 시스템"

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p data/raw data/processed outputs/predictions outputs/reports outputs/visualizations logs

# 비루트 사용자 생성 및 권한 설정
RUN useradd --create-home --shell /bin/bash lotto && \
    chown -R lotto:lotto /app
USER lotto

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 노출 (향후 웹 인터페이스용)
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from src.utils.config import Config; print('Health OK')" || exit 1

# 기본 실행 명령
CMD ["python", "main.py", "--predictions", "5"]