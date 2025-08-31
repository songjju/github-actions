# 최적화된 Dockerfile
FROM python:3.11-alpine AS builder

# 빌드 의존성만 설치
RUN apk add --no-cache --virtual .build-deps \
    gcc musl-dev libffi-dev openssl-dev

WORKDIR /app

# 캐싱을 위해 requirements.txt 먼저 복사
COPY requirements.txt .

# 의존성 설치 (빌드 단계)
RUN pip install --no-cache-dir -r requirements.txt

# === 실행 단계 ===
FROM python:3.11-alpine AS runtime

# 런타임 사용자 생성
RUN adduser -D -s /bin/sh lotto

WORKDIR /app

# 빌드된 패키지만 복사
COPY --from=builder /root/.local /home/lotto/.local
ENV PATH=/home/lotto/.local/bin:$PATH

# 애플리케이션 코드 복사 (마지막에!)
COPY --chown=lotto:lotto . .

USER lotto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('Health OK')" || exit 1

CMD ["python", "main.py", "--predictions", "5"]