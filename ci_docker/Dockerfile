
FROM python:3.10-buster as builder


RUN useradd -rm -d /home/user -u 1001 user && \
    mkdir -p /home/user/app && \
    chown -R user:user /home/user/app
USER user



WORKDIR /home/user/ 


COPY app.py poetry.lock pyproject.toml app/
# COPY .env app/
COPY backend_service/ app/backend_service/


WORKDIR /home/user/app

# SPACE to run Deployment specific poetry commands for config other packages
# RUN --mount=type=secret, id=any_PYPI_URL --mount=type=secret, id=any_PYPI_TOKEN \
# pip install poetry==1.4.2 --no-cache-dir && \
# export PATH="$(PATH)":"$(HOME)"/.local/bin && \
# poetry config virtualenvs.create true

# RUN --mount=type=secret, id=any_PYPI_URL --mount=type=secret, id=any_PYPI_TOKEN uid=1001 \

RUN pip install --upgrade pip && \
    pip install --upgrade pip && \
    pip install poetry==1.6.1 --user --no-cache-dir && \
    export PATH="$(PATH)":"$(HOME)"/.local/bin && \
    poetry config virtualenvs.in-project true && \
    # poetry config repositories.any_PYPI_URL.url $(cat /run/secrets/any_PYPI_URL) && \
    # poetry config http-basic.any_PYPI_URL $(cat /run/secrets/any_PYPI_URL) $(cat /run/secrets/any_PYPI_TOKEN) && \
    poetry install --only main


FROM python:3.10-buster

RUN useradd -rm -d /home/user -u 1001 user && \
    mkdir -p /home/user/app && \
    chown -R user:user /home/user/app
USER user

COPY --from=builder /home/user/app/ /home/user/app/
ENV PATH=/home/user/app/.venv/bin:$PATH
ENV PYTHONUNBUFFERED 1

WORKDIR /home/user/app

# CMD ["uvicorn", "backend_service.main:app", "--root-path", "/api/v1", "--host", "0.0.0.0" , "--port", "8000"]

CMD ["uvicorn", "backend_service.main:app", "--host", "0.0.0.0" , "--port", "8000"]


