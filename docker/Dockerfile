FROM registry.gitlab.com/salus/builder

COPY tensorflow tensorflow
COPY ./*.whl /tmp/

RUN pip --no-cache-dir install /tmp/*.whl