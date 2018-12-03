FROM registry.gitlab.com/salus/builder

COPY tensorflow tensorflow

ENV TENSORFLOW_ROOT=${SALUS_WORK_ROOT}/tensorflow
ENV TensorFlow_DIR=${TENSORFLOW_ROOT}

COPY ./*.whl /tmp/

RUN pip --no-cache-dir install /tmp/*.whl
