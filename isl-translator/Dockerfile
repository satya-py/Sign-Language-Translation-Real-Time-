# The backend, ready to run anywhere that takes a container (Render, Fly, a VM).
#
# The build context is this whole folder, not backend/, because the API imports
# the recogniser from live/ and the fingerspelling models from fingerspell/.
#
# It is worth knowing before deploying: MediaPipe builds its tracking graph
# through XNNPACK, which needs a few hundred megabytes in one piece, and when it
# cannot get them it aborts the process rather than raising. Together with torch
# this image needs about 1 GB of memory to answer its first frame. A 512 MB
# instance will start, serve /api/health, and then die on the first camera frame
# with "failed to setup XNNPACK runtime".

FROM python:3.12-slim

# libGL and libglib are what OpenCV asks for at import time; mediapipe depends on
# the GUI build of OpenCV whether or not a screen exists. Without these the first
# import fails with "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not rebuild the 200 MB torch layer.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Only what the server actually reads. The dataset videos are not in the
# repository (11 GB); the Learn page falls back to its poster images, which are.
COPY live/ live/
COPY fingerspell/ fingerspell/
COPY backend/ backend/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    # CPU-only, single-threaded: on a small instance the extra threads cost
    # memory and win nothing on a model this size.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 8000
WORKDIR /app/backend

# The frontend is deployed separately (Vercel), so this only serves the API; if
# frontend/dist is ever copied in, the same process serves the page too.
CMD ["python", "app.py"]
