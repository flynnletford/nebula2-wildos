FROM ros:humble-ros-core-jammy

# Install build essentials and development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-foxglove-bridge \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only PyTorch to avoid CUDA downloads)
# Use numpy<2 for compatibility with cv_bridge
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    opencv-python \
    'numpy<2' \
    timm==1.0.15 \
    open-clip-torch==2.32.0 \
    transformers==4.51.3 \
    einops==0.8.1 \
    fairscale==0.4.4 \
    scikit-image \
    onnx \
    matplotlib \
    ruamel.yaml==0.18.11 \
    omegaconf \
    folium \
    selenium
   
# Set up ROS2 workspace
WORKDIR /ros2_ws

# Copy model checkpoints into the container
COPY ros2/model_ckpts /ros2_ws/model_ckpts

# Copy parent directory packages (nvidia_radio and explorfm)
COPY nvidia_radio /tmp/nvidia_radio
COPY explorfm /tmp/explorfm_raw

# Restructure explorfm to be a proper python package layout
RUN mkdir -p /tmp/explorfm/explorfm && \
    cp -r /tmp/explorfm_raw/* /tmp/explorfm/explorfm/ && \
    mv /tmp/explorfm/explorfm/setup.py /tmp/explorfm/ && \
    mv /tmp/explorfm/explorfm/README.md /tmp/explorfm/ || true

# Install nvidia_radio and explorfm from parent directories
RUN pip install /tmp/nvidia_radio /tmp/explorfm

RUN pip install --no-cache-dir rosbags

RUN apt-get update && apt-get install -y \
   ros-humble-ros2bag \
   ros-humble-rosbag2-transport \
   ros-humble-rosbag2-storage-default-plugins

# Copy ROS2 source code
COPY ros2/src /ros2_ws/src

# Build ROS2 packages with colcon
# Build explorfm_msgs first, then other packages
RUN . /opt/ros/humble/setup.sh && colcon build --packages-up-to explorfm_msgs
RUN . /opt/ros/humble/setup.sh && . /ros2_ws/install/setup.sh && colcon build

# Source ROS2 and built packages in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /etc/bash.bashrc

# Create startup script that ensures proper sourcing
RUN echo '#!/bin/bash' > /ros2_ws/entrypoint.sh && \
    echo 'set -e' >> /ros2_ws/entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /ros2_ws/entrypoint.sh && \
    echo 'source /ros2_ws/install/setup.bash' >> /ros2_ws/entrypoint.sh && \
    echo 'exec "$@"' >> /ros2_ws/entrypoint.sh && \
    chmod +x /ros2_ws/entrypoint.sh

# Use the startup script as entrypoint
ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["/bin/bash", "-i"]
